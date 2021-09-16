import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.resnet_GN_WS as resnet_GN_WS
import networks.layers_WS as L
import networks.resnet_bn as resnet_bn

from networks.lap_pyramid_loss import LapLoss
from networks.spatial_gradient_2d import SpatialGradient
from networks.util import weight_init, norm, ResnetDilated, ResnetDilatedBN 
from networks.ppm import PPM, ASPP


def build_model(args):
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(args)
    net_decoder = builder.build_decoder(args)
    model = MattingModule(args, net_encoder, net_decoder)
    return model


class MattingModule(nn.Module):
    def __init__(self, args, net_enc, net_dec):
        super(MattingModule, self).__init__()
        self.inc = args.arch.n_channel
        self.encoder = net_enc
        self.decoder = net_dec
        self.args = args

    def forward(self, image, two_chan_trimap, image_n, trimap_transformed, smap, inputs=None, is_training=True): 

        if self.inc == 5:
            resnet_input = torch.cat((image_n, two_chan_trimap), 1)
        elif self.inc == 11:
            resnet_input = torch.cat((image_n, trimap_transformed, two_chan_trimap), 1)
        else:
            raise NotImplementedError

        conv_out, indices = self.encoder(resnet_input, return_feature_maps=True, smap=smap)
        out = self.decoder(conv_out, image, indices, two_chan_trimap, smap=smap, inputs=inputs, is_training=is_training)
        return out


class ModelBuilder():
    def build_encoder(self, args):
        if args.arch.encoder == 'resnet50_GN_WS':
            orig_resnet = resnet_GN_WS.__dict__['l_resnet50'](pretrained=True)
            net_encoder = ResnetDilated(args.arch, orig_resnet, dilate_scale=8)
        elif args.arch.encoder == 'resnet50_BN':
            orig_resnet = resnet_bn.__dict__['l_resnet50'](pretrained=True)
            net_encoder = ResnetDilatedBN(args.arch, orig_resnet, dilate_scale=8)
        else:
            raise Exception('Architecture undefined!')

        num_channels = args.arch.n_channel

        if(num_channels > 3):
            print(f'modifying input layer to accept {num_channels} channels')
            net_encoder_sd = net_encoder.state_dict()
            conv1_weights = net_encoder_sd['conv1.weight']

            c_out, c_in, h, w = conv1_weights.size()
            conv1_mod = torch.zeros(c_out, num_channels, h, w)
            conv1_mod[:, :3, :, :] = conv1_weights

            conv1 = net_encoder.conv1
            conv1.in_channels = num_channels
            conv1.weight = torch.nn.Parameter(conv1_mod)

            net_encoder.conv1 = conv1

            net_encoder_sd['conv1.weight'] = conv1_mod

            net_encoder.load_state_dict(net_encoder_sd)
        return net_encoder

    def build_decoder(self, args):
        net_decoder = Decoder(args)
        return net_decoder


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.args = args.arch
        self.batch_norm = True
        middle_chn = 2048

        self.global_module = ASPP(middle_chn, self.args.atrous_rates, self.args.aspp_channel)
        en_chn = middle_chn + self.args.aspp_channel

        self.conv_up1 = nn.Sequential(
            L.Conv2d(en_chn, 256, kernel_size=3, padding=1, bias=True),
            norm(256, self.batch_norm),
            nn.LeakyReLU(),
            L.Conv2d(256, 256, kernel_size=3, padding=1),
            norm(256, self.batch_norm),
            nn.LeakyReLU()
        )

        self.conv_up2 = nn.Sequential(
            L.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=True),
            norm(256, self.batch_norm),
            nn.LeakyReLU()
        )

        self.conv_up3 = nn.Sequential(
            L.Conv2d(256 + 128, 64, kernel_size=3, padding=1, bias=True),
            norm(64, self.batch_norm),
            nn.LeakyReLU()
        )

        self.conv_up4_alpha = nn.Sequential(
            nn.Conv2d(64 + 3 + 3 + 2, 32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=False)
        )

        self.conv_up4_fb = nn.Sequential(
            nn.Conv2d(64 + 3 + 3 + 2, 32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 6, kernel_size=1, padding=0, bias=False)
        )

        self.conv_up4_attn = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, conv_out, img, indices, two_chan_trimap, smap=None, inputs=None, is_training=True):
        conv5 = conv_out[-1]

        global_ctx = self.global_module(conv5)
        x = torch.cat([conv5, global_ctx], 1)

        x = self.conv_up1(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, conv_out[-4]), 1)
        x = self.conv_up2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, conv_out[-5]), 1)
        x = self.conv_up3(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        y = torch.cat((x, conv_out[-6][:, :3], img, two_chan_trimap), 1)

        a_out = self.conv_up4_alpha(y)

        alpha = torch.clamp(a_out, 0, 1)
 
        output = {"alpha": alpha}

        if is_training:
            fb_out = self.conv_up4_fb(y)
            F = torch.sigmoid(fb_out[:, 0:3])
            B = torch.sigmoid(fb_out[:, 3:6])
            output.update({"fg": F, "bg": B})

            attn_out = self.conv_up4_attn(x)
            attn = torch.sigmoid(attn_out)
            r1 = attn[:, 0:1]
            r2 = attn[:, 1:2]
            r3 = attn[:, 2:3]
            output.update({"r1": r1, "r2": r2, "r3": r3})

        return output



class SIMLoss(object):
    def __init__(self, args):
        self.args = args.loss

        self.use_comploss = args.loss.use_comploss
        self.use_laploss = args.loss.use_laploss
        self.use_fbloss = args.loss.use_fbloss
        self.use_fbcloss = args.loss.use_fbcloss
        self.use_fblaploss = args.loss.use_fblaploss

        self.kernel_diagonal = args.loss.kernel_diagonal
        self.kernel_laplacian = args.loss.kernel_laplacian
        self.kernel_second_order = args.loss.kernel_second_order

        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.lap_loss = LapLoss(5, device=torch.device('cuda'))

        self.gradient = SpatialGradient(diagonal=self.kernel_diagonal, 
                                        laplacian=self.kernel_laplacian, 
                                        second_order=self.kernel_second_order)

        self.loss_keys = ['loss_alpha']
        if self.use_comploss:
            self.loss_keys += ['loss_comp']
        if self.use_laploss:
            self.loss_keys += ['loss_lap']
        if self.use_fbloss:
            self.loss_keys += ['loss_fb', 'loss_comp_fb']
        if self.use_fbcloss:
            self.loss_keys += ['loss_fbc']
        if self.use_fblaploss:
            self.loss_keys += ['loss_fblap']

        if self.use_attention:
            self.loss_keys += ['loss_reg']
        if self.use_discriminator:
            self.loss_keys += ['loss_D']

    def gen_alpha_loss(self, pred, alpha, mask):
        diff = (pred - alpha) * mask
        loss = torch.sqrt(diff ** 2 + 1e-12)
        loss = loss.sum(dim=(1,2,3)) / (mask.sum(dim=(1,2,3)) + 1.)
        loss = loss.sum() / pred.shape[0]
        return loss

    def gen_fb_loss(self, pf, gf, pb, gb, fmask, bmask):
        df = (pf - gf) * fmask
        db = (pb - gb) * bmask
        loss = torch.sqrt(df**2 + 1e-12) + torch.sqrt(db**2 + 1e-12)
        loss = loss.sum(dim=(1,2,3)) / (fmask.sum(dim=(1,2,3)) + bmask.sum(dim=(1,2,3)) + 1.)
        loss = loss.sum() / pf.shape[0]
        return loss

    def gen_comp_loss(self, img, fg, bg, alpha, mask):
        comp = alpha * fg + (1. - alpha) * bg
        diff = (comp - img) * mask
        loss = torch.sqrt(diff ** 2 + 1e-12)
        loss = loss.sum(dim=(1,2,3)) / (mask.sum(dim=(1,2,3)) + 1.) / 3.
        loss = loss.sum() / alpha.shape[0]
        return loss

    def gen_attention_loss(self, grad_a, grad_f, grad_b, grad_i, attn_a, attn_f, attn_b, mask):
        grad_a_m = grad_a.abs().sum(dim=2)
        grad_f_m = grad_f.abs().sum(dim=2)
        grad_b_m = grad_b.abs().sum(dim=2)
        grad_i_m = grad_i.abs().sum(dim=2)
        grad_fba = grad_a_m*attn_a + grad_f_m*attn_f + grad_b_m*attn_b 
        diff = torch.sqrt((grad_fba - grad_i_m)**2 + 1e-12)
        loss_reg = (diff * mask).sum() / (mask.sum() + 1.) / 3.
        return loss_reg

    def gen_discriminator_loss(self, d_out):
        bce_loss = torch.nn.BCELoss()
        mse_loss = torch.nn.MSELoss()

        fake_ret = d_out['fake_ret']
        real_ret = d_out['real_ret']
        fake_feats = d_out['fake_feats']
        real_feats = d_out['real_feats']

        loss_D = bce_loss(torch.sigmoid(fake_ret), torch.sigmoid(real_ret))

        loss_perp = [] 
        for i in range(len(fake_feats)):
            loss_perp.append(mse_loss(fake_feats[i], real_feats[i]))
        loss_D += torch.tensor(loss_perp).mean()
        return loss_D

    def calc_loss(self, out, gt):
        trimap = gt['trimap']

        g_a = gt['alpha']
        g_i = gt['image']
        p_a = out['alpha']
        g_f = gt['fg']
        g_b = gt['bg']

        umask = (trimap == 128).float()
        fmask = (trimap >= 0).float()
        bmask = (trimap <= 128).float()

        loss_dict = {}
        loss_alpha = self.gen_alpha_loss(p_a, g_a, umask) 
        loss_dict['loss_alpha'] = loss_alpha

        if self.use_comploss:
            p_f = out['fg']
            p_b = out['bg']
            loss_comp = self.gen_comp_loss(g_i, g_f, g_b, p_a, umask)
            loss_dict['loss_comp'] = loss_comp * self.args.weight_comp

        if self.use_laploss:
            loss_lap = self.lap_loss(p_a, g_a)
            loss_dict['loss_lap'] = loss_lap * self.args.weight_lap

        if self.use_fbloss:
            p_f = out['fg']
            p_b = out['bg']
            loss_fb = self.gen_fb_loss(p_f, g_f, p_b, g_b, fmask, bmask)
            loss_comp_fb = self.gen_comp_loss(g_i, p_f, p_b, g_a, umask)
            loss_dict['loss_fb'] = loss_fb * self.args.weight_fb
            loss_dict['loss_comp_fb'] = loss_comp_fb * self.args.weight_fb

        if self.use_fbcloss:
            p_f = out['fg']
            p_b = out['bg']
            loss_fc = self.gen_comp_loss(g_i, p_f, g_b, g_a, umask)
            loss_bc = self.gen_comp_loss(g_i, g_f, p_b, g_a, umask)
            loss_fbc = self.gen_comp_loss(g_i, p_f, p_b, p_a, umask)
            loss_fbc = (loss_fc + loss_bc + loss_fbc) / 3.
            loss_dict['loss_fbc'] = loss_fbc * self.args.weight_fb

        if self.use_fblaploss:
            loss_flap = self.lap_loss(p_f, g_f)
            loss_blap = self.lap_loss(p_b, g_b)
            loss_fblap = (loss_flap + loss_blap) / 2.
            loss_dict['loss_fblap'] = loss_fblap * self.args.weight_fb

        if self.use_discriminator:
            loss_D = self.gen_discriminator_loss(out)
            loss_dict['loss_D'] = loss_D * self.args.weight_D

        if self.use_attention:
            grad_f = self.gradient(p_f)
            grad_b = self.gradient(p_b)
            grad_a = self.gradient(p_a)
            grad_i = self.gradient(g_i)
            r1, r2, r3 = out['r1'], out['r2'], out['r3']
            loss_reg = self.gen_attention_loss(grad_a, grad_f, grad_b, grad_i, r3, r1, r2, umask)
            loss_reg += (grad_f_l1 * grad_b_l1).mean() + (grad_a_l1 * grad_b_l1).mean() 
            loss_dict['loss_reg'] = loss_reg * self.args.weight_reg

        loss = 0.
        for key in self.loss_keys:
            loss += loss_dict[key] 
        loss_dict['loss'] = loss
        return loss_dict
