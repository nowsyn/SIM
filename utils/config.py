from easydict import EasyDict

CONFIG = EasyDict({})
CONFIG.is_default = True
CONFIG.task = "SIM"
CONFIG.version = "SIM"
CONFIG.debug = False
CONFIG.phase = "train"
CONFIG.dataset = "SIMD"
# distributed training
CONFIG.dist = False
# global variables which will be assigned in the runtime
CONFIG.local_rank = 0
CONFIG.gpu = 0
CONFIG.world_size = 1
CONFIG.devices = (0,)


# ===============================================================================
# Model config
# ===============================================================================
CONFIG.classifier = EasyDict({})
CONFIG.classifier.arch = "resnet50"
CONFIG.classifier.n_channel = 4
CONFIG.classifier.num_classes = 20
CONFIG.classifier.resume_checkpoint = None
CONFIG.classifier.load_size = 320

CONFIG.discriminator = EasyDict({})
CONFIG.discriminator.arch = "resnet50"
CONFIG.discriminator.n_channel = 4
CONFIG.discriminator.num_classes = 20
CONFIG.discriminator.resume_checkpoint = None
CONFIG.discriminator.load_size = 320

CONFIG.model = EasyDict({})
CONFIG.model.num_classes = 20
CONFIG.model.pretrain_checkpoint = None
CONFIG.model.resume_checkpoint = None
CONFIG.model.trimap_channel = 3

CONFIG.model.arch = EasyDict({})
CONFIG.model.arch.n_channel = 11
CONFIG.model.arch.encoder = "resnet_BN"
# parameters for ppm
CONFIG.model.arch.pool_scales = (1,2,3,6)
CONFIG.model.arch.ppm_channel = 256
# parameters for aspp
CONFIG.model.arch.atrous_rates = (12, 24, 36)
CONFIG.model.arch.aspp_channel = 256


# ===============================================================================
# Loss config
# ===============================================================================
CONFIG.loss = EasyDict({})
CONFIG.loss.use_laploss = False
CONFIG.loss.use_comploss = False
CONFIG.loss.use_fbloss = False
CONFIG.loss.use_fbcloss = False
CONFIG.loss.use_fblaploss = False
CONFIG.loss.use_attention = False
CONFIG.loss.use_discriminator = False
CONFIG.loss.kernel_diagonal = False
CONFIG.loss.kernel_laplacian = False
CONFIG.loss.kernel_second_order = False
CONFIG.loss.weight_comp = 1.0
CONFIG.loss.weight_fb = 1.0
CONFIG.loss.weight_reg = 1.0
CONFIG.loss.weight_D = 1.0


# ===============================================================================
# Dataloader config
# ===============================================================================
CONFIG.data = EasyDict({})
CONFIG.data.workers = 0
CONFIG.data.online = True
CONFIG.data.num_classes = 20
CONFIG.data.load_size = 320
CONFIG.data.max_size = 1920
CONFIG.data.min_size = 800
CONFIG.data.augmentation = True

CONFIG.data.train_alpha_dir = None
CONFIG.data.train_fg_dir = None
CONFIG.data.train_bg_dir = None
CONFIG.data.test_img_dir = None
CONFIG.data.test_alpha_dir = None
CONFIG.data.test_trimap_dir = None
CONFIG.data.test_dir = None

CONFIG.data.aug = EasyDict({})
CONFIG.data.aug.crop_sizes = (320, 480, 640)
CONFIG.data.aug.ksize_range = (3, 5)
CONFIG.data.aug.iteration_range = (5, 15)
CONFIG.data.aug.flip = True
CONFIG.data.aug.adjust_gamma = False
CONFIG.data.aug.gamma_range = (0.2, 2)
CONFIG.data.aug.adjust_color = False
CONFIG.data.aug.color_delta = 0.2
CONFIG.data.aug.rescale = False
CONFIG.data.aug.rescale_min = 0.25
CONFIG.data.aug.rescale_max = 0.5
CONFIG.data.aug.rotate = False
CONFIG.data.aug.rotate_degree = 60
CONFIG.data.aug.rejpeg = False
CONFIG.data.aug.composite_fg = False
CONFIG.data.aug.gaussian_noise = False


# ===============================================================================
# Training config
# ===============================================================================
CONFIG.train = EasyDict({})
CONFIG.train.batch_size = 8
CONFIG.train.epochs = 30
CONFIG.train.start_epoch = 0
CONFIG.train.decay_step = 10
CONFIG.train.warmup_step = 0
CONFIG.train.lr = 1e-5
CONFIG.train.min_lr = 1e-8
CONFIG.train.reset_lr = False
CONFIG.train.adaptive_lr = False
CONFIG.train.optim = "Adam"
CONFIG.train.eps = 1e-5
CONFIG.train.beta1 = 0.9
CONFIG.train.beta2 = 0.999
CONFIG.train.momentum = 0.9
CONFIG.train.weight_decay = 1e-4
CONFIG.train.clip_grad = True
CONFIG.train.print_freq = 10
CONFIG.train.save_freq = 10
CONFIG.train.test_freq = 10


# ===============================================================================
# Testing config
# ===============================================================================
CONFIG.test = EasyDict({})
CONFIG.test.max_size = 1920
CONFIG.test.min_size = 800
CONFIG.test.batch_size = 1
CONFIG.test.checkpoint = "best_model"
CONFIG.test.fast_eval = True


# ===============================================================================
# Logging config
# ===============================================================================
CONFIG.log = EasyDict({})
CONFIG.log.tensorboard_path = "./logs/tensorboard"
CONFIG.log.tensorboard_step = 10
CONFIG.log.tensorboard_image_step = 10
CONFIG.log.logging_path = "./logs/stdout"
CONFIG.log.logging_step = 10
CONFIG.log.logging_level = "DEBUG"
CONFIG.log.checkpoint_path = "./checkpoints"
CONFIG.log.checkpoint_step = 10
CONFIG.log.visualize_path = "./logs/visualizations"


# ===============================================================================
# util functions
# ===============================================================================
def parse_config(custom_config, default_config=CONFIG, prefix="CONFIG"):
    """
    This function will recursively overwrite the default config by a custom config
    :param default_config:
    :param custom_config: parsed from config/config.toml
    :param prefix: prefix for config key
    :return: None
    """
    if "is_default" in default_config:
        default_config.is_default = False

    for key in custom_config.keys():
        full_key = ".".join([prefix, key])
        if key not in default_config:
            raise NotImplementedError("Unknown config key: {}".format(full_key))
        elif isinstance(custom_config[key], dict):
            if isinstance(default_config[key], dict):
                parse_config(default_config=default_config[key],
                            custom_config=custom_config[key],
                            prefix=full_key)
            else:
                raise ValueError("{}: Expected {}, got dict instead.".format(full_key, type(custom_config[key])))
        else:
            if isinstance(default_config[key], dict):
                raise ValueError("{}: Expected dict, got {} instead.".format(full_key, type(custom_config[key])))
            else:
                default_config[key] = custom_config[key]


def load_config(config_path):
    import toml
    with open(config_path) as fp:
        custom_config = EasyDict(toml.load(fp))
    parse_config(custom_config=custom_config)
    return CONFIG
   

if __name__ == "__main__":
    from pprint import pprint

    pprint(CONFIG)
    load_config("../config/example.toml")
    pprint(CONFIG)
