# Semantic Image Matting
---
This is the official repository of Semantic Image Matting (CVPR2021). 


### Overview
<img src="./figures/framework.jpg" width="800" alt="framework" align=center/>

Natural image matting separates the foreground from background in fractional occupancy which can be caused by highly transparent objects, complex foreground (e.g., net or tree), and/or objects containing very fine details (e.g., hairs). Although conventional matting formulation can be applied to all of the above cases, no previous work has attempted to reason the underlying causes of matting due to various foreground semantics.

We show how to obtain better alpha mattes by incorporating into our framework semantic classification of matting regions. Specifically, we consider and learn 20 classes of matting patterns, and propose to extend the conventional trimap to semantic trimap. The proposed semantic trimap can be obtained automatically through patch structure analysis within trimap regions. Meanwhile, we learn a multi-class discriminator to regularize the alpha prediction at semantic level, and content-sensitive weights to balance different regularization losses. 

### Dataset
Download our semantic image matting dataset (SIMD) [here](https://drive.google.com/file/d/1Cl_Nacgid9ZLVZ7j-cMHnim4SocTMY92/view?usp=sharing). SIMD is composed self-collected images and a subset of adobe images. To obtain the complete dataset, please contact Brian Price (bprice@adobe.com) for the Adobe Image Matting dataset first and follow the instructions within SIMD.zip. 

### Requirements
The codes are tested in the following environment:

* Python 3.7
* Pytorch 1.9.0
* CUDA 10.2 & CuDNN 7.6.5

### Performance
Some pretrained models are listed below with their performance.
<table>
    <thead>
        <tr>
            <th>Methods</th>
            <th>SAD</th>
            <th>MSE</th>
            <th>Grad</th>
            <th>Conn</th>
            <th>Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>SIMD</td>
            <td>27.9</td>
            <td>4.7</td>
            <td>11.6</td>
            <td>20.8</td>
            <th><a href="https://drive.google.com/file/d/1kmhQtO-6wXTxgHtQCLRj3xPOEtXyzTXC/view?usp=sharing"> model </a></th>
        </tr>
        <tr>
            <td>Composition-1K (paper)</td>
            <td>28.0</td>
            <td>5.8</td>
            <td>10.8</td>
            <td>24.8</td>
            <td></td>
        </tr>
        <tr>
            <td>Composition-1K (repo)</td>
            <td>27.7</td>
            <td>5.6</td>
            <td>10.7</td>
            <td>24.4</td>
            <td><a href=""https://drive.google.com/file/d/1uUIhzBKyTRfu0Ylhm0O9pyV9TayPaODt/view?usp=sharing> model </a></td>
        </tr>
    </tbody>
</table>

### Run

Download the model and put it under `checkpoints/DIM` or `checkpoints/Adobe` in the root directory. Download the classifier [here](https://drive.google.com/file/d/12JCGqDylBXJpgDhj4hg_JZYdbHlX8TKe/view?usp=sharing) and put it under `checkpoints`. Run the inference and evaluation by
```
python scripts/main.py -c config/CONFIG.yaml 
``` 

### Results
<img src="./figures/example1.png" width="800" alt="example1" align=center/>
<img src="./figures/example2.png" width="800" alt="example2" align=center/>


### Reference
If you find our work useful in your research, please consider citing:

```
@inproceedings{sun2021sim,
  author    = {Yanan Sun and Chi-Keung Tang and Yu-Wing Tai}
  title     = {Semantic Image Matting},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2021},
}
```

### Acknowledgment
This repo borrows code from several repos, like [GCA](https://github.com/Yaoyi-Li/GCA-Matting) and [FBA](https://github.com/MarcoForte/FBA_Matting).
