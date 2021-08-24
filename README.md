# Semantic Image Matting [Under construction]
---
This is the official repository of Semantic Image Matting (CVPR2021). 


### Overview
<img src="./figures/framework.jpg" width="800" alt="framework" align=center/>

Natural image matting separates the foreground from background in fractional occupancy which can be caused by highly transparent objects, complex foreground (e.g., net or tree), and/or objects containing very fine details (e.g., hairs). Although conventional matting formulation can be applied to all of the above cases, no previous work has attempted to reason the underlying causes of matting due to various foreground semantics.

We show how to obtain better alpha mattes by incorporating into our framework semantic classification of matting regions. Specifically, we consider and learn 20 classes of matting patterns, and propose to extend the conventional trimap to semantic trimap. The proposed semantic trimap can be obtained automatically through patch structure analysis within trimap regions. Meanwhile, we learn a multi-class discriminator to regularize the alpha prediction at semantic level, and content-sensitive weights to balance different regularization losses. 

### Dataset
Download our semantic image matting dataset (SIMD) [here](https://drive.google.com/file/d/1Cl_Nacgid9ZLVZ7j-cMHnim4SocTMY92/view?usp=sharing). SIMD is composed self-collected images and a subset of adobe images. To obtain the complete dataset, please contact Brian Price (bprice@adobe.com) for the Adobe Image Matting dataset first and follow the instructions within SIMD.zip. 

### Results
<img src="./figures/example1.png" width="800" alt="example1" align=center/>
<img src="./figures/example2.png" width="800" alt="example2" align=center/>
