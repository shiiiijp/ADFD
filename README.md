# Age-Dependent Face Diversification via Latent Space Analysis (CGI2023)

[Project Page](http://cgg.cs.tsukuba.ac.jp/~itohlee/pub/ADFD)

> Facial age transformation methods can change facial appearance according to the target age. However, most existing methods do not consider that people get older with different attribute changes (e.g., wrinkles, hair volume, and face shape) depending on their circumstances and environment. Diversifying such age-dependent attributes while preserving a person’s identity is crucial to broaden the applications of age transformation. In addition, the accuracy of age transformation to childhood is limited due to dataset bias. To solve these problems, we propose an age transformation method based on latent space analysis of StyleGAN. Our method obtains diverse age-transformed images by randomly manipulating age-dependent attributes in a latent space. To do so, we analyze the latent space and perturb channels affecting age-dependent attributes. We then optimize the perturbed latent code to refine the age and identity of the output image. We also present an unsupervised approach for improving age transformation to childhood. Our approach is based on the assumption that existing methods cannot sufficiently move a latent code toward a desired direction. We extrapolate an estimated latent path and iteratively update the latent code along the extrapolated path until the output image reaches the target age. Quantitative and qualitative comparisons with existing methods show that our method improves output diversity and preserves the target age and identity. We also show that our method can more accurately perform age transformation to childhood. 
