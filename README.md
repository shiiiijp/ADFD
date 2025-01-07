# Age-Dependent Face Diversification via Latent Space Analysis (CGI2023)

![teaser](docs/teaser.jpg)

This code is our implementation of the following paper:

Taishi Ito, Yuki Endo, Yoshihiro Kanamori: "Age-Dependent Face Diversification via Latent Space Analysis" The Visual Computer (Proc. of Computer Graphics International 2023), 2023. [[Project Page](http://cgg.cs.tsukuba.ac.jp/~itohlee/pub/ADFD)][[PDF (11MB)](http://www.cgg.cs.tsukuba.ac.jp/~itohlee/pub/ADFD/pdf/CGI_2023_paper_Ito_f.pdf)]

## Abstract
> Facial age transformation methods can change facial appearance according to the target age. However, most existing methods do not consider that people get older with different attribute changes (e.g., wrinkles, hair volume, and face shape) depending on their circumstances and environment. Diversifying such age-dependent attributes while preserving a person’s identity is crucial to broaden the applications of age transformation. In addition, the accuracy of age transformation to childhood is limited due to dataset bias. To solve these problems, we propose an age transformation method based on latent space analysis of StyleGAN. Our method obtains diverse age-transformed images by randomly manipulating age-dependent attributes in a latent space. To do so, we analyze the latent space and perturb channels affecting age-dependent attributes. We then optimize the perturbed latent code to refine the age and identity of the output image. We also present an unsupervised approach for improving age transformation to childhood. Our approach is based on the assumption that existing methods cannot sufficiently move a latent code toward a desired direction. We extrapolate an estimated latent path and iteratively update the latent code along the extrapolated path until the output image reaches the target age. Quantitative and qualitative comparisons with existing methods show that our method improves output diversity and preserves the target age and identity. We also show that our method can more accurately perform age transformation to childhood. 

## Prerequisites
Run the following code to install all conda packages.
```
conda env create -f environment/ADFD_env.yml
```

## Testing
1. Download following pretrained models from each repository and save to the directory `pretrained_models`.

      Repository | Model(s) to download
      ---------- | -----------------
      [SAM](https://github.com/yuval-alaluf/SAM#pretrained-models)  | **SAM**: Pretrained SAM for age transformation. <br> **VGG Age Classifier**: VGG age classifier from DEX for use in our aging loss. Fine-tuned by [yuval-alaluf](https://github.com/yuval-alaluf/SAM) on the FFHQ-Aging dataset.
      [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)  | **IR-SE50**: Pretrained IR-SE50 model for use in our ID loss.

3. Run `scripts/inference_ADFD.py` for Age-Dependent Face Diversification, or `scripts/inference_guided_optimization.py` for Guided Optimization.

      Age-Dependent Face Diversification:
      ```
      python scripts/inference_ADFD.py \
      --exp_dir=/path/to/experiment \
      --checkpoint_path=/path/to/pretrained_model \
      --data_path=/path/to/test_data \
      --test_batch_size=4 \
      --test_workers=4 \
      --aging_lambda=5 \
      --id_lambda=0.1 \
      --l2_lambda=0.01 \
      --lpips_lambda=0.01 \
      --use_weighted_id_loss \
      --couple_outputs \
      --target_age=80
      ```
      Guided Optimization:
      ```
      python scripts/inference_guided_optimization.py \
      --exp_dir=/path/to/experiment \
      --checkpoint_path=/path/to/pretrained_model \
      --data_path=/path/to/test_data \
      --test_batch_size=4 \
      --test_workers=4 \
      --target_age=5
      ```

## Citation
Please cite our paper if you find the code useful:
```
@article{ItoCGI23,
      author    = {Taishi Ito and Yuki Endo and Yoshihiro Kanamori},
      title     = {Age-Dependent Face Diversification via Latent Space Analysis},
      journal   = {The Visual Computer (Proc. of Computer Graphics Internatinal 2023)},
      volume    = {39},
      number    = {8},
      pages     = {3221--3233},
      year      = {2023},
      publisher = {Springer}
}
```


## Acknowledgements
This code heavily borrows from the [SAM](https://github.com/yuval-alaluf/SAM) repository.
