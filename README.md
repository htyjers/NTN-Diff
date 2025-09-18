# NTN-Diff

This repository is the official code for the paper "One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting" by Haipeng Liu (hpliu_hfut@hotmail.com), Yang Wang (corresponding author), Meng Wang. NeurIPS 2025, San Diego, USA

#
## Introduction
In this paper, we propose a null-text-null frequency-aware diffusion models, dubbed **NTN-Diff**, for text-guided image inpainting, by decomposing the semantics consistency across masked and unmasked regions into the consistencies as per each frequency band, while preserving the unmasked regions, to circumvent two challenges in a row. Based on the diffusion process, we further divide the denoising process into early (high-level noise) and late (low-level noise) stages, where the mid-and-low frequency bands are disentangled during the denoising process. As observed that, the stable mid-frequency band is progressively denoised to be semantically aligned during text-guided denoising process, which, meanwhile, serves as the guidance to the null-text denoising process to denoise low-frequency band for the masked regions, followed by a subsequent text-guided denoising process at late stage, to achieve the semantics consistency for mid-and-low frequency bands across masked and unmasked regions, while preserve the unmasked regions. Extensive experiments validate the superiority of NTN-Diff over the state-of-the-art diffusion models to text-guided diffusion models. 


![](image/image1.png)
<p align="center">Figure 1. Illustration of the proposed NTN-Diff pipeline.</p>

#
## Inference
1. Dataset Preparation:
[BrushBench](https://github.com/TencentARC/BrushNet)
   
2. Pre-trained models:
[Realistic Vision V6.0 B1](https://civitai.com/models/4201?modelVersionId=501240)
  
3. Run the following command:
```
Python3 test.py
```

4. Inpainted Image: 
[Baidu](https://pan.baidu.com/s/1VG7yR5JD_gbA29rWHzxh9A?pwd=yyve)


#
## Example Results

- Visual comparison between our method and the competitors.
![](image/image2.png)

#
## Citation

If any part of our paper and repository is helpful to your work, please generously cite with:

```

```
