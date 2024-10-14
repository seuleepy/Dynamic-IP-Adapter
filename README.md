# Dynamic-IP-Adapter
---
## Introduction
This is an extension of the [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter).
It applies dynamic weights during inference based on time steps.
It allows independent control of each condition's influence on image generation.

## Installation
```
# install diffusers
pip install diffusers==0.22.1

# install ip-adapter
pip install ip-adapter @ git+https://github.com/tencent-ailab/IP-Adapter.git@62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2

# install clip (to evaluate)
pip install git+https://github.com/openai/CLIP.git

# download the models
cd IP-Adapter
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
```
After installation, add that files(diffusers, ip_adapter).
## Generation
```
bash ./gen-and-eval/generation/generation_script
```
## Evaluation
```
bash ./gen-and-eval/evlauation/clip_text_score_script
bash ./gen-and-eval/evaluation/clip_image_score_script
```
To evaluate, dataset can be downloaded from [dreambooth](https://github.com/google/dreambooth).

## Result
In this image, Dynamic model is a linear model.

<img src="https://github.com/user-attachments/assets/e7375bab-d5a9-4659-934d-da95d899b165" width="350" height="566"/>
