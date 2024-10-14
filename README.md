# Dynamic-IP-Adapter
---
## Introduction
This is an extension of the [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter).
It applies dynamic weights during inference based on time steps.
It allows independent control of each condition's influence on image generation.

## Installation
```
pip install diffusers==0.22.1
pip install ip-adapter @ git+https://github.com/tencent-ailab/IP-Adapter.git@62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2
