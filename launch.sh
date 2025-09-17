#!/bin/bash

# UniTok Inference Launch Script
# Modified from https://github.com/FoundationVision/UniTok to support 
# transformers-style model loading with assistance from Claude Code.

python inference.py \
    --model_path ckpt/384-4 \
    --src_img assets/vis_imgs/v7.jpg \
    --rec_img ./assets/rec_imgs/ori/rec_v7.png \
    --max_resolution 1024