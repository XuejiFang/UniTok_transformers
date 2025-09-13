#!/bin/bash

# UniTok Inference Launch Script
# Modified from https://github.com/FoundationVision/UniTok to support 
# transformers-style model loading with assistance from Claude Code.

python inference.py \
    --model_path ./ckpt/unitok \
    --src_img assets/vis_imgs/v0.jpg \
    --rec_img ./assets/rec_imgs/v0.png

echo "Inference completed! View result image: ./rec_img.png"