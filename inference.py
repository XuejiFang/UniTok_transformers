"""
UniTok Inference Script

Modified from https://github.com/FoundationVision/UniTok to support 
transformers-style model loading with assistance from Claude Code.
"""

import os
import torch
import argparse
from PIL import Image
from UniTok import UniTok
from torchvision.transforms import transforms


def normalize_01_into_pm1(tensor):
    """Normalize tensor from [0,1] range to [-1,1] range"""
    return tensor * 2.0 - 1.0


def save_img(img: torch.Tensor, path):
    img = img.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
    img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    img = Image.fromarray(img[0])
    img.save(path)


def get_transform(img_size=256, resize_ratio=1.125):
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize(int(img_size * resize_ratio)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), 
        normalize_01_into_pm1,
    ])


def main(args):
    # Load model using transformers-style loading
    model_path = args.model_path
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading model from: {model_path}")
    unitok = UniTok.from_pretrained(model_path)
    img_size = unitok.config.img_size
    resize_ratio = unitok.config.resize_ratio
    
    unitok.to(device)
    unitok.eval()

    # Prepare image preprocessing
    preprocess = get_transform(img_size, resize_ratio)
    img = Image.open(args.src_img).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        code_idx = unitok.img_to_idx(img)
        rec_img = unitok.idx_to_img(code_idx)

    final_img = torch.cat((img, rec_img), dim=3)
    save_img(final_img, args.rec_img)

    print('The image is saved to {}. The left one is the original image after resizing and cropping. The right one is the reconstructed image.'.format(args.rec_img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--src_img', type=str, default='')
    parser.add_argument('--rec_img', type=str, default='')
    args = parser.parse_args()
    main(args)

