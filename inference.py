"""
UniTok Inference Script

Modified from https://github.com/FoundationVision/UniTok to support 
transformers-style model loading with assistance from Claude Code.
"""

import os
import torch
import argparse
from PIL import Image
from UniTok import UniTok, MultiResolutionProcessor, normalize_01_into_pm1, save_img
from torchvision import transforms

def main(args):
    # Load model using transformers-style loading
    model_path = args.model_path
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading model from: {model_path}")
    unitok = UniTok.from_pretrained(model_path)
    
    unitok.to(device)
    unitok.eval()

    img = Image.open(args.src_img).convert("RGB")
    original_size = img.size
    processor = MultiResolutionProcessor(max_size=getattr(args, 'max_resolution', 256), patch_size=16)
    result = processor.process_image(img, training=False)
    img_tensor = transforms.ToTensor()(result['image'])
    img_tensor = normalize_01_into_pm1(img_tensor).unsqueeze(0).to('cuda')

    with torch.no_grad():
        rec_img = unitok.img_to_reconstructed_img(
            img_tensor,
            shape_info=result['shape_info'],
            attention_mask=None  # set none for inference
        )
    
    print(f'Original Image Size: {original_size} -> Resized To: {result["shape_info"]["processed_size"]}')
    
    if args.rec_img:
        final_path = args.rec_img
        save_img(rec_img, final_path)
        print(f'Saved Reconstructed Image To: {final_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='XuejiFang/UniTok_transformers', 
                       help='Path to model directory or Hugging Face model ID')
    parser.add_argument('--src_img', type=str, default='assets/vis_imgs/v0.jpg',
                       help='Path to source image')
    parser.add_argument('--rec_img', type=str, default='./rec_img.png',
                       help='Path to save reconstructed image')
    parser.add_argument('--max_resolution', type=int, default=256,
                       help='Maximum resolution for multi-resolution processing')
    args = parser.parse_args()
    main(args)

