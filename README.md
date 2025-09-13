# UniTok Transformers

**Unofficial** implementation of UniTok with Hugging Face Transformers support.

> ⚠️ **Disclaimer**: This is an unofficial modification of the original [UniTok](https://github.com/FoundationVision/UniTok) repository. This version has been adapted to support Hugging Face transformers-style model loading and saving functionality.

## 🤗 Pretrained Model

The pretrained model is available on Hugging Face Hub:

**Model Hub**: [XuejiFang/UniTok_transformers](https://huggingface.co/XuejiFang/UniTok_transformers)

You can directly load the model from Hugging Face Hub without downloading files manually:

```python
from UniTok import UniTok

# Load directly from Hugging Face Hub
model = UniTok.from_pretrained("XuejiFang/UniTok_transformers")
```

## Overview

UniTok is a unified tokenizer for images with vector quantization, enabling both image reconstruction and generation tasks. This repository modifies the original UniTok implementation to integrate seamlessly with the Hugging Face transformers ecosystem.

**Key Features:**
- 🤗 **Hugging Face Compatibility**: Support for `save_pretrained()` and `from_pretrained()` methods
- 🎯 **Simplified API**: Easy-to-use interface following transformers conventions  
- 🚀 **Ready-to-use**: Pre-configured inference pipeline with minimal setup
- 📦 **Standardized Format**: Uses standard `config.json` and `model.safetensors` files

## Installation

```bash
git clone https://github.com/XuejiFang/UniTok_transformers.git
cd UniTok_transformers
pip install torch torchvision transformers timm pillow open_clip_torch
```

**Note**: No manual model download required! The model will be automatically downloaded from Hugging Face Hub on first use.

## Quick Start

### Basic Usage

```python
from UniTok import UniTok, UniTokConfig

# Load pretrained model from Hugging Face Hub
model = UniTok.from_pretrained("XuejiFang/UniTok_transformers")

# Or load from local directory
model = UniTok.from_pretrained('./ckpt/unitok')

# Create custom configuration
config = UniTokConfig(
    img_size=224,
    vocab_size=8192,
    embed_dim=768
)
custom_model = UniTok(config)

# Save model
model.save_pretrained('./my_unitok_model')
```

### Image Reconstruction

```python
import torch
from PIL import Image
from UniTok import UniTok

# Load model from Hugging Face Hub
model = UniTok.from_pretrained("XuejiFang/UniTok_transformers")
model.eval()

# Process image
img = Image.open('your_image.jpg')
# ... (preprocessing code)

# Encode and reconstruct
with torch.no_grad():
    indices = model.img_to_idx(img_tensor)
    reconstructed = model.idx_to_img(indices)
```

### Command Line Inference

```bash
# Quick inference (automatically downloads model from Hugging Face Hub)
./launch.sh

# Use local model
python inference.py --model_path ./ckpt/unitok --src_img path/to/image.jpg --rec_img output.png

# Use Hugging Face Hub model
python inference.py --model_path XuejiFang/UniTok_transformers --src_img path/to/image.jpg --rec_img output.png
```

## Model Configuration

The `UniTokConfig` class provides comprehensive configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | 'vitamin_base' | Encoder model name |
| `img_size` | 256 | Input image size |
| `vocab_size` | 16384 | Vocabulary size for VQ |
| `vocab_width` | 768 | Quantization vocabulary width |
| `num_codebooks` | 1 | Number of codebooks |
| `embed_dim` | 512 | CLIP embedding dimension |
| `quant_proj` | 'linear' | Quantization projection type |

## Project Structure

```
UniTok_transformers/
├── UniTok/                    # Core module
│   ├── unitok.py             # Main UniTok model (PreTrainedModel)
│   ├── configuration_unitok.py # Configuration class
│   ├── vitamin.py            # ViTamin architecture
│   ├── quant.py              # Vector quantization
│   └── vqvae.py              # VQVAE components
├── inference.py              # Inference script
├── launch.sh                 # Quick start script
├── ckpt/unitok/              # Local pretrained model (optional)
│   ├── config.json           # Model configuration
│   └── model.safetensors     # Model weights
└── assets/                   # Sample images
```

**Note**: The `ckpt/unitok/` directory is optional. Models can be loaded directly from Hugging Face Hub.

## Key Differences from Original

This repository differs from the [original UniTok](https://github.com/FoundationVision/UniTok) in several ways:

### ✅ **Added Features:**
- Hugging Face transformers integration
- Standardized configuration system (`UniTokConfig`)
- Simplified model loading/saving
- English documentation and comments
- Streamlined codebase (removed training code)

### 🔄 **Modified Components:**
- `UniTok` class now inherits from `PreTrainedModel`
- Configuration handled via `PretrainedConfig`
- Inference pipeline simplified and optimized
- Removed dependency on custom `utils` modules

### ❌ **Removed Features:**
- Training scripts and related utilities
- Evaluation pipelines
- Large-scale experiment configurations
- Legacy checkpoint loading methods

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- transformers >= 4.20.0
- timm
- Pillow
- open_clip_torch

## Citation

If you use this code, please cite both this repository and the original work:

```bibtex
# Original UniTok paper
@article{unitok,
  title={UniTok: A Unified Tokenizer for Visual Generation and Understanding},
  author={Ma, Chuofan and Jiang, Yi and Wu, Junfeng and Yang, Jihan and Yu, Xin and Yuan, Zehuan and Peng, Bingyue and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2502.20321},
  year={2025}
}

# This repository
@misc{unitok_transformers2025,
  title={UniTok Transformers: Unofficial Hugging Face Integration},
  author={Xueji Fang},
  url={https://github.com/XuejiFang/UniTok_transformers},
  year={2025}
}
```

## License

This project follows the same license as the original UniTok repository. Please check the [original repository](https://github.com/FoundationVision/UniTok) for license details.

## Acknowledgments

- Original UniTok implementation: [FoundationVision/UniTok](https://github.com/FoundationVision/UniTok)
- Hugging Face transformers library for the excellent framework
- Hugging Face Hub for hosting the pretrained model: [XuejiFang/UniTok_transformers](https://huggingface.co/XuejiFang/UniTok_transformers)
- Claude Code for assistance in the adaptation process

## Contributing

This is an unofficial adaptation. For issues related to the core UniTok functionality, please refer to the [original repository](https://github.com/FoundationVision/UniTok). For issues specific to the transformers integration, feel free to open issues in this repository.

---

**Note**: This repository is for research and educational purposes. Please refer to the original UniTok paper and repository for the authoritative implementation and latest developments.