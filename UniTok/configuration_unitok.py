"""
UniTok Configuration

Modified from https://github.com/FoundationVision/UniTok to support 
Hugging Face transformers-style configuration with assistance from Claude Code.
"""

import json
from transformers import PretrainedConfig


class UniTokConfig(PretrainedConfig):
    """
    Configuration class for UniTok model
    
    Inherits from transformers.PretrainedConfig to support from_pretrained and save_pretrained functionality
    """
    model_type = "unitok"
    
    def __init__(
        self,
        model='vitamin_base',
        num_query=0,
        img_size=256,
        drop_path=0.1,
        quant_proj='linear',
        vocab_width=768,
        vocab_size=16384,
        vq_beta=0.25,
        le=0.1,
        e_temp=1.0,
        num_codebooks=1,
        grad_ckpt=False,
        embed_dim=512,
        text_width=512,
        text_heads=8,
        text_layers=12,
        text_vocab_size=49408,
        text_context_length=77,
        resize_ratio=1.125,
        **kwargs
    ):
        """
        Args:
            model: Encoder model name (default: 'vitamin_base')
            num_query: Number of query tokens (default: 0)
            img_size: Input image size (default: 256)
            drop_path: Drop path rate (default: 0.1)
            quant_proj: Quantization projection type, 'linear' or 'attn' (default: 'linear')
            vocab_width: Quantization vocabulary width (default: 768)
            vocab_size: Vocabulary size for VQ (default: 16384)
            vq_beta: VQ loss weight (default: 0.25)
            le: Entropy loss weight (default: 0.1)
            e_temp: Entropy temperature (default: 1.0)
            num_codebooks: Number of codebooks (default: 1)
            grad_ckpt: Use gradient checkpointing (default: False)
            embed_dim: CLIP embedding dimension (default: 512)
            text_width: Text encoder width (default: 512)
            text_heads: Text encoder heads (default: 8)
            text_layers: Text encoder layers (default: 12)
            text_vocab_size: Text vocabulary size (default: 49408)
            text_context_length: Text context length (default: 77)
            resize_ratio: Image resize ratio for preprocessing (default: 1.125)
        """
        super().__init__(**kwargs)
        
        self.model = model
        self.num_query = num_query
        self.img_size = img_size
        self.drop_path = drop_path
        self.quant_proj = quant_proj
        self.vocab_width = vocab_width
        self.vocab_size = vocab_size
        self.vq_beta = vq_beta
        self.le = le
        self.e_temp = e_temp
        self.num_codebooks = num_codebooks
        self.grad_ckpt = grad_ckpt
        self.embed_dim = embed_dim
        self.text_width = text_width
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_vocab_size = text_vocab_size
        self.text_context_length = text_context_length
        self.resize_ratio = resize_ratio