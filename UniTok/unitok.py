"""
UniTok: A Unified Tokenizer for Visual Generation and Understanding

Modified from https://github.com/FoundationVision/UniTok to support 
save_pretrained and from_pretrained functionality with assistance from Claude Code.
"""

import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import PreTrainedModel

from .vitamin import GeGluMlp, ViTaminDecoder
from .quant import VectorQuantizerM
from .vqvae import AttnProjection
from .configuration_unitok import UniTokConfig


class UniTok(PreTrainedModel):
    config_class = UniTokConfig
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize UniTok model
        
        Args:
            config: UniTokConfig object or None for default configuration
            **kwargs: Additional parameters that override config values
        """
        # Handle configuration parameters
        if config is None:
            config = UniTokConfig(**kwargs)
        elif isinstance(config, dict):
            config = UniTokConfig(**config)
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        super().__init__(config)
        
        self.num_query = config.num_query
        self.img_size = config.img_size
        self.resize_ratio = config.resize_ratio

        self.encoder = timm.create_model(
            config.model,
            patch_size=1,
            fc_norm=False,
            drop_rate=0.0,
            num_classes=0,
            global_pool='',
            pos_embed='none',
            class_token=False,
            mlp_layer=GeGluMlp,
            reg_tokens=config.num_query,
            img_size=config.img_size,
            drop_path_rate=config.drop_path,
        )
        self.encoder.pos_embed = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim), requires_grad=False)

        if config.quant_proj == 'linear':
            self.quant_proj = nn.Linear(self.encoder.embed_dim, config.vocab_width)
        elif config.quant_proj == 'attn':
            self.quant_proj = AttnProjection(self.encoder.embed_dim, config.vocab_width, self.encoder.embed_dim // config.vocab_width)
        else:
            raise NotImplementedError(f"Unsupported quant_proj type: {config.quant_proj}")

        self.quantizer = VectorQuantizerM(
            vocab_size=config.vocab_size,
            vocab_width=config.vocab_width,
            beta=config.vq_beta,
            use_entropy_loss=config.le > 0,
            entropy_temp=config.e_temp,
            num_codebooks=config.num_codebooks,
        )

        if config.quant_proj == 'linear':
            self.post_quant_proj = nn.Linear(config.vocab_width, self.encoder.embed_dim)
        elif config.quant_proj == 'attn':
            self.post_quant_proj = AttnProjection(config.vocab_width, self.encoder.embed_dim, self.encoder.embed_dim // config.vocab_width)
        else:
            raise NotImplementedError(f"Unsupported quant_proj type: {config.quant_proj}")

        self.decoder = ViTaminDecoder(
            config.model,
            num_query=config.num_query,
            img_size=config.img_size,
            drop_path=config.drop_path,
            grad_ckpt=config.grad_ckpt,
        )

        text_cfg = {
            "width": config.text_width,
            "heads": config.text_heads,
            "layers": config.text_layers,
            "vocab_size": config.text_vocab_size,
            "context_length": config.text_context_length,
        }
        from open_clip.model import _build_text_tower
        self.text_encoder = _build_text_tower(config.embed_dim, text_cfg)

        self.fc_norm = nn.LayerNorm(self.encoder.embed_dim, eps=1e-6)
        self.projection = nn.Linear(self.encoder.embed_dim, config.embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.context_length = self.text_encoder.context_length
        self.vocab_size = self.text_encoder.vocab_size
        self.maybe_record_function = nullcontext

        self.text_no_grad = False
        self.encoder.set_grad_checkpointing(config.grad_ckpt)
        self.text_encoder.set_grad_checkpointing(config.grad_ckpt)

    def forward(self, img, vae_bs, text=None, ret_usages=False):
        img_tokens = self.encoder(img).float()
        with torch.cuda.amp.autocast(enabled=False):
            img_tokens = torch.utils.checkpoint.checkpoint(self.quant_proj, img_tokens, use_reentrant=False)
            img_tokens, vq_loss, entropy_loss, usages = self.quantizer(img_tokens)
            img_tokens = torch.utils.checkpoint.checkpoint(self.post_quant_proj, img_tokens, use_reentrant=False)
        img_rec = self.decoder(img_tokens[:vae_bs]).float()

        clip_visual = img_tokens.mean(dim=1)
        clip_visual = self.projection(self.fc_norm(clip_visual))
        clip_visual = F.normalize(clip_visual, dim=-1)
        if text is not None:
            clip_text = self.text_encoder(text)
            clip_text = F.normalize(clip_text, dim=-1)
        else:
            clip_text = None

        output_dict = {
            "img_rec": img_rec,
            "vq_loss": vq_loss,
            "entropy_loss": entropy_loss,
            "codebook_usages": usages,
            "clip_image_features": clip_visual,
            "clip_text_features": clip_text,
            "logit_scale": self.logit_scale.exp()
        }
        return output_dict

    def encode_image(self, image, normalize: bool = False):
        img_tokens = self.encoder(image)
        img_tokens = self.quant_proj(img_tokens)
        img_indices = self.quantizer.f_to_idx(img_tokens)
        img_tokens = self.quantizer.idx_to_f(img_indices)
        img_tokens = self.post_quant_proj(img_tokens)
        features = img_tokens.mean(dim=1)
        features = self.projection(self.fc_norm(features))
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text_encoder(text)
        return F.normalize(features, dim=-1) if normalize else features

    def img_to_idx(self, img):
        features = self.encoder(img).float()
        features = self.quant_proj(features)
        return self.quantizer.f_to_idx(features)

    def idx_to_img(self, indices):
        features = self.quantizer.idx_to_f(indices)
        features = self.post_quant_proj(features)
        img = self.decoder(features).clamp_(-1, 1)
        return img

    def img_to_reconstructed_img(self, image) -> torch.Tensor:
        img_tokens = self.encoder(image)
        img_tokens = self.quant_proj(img_tokens)
        img_tokens, _, _, _ = self.quantizer(img_tokens)
        img_tokens = self.post_quant_proj(img_tokens)
        img_rec = self.decoder(img_tokens).clamp_(-1, 1)
        return img_rec

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True, unlock_text_proj=False):
        self.text.lock(unlocked_layers, freeze_layer_norm, unlock_text_proj)
        self.text_no_grad = True


if __name__ == '__main__':
    model = timm.create_model(
        'vitamin_base',
        patch_size=1,
        fc_norm=True,
        drop_rate=0.0,
        num_classes=0,
        global_pool='',
        pos_embed='none',
        class_token=False,
        mlp_layer=GeGluMlp,
        reg_tokens=0,
        img_size=256,
        drop_path_rate=0.1,
    )
    model.pos_embed = nn.Parameter(torch.zeros(1, 1, model.embed_dim), requires_grad=False)

    model_dict = model.state_dict()
    ckpt_dict = torch.load('ViTamin-B/pytorch_model.bin')
    visual_dict = dict()
    for k, v in ckpt_dict.items():
        if k.startswith('visual.'):
            if 'head' in k or 'pos_embed' in k:
                continue
            new_k = k.replace('visual.trunk.', '')
            visual_dict[new_k] = v

    model.load_state_dict(visual_dict, strict=False)
    print(set(model_dict.keys()) - set(visual_dict.keys()))
    print(set(visual_dict.keys() - set(model_dict.keys())))

