import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any
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

class MultiResolutionProcessor:
    """
    多分辨率图像处理器
    - 训练时：用于batch中不同尺寸图像的padding和masking
    - 推理时：用于处理任意尺寸图像（裁剪到16的倍数，不使用padding）
    """
    
    def __init__(self, max_size: int = 256, patch_size: int = 16):
        self.max_size = max_size
        self.patch_size = patch_size
    
    def process_image_for_inference(self, image: Image.Image) -> Dict[str, Any]:
        """
        推理时的图像处理：直接裁剪到16的倍数，不使用padding
        
        Args:
            image: PIL图像
            
        Returns:
            处理结果字典，包含处理后的图像和相关信息
        """
        # 获取原始尺寸
        original_size = image.size  # (width, height)
        
        # 保持长宽比缩放到合适尺寸
        aspect_ratio = original_size[0] / original_size[1]
        
        if aspect_ratio > 1:  # 宽图
            new_width = min(self.max_size, original_size[0])
            new_height = int(new_width / aspect_ratio)
        else:  # 高图或方图
            new_height = min(self.max_size, original_size[1])
            new_width = int(new_height * aspect_ratio)
        
        # 确保尺寸是patch_size的倍数（通过裁剪）
        new_width = (new_width // self.patch_size) * self.patch_size
        new_height = (new_height // self.patch_size) * self.patch_size
        
        # 确保最小尺寸
        new_width = max(new_width, self.patch_size)
        new_height = max(new_height, self.patch_size)
        
        # 缩放图像
        processed_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 计算patch数量
        patch_w = new_width // self.patch_size
        patch_h = new_height // self.patch_size
        
        shape_info = {
            'original_size': original_size,
            'processed_size': (new_width, new_height),
            'patch_w': patch_w,
            'patch_h': patch_h,
        }
        
        return {
            'image': processed_image,
            'shape_info': shape_info,
            'attention_mask': None,  # 推理时不需要attention mask
        }
    
    def process_image_for_training(self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        训练时的图像处理：padding到统一尺寸用于batch处理
        
        Args:
            image: PIL图像
            target_size: 目标尺寸，用于batch中的padding对齐
            
        Returns:
            处理结果字典，包含处理后的图像、attention mask和相关信息
        """
        # 如果没有指定目标尺寸，使用max_size
        if target_size is None:
            target_size = (self.max_size, self.max_size)
        
        original_size = image.size  # (width, height)
        
        # 保持长宽比缩放
        aspect_ratio = original_size[0] / original_size[1]
        target_w, target_h = target_size
        
        # 计算缩放后的尺寸（保持长宽比，适配目标尺寸）
        if aspect_ratio > (target_w / target_h):  # 图像更宽
            new_width = target_w
            new_height = int(target_w / aspect_ratio)
        else:  # 图像更高或相等
            new_height = target_h
            new_width = int(target_h * aspect_ratio)
        
        # 确保是patch_size的倍数
        new_width = (new_width // self.patch_size) * self.patch_size
        new_height = (new_height // self.patch_size) * self.patch_size
        
        # 确保最小尺寸
        new_width = max(new_width, self.patch_size)
        new_height = max(new_height, self.patch_size)
        
        # 缩放图像
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 计算padding
        pad_left = (target_w - new_width) // 2
        pad_right = target_w - new_width - pad_left
        pad_top = (target_h - new_height) // 2
        pad_bottom = target_h - new_height - pad_top
        
        # 创建padded图像
        padded_image = Image.new('RGB', target_size, (128, 128, 128))  # 灰色padding
        padded_image.paste(resized_image, (pad_left, pad_top))
        
        # 创建attention mask
        patch_w = target_w // self.patch_size
        patch_h = target_h // self.patch_size
        attention_mask = torch.zeros(patch_h, patch_w)
        
        # 标记实际图像区域
        start_patch_x = pad_left // self.patch_size
        end_patch_x = (pad_left + new_width) // self.patch_size
        start_patch_y = pad_top // self.patch_size
        end_patch_y = (pad_top + new_height) // self.patch_size
        
        attention_mask[start_patch_y:end_patch_y, start_patch_x:end_patch_x] = 1
        
        shape_info = {
            'original_size': original_size,
            'processed_size': (new_width, new_height),
            'target_size': target_size,
            'padding': (pad_left, pad_top, pad_right, pad_bottom),
            'patch_w': patch_w,
            'patch_h': patch_h,
            'valid_patch_w': end_patch_x - start_patch_x,
            'valid_patch_h': end_patch_y - start_patch_y,
        }
        
        return {
            'image': padded_image,
            'shape_info': shape_info,
            'attention_mask': attention_mask,
        }
    
    def process_image(self, image: Image.Image, training: bool = False, target_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        统一的图像处理接口
        
        Args:
            image: PIL图像
            training: 是否为训练模式
            target_size: 训练时的目标尺寸
        """
        if training:
            return self.process_image_for_training(image, target_size)
        else:
            return self.process_image_for_inference(image)
    
    def postprocess_image(self, reconstructed_img: torch.Tensor, shape_info: Dict[str, Any]) -> torch.Tensor:
        """
        后处理重建图像
        
        Args:
            reconstructed_img: 重建的图像tensor [B, C, H, W]
            shape_info: 形状信息
            
        Returns:
            处理后的图像tensor
        """
        if 'padding' in shape_info:
            # 训练模式：去除padding
            pad_left, pad_top, pad_right, pad_bottom = shape_info['padding']
            processed_size = shape_info['processed_size']
            
            # 提取有效区域
            h_start = pad_top
            h_end = pad_top + processed_size[1]
            w_start = pad_left
            w_end = pad_left + processed_size[0]
            
            cropped_img = reconstructed_img[:, :, h_start:h_end, w_start:w_end]
            
            # 缩放回原始尺寸
            original_size = shape_info['original_size']
            final_img = F.interpolate(
                cropped_img, 
                size=(original_size[1], original_size[0]),  # (height, width)
                mode='bilinear', 
                align_corners=False
            )
            
            return final_img
        else:
            # 推理模式：直接缩放回原始尺寸
            original_size = shape_info['original_size']
            final_img = F.interpolate(
                reconstructed_img,
                size=(original_size[1], original_size[0]),  # (height, width)
                mode='bilinear',
                align_corners=False
            )
            
            return final_img


class MultiResolutionBatch:
    """处理批量多分辨率图像的工具类"""
    
    @staticmethod
    def collate_fn(batch_data):
        """
        自定义collate函数，将多个不同尺寸的图像组合成batch
        
        Args:
            batch_data: List of processed image dictionaries
            
        Returns:
            batched_data: 包含batched tensors和metadata的字典
        """
        # 找到batch中的最大尺寸
        max_h = max(data['shape_info']['processed_size'][1] for data in batch_data)
        max_w = max(data['shape_info']['processed_size'][0] for data in batch_data)
        
        batch_size = len(batch_data)
        batch_images = []
        batch_masks = []
        batch_shape_infos = []
        
        for data in batch_data:
            image = data['image']
            mask = data['attention_mask']
            shape_info = data['shape_info']
            
            # Convert PIL to tensor if needed
            if isinstance(image, Image.Image):
                import torchvision.transforms as T
                image = T.ToTensor()(image) * 2.0 - 1.0  # normalize to [-1, 1]
            
            # Pad image to max size
            curr_h, curr_w = image.shape[1], image.shape[2]
            pad_h = max_h - curr_h
            pad_w = max_w - curr_w
            
            if pad_h > 0 or pad_w > 0:
                image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
            
            # Pad mask to max size (in patch coordinates)
            mask_pad_h = (max_h // 16) - mask.shape[0]  # assuming patch_size=16
            mask_pad_w = (max_w // 16) - mask.shape[1]
            
            if mask_pad_h > 0 or mask_pad_w > 0:
                mask = F.pad(mask, (0, mask_pad_w, 0, mask_pad_h), value=0)
            
            batch_images.append(image)
            batch_masks.append(mask)
            batch_shape_infos.append(shape_info)
        
        batch_images = torch.stack(batch_images, dim=0)
        batch_masks = torch.stack(batch_masks, dim=0)
        
        return {
            'images': batch_images,
            'attention_masks': batch_masks,
            'shape_infos': batch_shape_infos,
            'batch_size': batch_size,
        }


# 使用示例
if __name__ == "__main__":
    # 创建处理器
    processor = MultiResolutionProcessor(max_size=256, patch_size=16)
    
    # 处理单张图像
    image = Image.open("test_image.jpg")
    result = processor.process_image(image)
    
    print(f"Original size: {result['original_size']}")
    print(f"Processed size: {result['processed_size']}")
    print(f"Attention mask shape: {result['attention_mask'].shape}")
    print(f"Shape info: {result['shape_info']}")