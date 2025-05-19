import torch.nn as nn

from .image_encoder import ImageEncoderViT
from .LoRA_layers import LoRALinear


class ImageEncoderViTLoRA(ImageEncoderViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.r = 4
        self.lora_alpha = 4
        self.replaced_layers = []
        self.apply_lora()
        self.print_replaced_layers()
        pass
    
    def apply_lora(self):
        for name, module in self.named_modules():
            for proj_name in ["qkv"]:
                if hasattr(module, proj_name):
                    original_module = getattr(module, proj_name)
                    if isinstance(original_module, nn.Linear):
                        new_module = self.replace_with_lora(original_module)
                        setattr(module, proj_name, new_module)

                        self.replaced_layers.append((f"{name}.{proj_name}", original_module, new_module))

    def replace_with_lora(self, module):
        return LoRALinear(
            module.in_features,
            module.out_features,
            r=self.r,
            lora_alpha=self.lora_alpha
        )

    def print_replaced_layers(self):
        print("\nReplaced Layers:")
        for layer_name, original_module, new_module in self.replaced_layers:
            print(f"{layer_name}: {original_module} -> {new_module}")