import torch.nn as nn
from typing import Tuple, Type
from .image_encoder import ImageEncoderViT, Block, Attention, add_decomposed_rel_pos, window_partition, window_unpartition
from .LoRA_layers import ConvLoRALinear


class ImageEncoderViTConvLoRA(ImageEncoderViT):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        hq = False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            hq=hq,
        )
        self.r = 4
        self.lora_alpha = 4
        self.conv_lora_expert_num = 8
        self.replaced_layers = []
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = BlockConvLoRA(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)
        self.apply_lora()
        self.print_replaced_layers()
        pass
    
    def apply_lora(self):
        for name, module in self.named_modules():
            # if name in ["blocks.0.attn", "blocks.1.attn"]:
            #     continue
            for proj_name in ["qkv"]:
                if hasattr(module, proj_name):
                    original_module = getattr(module, proj_name)
                    if isinstance(original_module, nn.Linear):
                        new_module = self.replace_with_lora(original_module)
                        setattr(module, proj_name, new_module)

                        self.replaced_layers.append((f"{name}.{proj_name}", original_module, new_module))

    def replace_with_lora(self, module):
        return ConvLoRALinear(
            module.in_features,
            module.out_features,
            r=self.r,
            lora_alpha=self.lora_alpha,
            conv_lora_expert_num=self.conv_lora_expert_num
        )


    def print_replaced_layers(self):
        print("\nReplaced Layers:")
        for layer_name, original_module, new_module in self.replaced_layers:
            print(f"{layer_name}: {original_module} -> {new_module}")
    
    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        all_moe_loss = 0.0

        if self.hq:
            interm_embeddings=[]
        for blk in self.blocks:
            x, moe_loss = blk(x)
            all_moe_loss += moe_loss
            if blk.window_size == 0 and self.hq:
                interm_embeddings.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        if self.hq:
            return x, interm_embeddings, all_moe_loss
        else:
            return x, all_moe_loss

class BlockConvLoRA(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn = AttentionConvLoRA(
            dim=kwargs["dim"],
            num_heads=kwargs["num_heads"],
            qkv_bias=kwargs["qkv_bias"],
            use_rel_pos=kwargs["use_rel_pos"],
            rel_pos_zero_init=kwargs["rel_pos_zero_init"],
            input_size=kwargs["input_size"] if kwargs["window_size"] == 0 else (kwargs["window_size"], kwargs["window_size"]),
        )
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        shortcut_norm = x
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        # modify for ConvLoRA
        x, moe_loss= self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        if self.adapter_type == 0 or self.adapter_type == 2:
            x = self.Space_Adapter_Series(x)
        elif self.adapter_type == 1:
            x = x + self.scale * self.Space_Adapter_Parallel(shortcut_norm)

        x = shortcut + x

        if self.adapter_type == 0:
            x = x + self.MLP_Adapter_Series(self.mlp(self.norm2(x)))

        elif self.adapter_type == 1 or self.adapter_type == 2:
            xn = self.norm2(x)
            x = x + self.mlp(xn) + self.scale * self.MLP_Adapter_Parallel(xn)
        else:
            x = x + self.mlp(self.norm2(x))

        return x, moe_loss
    
class AttentionConvLoRA(Attention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        # modify for ConvLoRA
        qkv, moe_loss = self.qkv(x)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x, moe_loss