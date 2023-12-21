import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        # Using separate layers for queries, keys, and values
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, x, v):
        # print(f"x: {x.size()}, v: {v.size()}")
        n, h, w, _ = x.size()
        
        x = x.permute(0, 3, 1, 2)  # Change the order to [batch, channels, height, width]
        v = v.permute(0, 3, 1, 2)
        
        keys = self.keys(v).reshape((n, self.key_channels, h * w))
        queries = self.queries(x).reshape(n, self.key_channels, h * w)
        values = self.values(v).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            # Apply softmax to keys and queries separately
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            # Compute context by multiplying key and value, and then multiply with query
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        # print(aggregated_values.size())
        reprojected_value = self.reprojection(aggregated_values)
        return reprojected_value + x

class SAMBLOCK(nn.Module):
    """ 
    Args:
        dim (int): Number of feature channels
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self,
                 dim,
                 num_heads,
                 key_channels,
                 v_dim,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.mlp_ratio = mlp_ratio
        act_layer=nn.GELU
        norm_layer=nn.LayerNorm

        self.norm1 = norm_layer(dim)
        self.normv = norm_layer(dim)

        self.attn = EfficientAttention(dim, key_channels=key_channels, head_count=num_heads, value_channels=v_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dim)
        mlp_hidden_dim = int(v_dim * mlp_ratio)
        self.mlp = Mlp(in_features=v_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, v, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        shortcut_v = v
        v = self.normv(v)
        v = v.view(B, H, W, C)

        # Efficient attention - directly use the entire feature map
        attn = self.attn(x, v)  # EfficientAttention call
        # reshape the attention
        attn = attn.view(B, H * W, self.v_dim)

        # FFN
        x = self.drop_path(attn) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, H, W


class SAM(nn.Module):
    def __init__(self,
                 input_dim=96,
                 embed_dim=96,
                 key_dim=64,
                 v_dim=64,
                 num_heads=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.embed_dim = embed_dim
        
        # Projection layers
        self.proj_e = nn.Conv2d(input_dim, embed_dim, 3, padding=1) if input_dim != embed_dim else None
        self.proj_q = nn.Conv2d(v_dim, embed_dim, 3, padding=1) if v_dim != embed_dim else None

        # Update v_dim to be the same as embed_dim
        v_dim = embed_dim
        self.sam_block = SAMBLOCK(
            dim=embed_dim,
            num_heads=num_heads,
            v_dim=v_dim,
            key_channels = key_dim,
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=norm_layer)

        # Normalization layer
        layer = norm_layer(embed_dim)
        layer_name = 'norm_sam'
        self.add_module(layer_name, layer)

    def forward(self, e, q):
        if self.proj_q is not None:
            q = self.proj_q(q)
        if self.proj_e is not None:
            e = self.proj_e(e)
        e_proj = e
        q_proj = q
        # print(f"Q: {q_proj.size()}, E: {e_proj.size()}")

        Wh, Ww = q.size(2), q.size(3)
        q = q.flatten(2).transpose(1, 2)
        e = e.flatten(2).transpose(1, 2)

        q_out, H, W = self.sam_block(q, e, Wh, Ww)
        norm_layer = getattr(self, f'norm_sam')
        q_out = norm_layer(q_out)
        q_out = q_out.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        return q_out+e_proj+q_proj