import torch
import torch.nn as nn
from mmengine.model import (BaseModule)


# 复用你之前的 CrossAttentionBlock
class CrossAttention(nn.Module):
    """
    交叉注意力模块，用于处理两个序列之间的注意力交互
    Args:
        dim (int): 输入通道维度
        num_heads (int): 注意力头的数量
        qkv_bias (bool): 是否给 Q、K、V 的线性映射加偏置
        qk_scale (float): 缩放因子，如果为None则使用默认的head_dim^-0.5
        attn_drop (float): 注意力dropout率
        proj_drop (float): 输出投影的dropout率
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, C]
        Returns:
            cls_out: Tensor of shape [B, 1, C]
        """
        B, N, C = x.shape
        # Q 只取第 0 个 token（CLS）
        q = self.wq(x[:, :1]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # K, V 用所有 Token
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B,H,1,N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B,H,1,C/H
        x = x.transpose(1, 2).reshape(B, 1, C)  # B,1,C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """交叉注意力块，用于处理序列中的CLS token与其他token之间的交互，输入和输出都是 [B, C, N]
    Args:
        dim (int): 输入通道维度
        num_heads (int): 注意力头的数量
        mlp_ratio (float): MLP隐藏层维度与输入维度的比例，默认为4
        qkv_bias (bool): 是否给Q、K、V的线性映射加偏置
        qk_scale (float): 缩放因子，如果为None则使用默认的head_dim^-0.5
        drop (float): 输出投影的dropout率
        attn_drop (float): 注意力dropout率
        drop_path (float): 残差连接中的dropout率
        act_layer (nn.Module): 激活函数，默认为GELU
        norm_layer (nn.Module): 归一化层，默认为LayerNorm
        has_mlp (bool): 是否包含MLP层
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias,
                                   qk_scale, attn_drop, drop)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                act_layer(),
                nn.Linear(int(dim * mlp_ratio), dim),
                nn.Dropout(drop),
            )

    def forward(self, x):
        """对多层 Hilbert‑Flatten 后的序列做 CLS‑Token 级跨层 Cross‑Attention
            Args:
                x: Tensor of shape [B, C, N]
            Returns:
                Tensor of shape [B, C, N]  （仅第一个 token 被更新）
        """
        # 1) 归一化
        x_norm = self.norm1(x)
        # 2) 交叉注意力更新 CLS
        cls_updated = self.drop_path(self.attn(x_norm)) # [B,1,C]

        cls, rest = x[:, :1], x[:, 1:]
        cls = cls + cls_updated  # 残差连接（residual connection）
        # 3) 可选 MLP 残差
        if self.has_mlp:
            cls = cls + self.drop_path(self.mlp(self.norm2(cls)))
        # 4) 拼回序列，剩下 token 保持不变
        out_seq = torch.cat([cls, rest], dim=1)  # [B,N,C]
        # 5) 转回 [B,C,N]
        return out_seq


class HilbertCrossScaleAttention(BaseModule):
    """基于 Hilbert 曲线展开的多尺度跨层注意力模块
    
    Args:
        channels (int): 输入特征的通道数
        num_scales (int): 多尺度特征的数量，默认为4
        num_heads (int): 注意力头的数量，默认为8
        mlp_ratio (float): MLP隐藏层维度与输入维度的比例，默认为4.0
        qkv_bias (bool): 是否在QKV投影中使用偏置，默认为True
        qk_scale (float): 缩放QK点积的因子，默认为None
        drop (float): 全连接层的dropout率，默认为0.0
        attn_drop (float): 注意力dropout率，默认为0.0
        drop_path (float): 残差连接中的dropout率，默认为0.0
        init_cfg (dict): 初始化配置，默认为None
    """

    def __init__(self,
                 channels: int,
                 num_scales: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_scales = num_scales
        self.channels = channels

        # 每层一个可学习的 CLS token
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, channels))
            for _ in range(num_scales)
        ])

        # 把第 i 层的 CLS 投到 (i+1)%num_scales 层的通道维度
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.GELU()
            )
            for _ in range(num_scales)
        ])
        # 反投回自己的通道
        self.reprojs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.GELU()
            )
            for _ in range(num_scales)
        ])

        # 一组 CrossAttentionBlock 来做跨层 CLS‑Token 融合
        self.fusions = nn.ModuleList([
            CrossAttentionBlock(
                dim=channels, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                has_mlp=False
            )
            for _ in range(num_scales)
        ])

    def forward(self, seqs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            seqs: list 长度 = num_scales，
                  每项 shape = [B, C, L_i] （Hilbert‑Flatten 后）
        Returns:
            new_seqs: 同样长度的 list，每项 shape = [B, C, L_i]
        """
        B = seqs[0].shape[0]

        # 1) 构造带 CLS 的 token 序列 [B, L_i+1, C]
        tokens = []
        for i, x in enumerate(seqs):
            # [B, C, L] -> [B, L, C]
            t = x.permute(0, 2, 1)
            # prepend CLS
            cls = self.cls_tokens[i].expand(B, -1, -1)
            tokens.append(torch.cat([cls, t], dim=1))

        # 2) 跨层投影 & cross-attention
        fused_tokens = []
        for i in range(self.num_scales):
            j = (i + 1) % self.num_scales
            # 把 i 层的 CLS 投到 j 层的通道
            proj_cls = self.projs[i](tokens[i][:, :1, :])  # [B,1,C]
            # 构造跨层输入：proj_cls 作为 Query，j 层所有 patch token 作为 K/V
            inp = torch.cat([proj_cls, tokens[j][:, 1:, :]], dim=1)  # [B, L_j+?, C]
            # 交叉注意力
            out = self.fusions[i](inp)  # [B, L_j+?, C], 仅 CLS 更新
            # 把融合后的 CLS 投回 i 层通道
            new_cls = self.reprojs[i](out[:, :1, :])  # [B,1,C]
            # 拼回 i 层原始的 patch token
            fused_tokens.append(torch.cat([new_cls, tokens[i][:, 1:, :]], dim=1))

        # 3) 丢弃 CLS，恢复形状 [B, C, L_i]
        new_seqs = []
        for i, t in enumerate(fused_tokens):
            t = t[:, 1:, :].permute(0, 2, 1)  # -> [B, C, L_i]
            new_seqs.append(t)

        return new_seqs
