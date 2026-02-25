import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from models.mamba.multi_mamba import MultiScan
from .unet_parts import *

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    if torch.__version__ > '2.0.0':
        from selective_scan_vmamba_pt202 import selective_scan_cuda_core
    else:
        from selective_scan_vmamba import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

# 我们最终的模型，是后缀big系列，用unet替代patch embedding
# SSMODE = "mamba_ssm"
# import selective_scan_cuda

# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


class SelectiveScan(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


"""
Local Mamba
"""


class MultiScanVSSM(MultiScan):
    ALL_CHOICES = MultiScan.ALL_CHOICES

    def __init__(self, dim, choices=None, win_size=8):
        super().__init__(dim, choices=choices, token_size=None, win_size=win_size)
        self.attn = BiAttn(dim)
        # 每个扫描方向设置一个权重
        self.K = len(choices)
        self.dir_weights = torch.nn.Parameter(torch.ones(self.K))

    # def merge(self, xs):# 消融-无扫描加权
    #     # xs: [B, K, D, L]
    #     # return: [B, D, L]
    #
    #     # remove the padded tokens
    #     xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
    #     xs = super().multi_reverse(xs)
    #     # xs = [self.attn(x.transpose(-2, -1)) for x in xs]
    #     xs = [x.transpose(-2, -1) for x in xs]
    #     x = super().forward(xs)
    #
    #     return x

    # our
    def merge(self, xs):
        # xs: [B, K, D, L] = [12, 6, 192, 4096]
        B, K, D, L = xs.shape  # B=12, K=6, D=192, L=4096
        # 1. remove padded tokens -> list of 6 tensors of shape [12, 192, Li]
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        # 2. reverse scan for each direction
        xs = super().multi_reverse(xs)  # still 6 tensors
        # 3. transpose each to [B, L, D]
        xs = [x.transpose(-2, -1) for x in xs]  # each: [12, Li, 192]
        # ----------- NEW: learnable weighted fusion -----------
        # stack to [B, K, L, D]
        # 注意：不同方向 Li 不一样，因此需要 pad 以对齐
        max_len = max(x.shape[1] for x in xs)
        xs_padded = []
        for x in xs:
            if x.shape[1] < max_len:
                pad_len = max_len - x.shape[1]
                x = F.pad(x, (0, 0, 0, pad_len))  # pad on length dimension
            xs_padded.append(x)
        # xs: [B, K, max_len, D]
        xs = torch.stack(xs_padded, dim=1)
        # 计算 softmax 权重 [K] → [1,K,1,1]
        weights = torch.softmax(self.dir_weights, dim=0).view(1, self.K, 1, 1)
        # weights = F.gumbel_softmax(self.dir_weights, tau=1.0, hard=False, dim=0).view(1, self.K, 1, 1)

        # 加权加和: [B,K,L,D] * [1,K,1,1] → [B,L,D]
        xs = (xs * weights)
        # 4. 送回父类 forward（它需要 [B, D, L]）
        xs = [x for x in xs.transpose(0, 1)]

        x = super().forward(xs)
        # x is [B, L, D] = [12, 4096, 192]
        return x



    def multi_scan(self, x):
        # x: [B, C, H, W]
        # return: [B, K, C, H * W]
        B, C, H, W = x.shape
        self.token_size = (H, W)

        xs = super().multi_scan(x)  # [[B, C, L], ...]

        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)

        # pad the tokens into the same length as VMamba compute all directions together
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('MultiScanVSSM', f'MultiScanVSSM[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn  # * s_attn  # [B, N, C]
        out = ori_x * attn
        return out


def multi_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        multi_scan=None,
        win_size=8,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = multi_scan.multi_scan(x)

    L = xs.shape[-1]
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)  # l fixed

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)

    y = multi_scan.merge(ys)

    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            directions=['h', 'h_flip', 'v', 'v_flip'],
            win_size=8,
            # fusion args
            use_fusion: bool = True,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.use_fusion = use_fusion

        self.out_norm = nn.LayerNorm(d_inner)

        self.K = len(MultiScanVSSM.ALL_CHOICES) if directions is None else len(directions)
        self.K2 = self.K

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.win_size = win_size
        # Local Mamba
        self.multi_scan = MultiScanVSSM(d_expand, choices=directions, win_size=self.win_size)

        # ===== 新增 α / β 模块 =====
        if self.use_fusion:

            self.cross_proj = nn.Sequential(
                nn.Conv2d(d_expand*2, d_expand // 2, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_expand // 2, d_expand, 1, bias=False),
            )

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = multi_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm,
            nrows=nrows, delta_softplus=True, multi_scan=self.multi_scan, win_size=self.win_size
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, other_feat: torch.Tensor = None):
        """
        Args:
            x: 输入特征 [B, H, W, C]
            return_mid: 若为 True，则同时返回中间特征 y
        Returns:
            out: 最终输出特征
            y: 中间特征 (仅当 return_mid=True 时)
        """
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)
            z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
        else:
            xz = self.act(xz)
            x, z = xz.chunk(2, dim=-1)

        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z

        # # ===== 在此处插入 α / β 融合 =====
        # if self.use_fusion and (other_feat is not None):
        #     if other_feat.shape[1] != y.shape[1]:
        #         other_feat = self.cross_proj(other_feat)
        #     y = y.permute(0, 3, 1, 2).contiguous() # B D H W
        #     other_feat = other_feat.permute(0, 3, 1, 2).contiguous()
        #     alpha = self.gate_alpha(y)
        #     beta = self.gate_beta(other_feat)
        #     y = alpha * y + beta * other_feat
        #     y = y.permute(0, 2, 3, 1).contiguous()# B H W D

        # ===== 轻量 Cross-Attention 融合（L-CMA） =====
        if self.use_fusion and (other_feat is not None):
            # 投影 other_feat 以匹配通道数
            if other_feat.shape[1] != y.shape[1]:
                other_feat = self.cross_proj(other_feat)
            # B H W D → B D H W
            y = y.permute(0, 3, 1, 2).contiguous()
            other_feat = other_feat.permute(0, 3, 1, 2).contiguous()
            # ---- 轻量注意力：concat → 1×1 conv → sigmoid ----
            # cross_proj 输入通道 = 2*D，输出 = D
            att = torch.sigmoid(self.cross_proj(torch.cat([y, other_feat], dim=1)))
            # 注意力引导融合
            y = y + att * other_feat
            # B D H W → B H W D
            y = y.permute(0, 2, 3, 1).contiguous()

        out = self.dropout(self.out_proj(y))

        return out, y  # <=== 新增输出中间特征


class VSSBlockFusion(nn.Module):
    """
    改进版 VSSBlock：
    - 支持模态间中间特征融合 (alpha/beta gating)
    - 可返回自身中间特征以供下一层融合使用
    """

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            win_size=8,
            # =============================
            use_checkpoint: bool = False,
            # directions=['h', 'h_flip', 'v', 'v_flip'],# 消融
            directions=['h', 'h_flip', 'v', 'v_flip', 'c2', 'c5'],
            # 新增参数
            use_fusion: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_fusion = use_fusion

        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            simple_init=ssm_simple_init,
            directions=directions,
            win_size=win_size,
            use_fusion=use_fusion,
        )
        self.drop_path = DropPath(drop_path)

    def _forward(self, input: torch.Tensor, other_mid: torch.Tensor = None):
        """
        :param input: BHWC
        :param other_mid: BHWC or None
        :return: out (BHWC), mid_feat (BHWC), alpha (BCHW), beta (BCHW)
        """
        x_norm = self.norm(input)
        # SS2D handles fusion internally and returns (out, mid)
        out_main, mid_feat = self.op(x_norm, other_feat=other_mid)
        # residual & drop path (out_main is BHWC)
        out = input + self.drop_path(out_main)
        # return out (BHWC), mid_feat (BHWC)
        return out, mid_feat

    def forward(self, x: torch.Tensor, other_mid: torch.Tensor = None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, other_mid)
        else:
            return self._forward(x, other_mid)


# -----------------------
# Permute
# -----------------------
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


# -----------------------
# Patch Expand
# -----------------------
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand1 = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.expand2 = nn.Linear(dim_scale *self.dim, dim_scale *dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand1(x)
        x = self.expand2(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C)
        x = self.norm(x)

        return x


# -----------------------
# CCViMFusionVNet
# -----------------------
class CCViMFusionVNet(nn.Module):
    def __init__(
            self,
            in_chans=1,
            num_classes=192,
            dims=[96, 192, 384, 768],
            dims_decoder=[768, 384, 192, 96],
            depths=[2, 2, 2, 2],
            depths_decoder=[2, 2, 2, 2],
            patch_size=4,
            patch_norm=True,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dims = dims
        self.dims_decoder = dims_decoder

        # -----------------------
        # Patch Embedding
        # -----------------------
        # self.patch_embed_mr = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], patch_size, patch_size),
        #     Permute(0, 2, 3, 1),
        #     norm_layer(dims[0]) if patch_norm else nn.Identity()
        # )
        # self.patch_embed_pet = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], patch_size, patch_size),
        #     Permute(0, 2, 3, 1),
        #     norm_layer(dims[0]) if patch_norm else nn.Identity()
        # )
        # self.patch_embed_prior = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], patch_size, patch_size),
        #     Permute(0, 2, 3, 1),
        #     norm_layer(dims[0]) if patch_norm else nn.Identity()
        # )

        self.patch_embed_mr = DoubleConv(in_chans, 32)
        self.patch_embed_pet = DoubleConv(in_chans, 32)
        self.patch_embed_prior = DoubleConv(in_chans, 32)
        self.ddown1_mr = Down(32, 64)      # 256 → 128
        self.ddown2_mr = Down(64, dims[0])     # 128 → 64
        # self.ddown3_mr = Down(dims[0], dims[1])     # 64 → 32
        self.ddown1_pet = Down(32, 64)      # 256 → 128
        self.ddown2_pet = Down(64, dims[0])     # 128 → 64
        # self.ddown3_pet = Down(dims[0], dims[1])     # 64 → 32
        self.ddown1_prior = Down(32, 64)      # 256 → 128
        self.ddown2_prior = Down(64, dims[0])     # 128 → 64
        # self.ddown3_prior = Down(dims[0], dims[1])     # 64 → 32


        # -----------------------
        # Encoder Layer0: 三路，每路3 block
        # -----------------------
        # self.layers0_mr = nn.ModuleList([VSSBlockFusion(dims[0]) for _ in range(depths[0])])
        # self.layers0_pet = nn.ModuleList([VSSBlockFusion(dims[0]) for _ in range(depths[0])])
        # self.layers0_prior = nn.ModuleList([VSSBlockFusion(dims[0]) for _ in range(depths[0])])

        # -----------------------
        # Encoder Layer1: 三路，每路2 block, Prior->MR
        # -----------------------
        self.layers1_mr = nn.ModuleList([VSSBlockFusion(dims[1]) for _ in range(depths[1])])
        self.layers1_pet = nn.ModuleList([VSSBlockFusion(dims[1]) for _ in range(depths[1])])
        self.layers1_prior = nn.ModuleList([VSSBlockFusion(dims[1]) for _ in range(depths[1])])

        self.down_mr = PatchMerging2D(dims[0], dims[1])
        self.down_pet = PatchMerging2D(dims[0], dims[1])
        self.down_prior = PatchMerging2D(dims[0], dims[1])

        # -----------------------
        # Encoder Layer2: MR/PET
        # -----------------------
        self.layers2_mr = nn.ModuleList([VSSBlockFusion(dims[2]) for _ in range(depths[2])])
        self.layers2_pet = nn.ModuleList([VSSBlockFusion(dims[2]) for _ in range(depths[2])])

        self.down2_mr = PatchMerging2D(dims[1], dims[2])
        self.down2_pet = PatchMerging2D(dims[1], dims[2])

        # -----------------------
        # Encoder Layer3: 单路
        # -----------------------
        layer3_dirs_list = [['h_flip', 'v', 'c2', 'c5'], ['h_flip', 'v_flip', 'c5', 'c5'], ['h', 'v', 'c5', 'c5']]
        # self.layers3 = nn.ModuleList([VSSBlockFusion(dims[3],directions=layer3_dirs_list[i]) for i in range(depths[3])])
        self.layers3 = nn.ModuleList([VSSBlockFusion(dims[3]) for i in range(depths[3])])

        self.down3 = PatchMerging2D(dims[2], dims[3])

        # -----------------------
        # Decoder
        # -----------------------
        delayer3_dirs_list = [['h', 'h_flip', 'c5', 'c5'],
                              ['h_flip', 'v_flip', 'c2', 'c2'],
                              ['h_flip', 'v_flip', 'c2', 'c5']]
        self.up3 = PatchExpand2D(dims_decoder[0], 2)
        # self.deLayer3 = nn.ModuleList([VSSBlockFusion(dims_decoder[0],directions=delayer3_dirs_list[i]) for i in range(depths_decoder[0])])
        self.deLayer3 = nn.ModuleList([VSSBlockFusion(dims_decoder[0]) for i in range(depths_decoder[0])])

        self.up2 = PatchExpand2D(dims_decoder[1], 2)
        self.deLayer2_pet = nn.ModuleList([VSSBlockFusion(dims_decoder[1]) for _ in range(depths_decoder[1])])
        self.deLayer2_skip = nn.ModuleList([VSSBlockFusion(dims_decoder[1]) for _ in range(depths_decoder[1])])

        self.up1 = PatchExpand2D(dims_decoder[2], 2)
        self.deLayer1_pet = nn.ModuleList([VSSBlockFusion(dims_decoder[2]) for _ in range(depths_decoder[2])])
        self.deLayer1_skip = nn.ModuleList([VSSBlockFusion(dims_decoder[2]) for _ in range(depths_decoder[2])])

        # self.final_up = Final_PatchExpand2D(dims_decoder[-1], 4, norm_layer)
        # self.deLayer0_pet = nn.ModuleList([VSSBlockFusion(dims_decoder[-1]) for _ in range(depths_decoder[-1])])
        # self.deLayer0_skip = nn.ModuleList([VSSBlockFusion(dims_decoder[-1]) for _ in range(depths_decoder[-1])])

        self.upp1 = Up(160, 128)
        self.upp2 = Up(160, dims_decoder[-1])
        # self.upp3 = Up(124, dims_decoder[-1])

        self.final_conv = nn.Conv2d(dims_decoder[-1] , num_classes, 1)

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, mr, pet, prior):
        # Patch embedding
        x_mr = self.patch_embed_mr(mr)
        x_pet = self.patch_embed_pet(pet)
        x_prior = self.patch_embed_prior(prior)

        x_pet1 = self.ddown1_pet(x_pet)
        x_pet2 = self.ddown2_pet(x_pet1)
        # x_pet3 = self.ddown3_pet(x_pet2)
        x_pet2 = x_pet2.permute(0, 2, 3, 1)

        x_mr0 = x_mr.clone()
        x_mr1 = self.ddown1_mr(x_mr)
        x_mr2 = self.ddown2_mr(x_mr1)
        # x_mr3 = self.ddown3_mr(x_mr2)
        x_mr2 = x_mr2.permute(0, 2, 3, 1)

        x_prior1 = self.ddown1_prior(x_prior)
        x_prior2 = self.ddown2_prior(x_prior1)
        # x_prior3 = self.ddown3_prior(x_prior2)
        x_prior2 = x_prior2.permute(0, 2, 3, 1)

        # Layer0: 三路并行
        # for blk_idx in range(len(self.layers0_mr)):
        #     x_mr, _ = self.layers0_mr[blk_idx](x_mr)
        #     x_pet, _ = self.layers0_pet[blk_idx](x_pet)
        #     x_prior, _ = self.layers0_prior[blk_idx](x_prior)
        # skip0 = x_mr


        # Layer1: 下采样 + Prior->MR
        x_mr = self.down_mr(x_mr2)
        x_pet = self.down_pet(x_pet2)
        x_prior = self.down_prior(x_prior2)
        for blk_idx in range(len(self.layers1_mr)):
            x_prior, x_prior_mid = self.layers1_prior[blk_idx](x_prior)
            x_mr, _ = self.layers1_mr[blk_idx](x_mr, other_mid=x_prior_mid)
            # x_mr, _ = self.layers1_mr[blk_idx](x_mr)# 消融-只有pet+mr
            x_pet, _ = self.layers1_pet[blk_idx](x_pet)
        skip1 = x_mr

        # Layer2: PET->MR
        x_mr = self.down2_mr(x_mr)
        x_pet = self.down2_pet(x_pet)
        for blk_idx in range(len(self.layers2_mr)):
            x_pet, x_pet_mid = self.layers2_pet[blk_idx](x_pet)
            x_mr, _ = self.layers2_mr[blk_idx](x_mr, other_mid=x_pet_mid)
            # x_mr, _ = self.layers2_mr[blk_idx](x_mr)  # 消融-只有mr
        skip2 = x_pet
        # skip2 = x_mr # 消融-只有mr

        # Layer3: 单路
        x = self.down3(x_mr)
        for blk in self.layers3:
            x, _ = blk(x)

        # -----------------------
        # Decoder
        # -----------------------
        for blk in self.deLayer3:
            x, _ = blk(x)
        x = self.up3(x)

        for blk_idx in range(len(self.deLayer2_pet)):
            # x_skip2, x_skip2_mid = self.deLayer2_pet[blk_idx](skip2)
            # x, _ = self.deLayer2_pet[blk_idx](x, other_mid=x_skip2_mid)
            # if blk_idx == 0:
            #     x = torch.cat([x, skip2], dim=-1)
            # x, _ = self.deLayer2_pet[blk_idx](x)
            if blk_idx == 0:
                x_skip, x_skip_mid = self.deLayer2_skip[blk_idx](skip2)
            else:
                x_skip, x_skip_mid = self.deLayer2_skip[blk_idx](x_skip)
            x, _ = self.deLayer2_pet[blk_idx](x, other_mid=x_skip_mid)

        x = self.up2(x)

        for blk_idx in range(len(self.deLayer1_pet)):
            # x_skip1, x_skip1_mid = self.deLayer1_pet[blk_idx](skip1)
            # x, _ = self.deLayer1_pet[blk_idx](x, other_mid=x_skip1_mid)
            # if blk_idx == 0:
            #     x = torch.cat([x, skip1], dim=-1)
            # x, _ = self.deLayer1_pet[blk_idx](x)
            if blk_idx == 0:
                x_skip, x_skip_mid = self.deLayer1_skip[blk_idx](skip1)
            else:
                x_skip, x_skip_mid = self.deLayer1_skip[blk_idx](x_skip)
            x, _ = self.deLayer1_pet[blk_idx](x, other_mid=x_skip_mid)

        x = self.up1(x)

        # for blk_idx in range(len(self.deLayer0_pet)):
        #     if blk_idx == 0:
        #         x_skip, x_skip_mid = self.deLayer0_skip[blk_idx](skip0)
        #     else:
        #         x_skip, x_skip_mid = self.deLayer0_skip[blk_idx](x_skip)
        #     x, _ = self.deLayer0_pet[blk_idx](x, other_mid=x_skip_mid)

        # x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x_mr2 = x_mr2.permute(0, 3, 1, 2)
        # x = self.upp1(x,x_mr3)
        x = self.upp1(x,x_mr1)
        x = self.upp2(x,x_mr0)
        x = self.final_conv(x)
        return x


# # VSSBlockFusion
# if __name__ == "__main__":
#     from thop import profile
#     from torchsummary import summary
#     # 创建模型实例
#     model = VSSBlockFusion(
#         hidden_dim=96,
#         use_fusion=True,   # 开启模态融合
#     ).cuda()
#     # 打印模型结构
#     print("=== Model Summary ===")
#     summary(model, [(64, 64, 96), (64, 64, 192)])  # 需要两个输入 (x, other_feat)
#     # 模拟输入
#     x = torch.randn(1, 64, 64, 96).cuda()          # 当前模态 (B, C, H, W)
#     other = torch.randn(1, 64, 64, 192).cuda()      # 另一模态
#     # 前向传播
#     out, mid = model(x, other)
#     print(f"Output shape: {out.shape}")
#     if mid is not None:
#         print(f"Mid feature shape: {mid.shape}")
#     # === FLOPs & Params ===
#     # 注意：thop 不支持多输入，需要包装成 lambda
#     flops, params = profile(model, inputs=(x, other), verbose=False)
#     print(f"FLOPs: {flops / 1e9:.3f} G")
#     print(f"Params: {params / 1e6:.3f} M")
#     # 测试不使用融合的情况
#     print("\n=== Test without fusion ===")
#     model.use_fusion = False
#     out, mid = model(x)
#     print(f"Output shape (no fusion): {out.shape}")
#     print(f"Mid feature: {mid}")

# ccvimfusionvnet
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # 创建模型
    # ---------------------------
    model = CCViMFusionVNet().to(device)

    # ---------------------------
    # 打印模型参数和结构
    # ---------------------------
    print("Model Summary:")
    # summary 需要 NCHW 输入
    # summary(model, [(1, 256, 256), (1, 256, 256), (1, 256, 256)])

    # ---------------------------
    # 构造假数据
    # ---------------------------
    B = 2  # batch size
    H = W = 256
    x_mr = torch.randn(B, 1, H, W).to(device)
    x_pet = torch.randn(B, 1, H, W).to(device)
    x_prior = torch.randn(B, 1, H, W).to(device)

    # ---------------------------
    # 前向传播
    # ---------------------------
    with torch.no_grad():
        out = model(x_mr, x_pet, x_prior)

    print(f"Input shapes: MR:{x_mr.shape}, PET:{x_pet.shape}, Prior:{x_prior.shape}")
    print(f"Output shape: {out.shape}")


