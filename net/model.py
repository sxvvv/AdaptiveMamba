import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from einops import rearrange, repeat
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# LoRA模块
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, bias=False, **kwargs):
        super().__init__()
        self.rank = rank
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 1 / rank
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        return F.linear(x, self.weight + lora_update, self.bias)


class AdaptiveMambaScan2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        use_lora=True,
        use_cache=True,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_lora = use_lora
        self.use_cache = use_cache

        # 输入投影
        if self.use_lora:
            self.in_proj = LoRALinear(self.d_model, self.d_inner * 2, rank=4, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 低秩投影
        self.x_proj_lowrank = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_inner, self.dt_rank, **factory_kwargs)),
            nn.Parameter(torch.randn(self.dt_rank, (self.dt_rank + self.d_state * 2), **factory_kwargs))
        ])

        # 缓存字典
        self.kv_cache_dict = {}

        # 路径权重
        self.path_weights = nn.Parameter(torch.ones(4, **factory_kwargs), requires_grad=True)
        self.out_norm = nn.LayerNorm(self.d_inner)

        if self.use_lora:
            self.out_proj = LoRALinear(self.d_inner, self.d_model, rank=4, bias=bias, **factory_kwargs)
        else:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def kv_cache_update(self, x, cache_key="default"):
        if not self.use_cache:
            return x
        if cache_key not in self.kv_cache_dict or self.training:
            self.kv_cache_dict[cache_key] = x.detach()
        else:
            x = x + self.kv_cache_dict[cache_key]
        return x

    def dynamic_feature_selection(self, x):
        importance_scores = torch.var(x, dim=-1)
        attention_weights = F.softmax(importance_scores, dim=-1)
        return x * attention_weights.unsqueeze(-1)

    def forward_core(self, x):
        B, C, H, W = x.shape
        cache_key_prefix = "forward_core"

        # 生成横向、竖向等路径特征
        x_hwwh = torch.stack([x.view(B, -1, H * W), x.permute(0, 1, 3, 2).reshape(B, -1, H * W)], dim=1)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        # 路径结果
        out_states = []
        for i in range(4):  # 4个路径
            cache_key = f"{cache_key_prefix}_{i}"
            xs_i = self.kv_cache_update(xs[:, i], cache_key)
            result = xs_i.view(B, C, H, W)  
            out_states.append(result)

        # 路径权重加权融合
        out_states = torch.stack(out_states, dim=1)
        path_probs = F.softmax(self.path_weights, dim=0).view(4, 1, 1, 1)
        return (out_states * path_probs).sum(dim=1)


    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        xz = self.dynamic_feature_selection(xz)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        out_y = self.forward_core(x)

        y = self.out_norm(out_y.permute(0, 2, 3, 1))
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        return out

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)


class AttBlock(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        
        # Activation
        self.act = nn.GELU() 
        
        # SS2D
        self.ss2d = AdaptiveMambaScan2D(d_model=chunk_dim, use_lora=True, use_cache=True)

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.ss2d(xc[i].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class AdaptiveBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.safm = AttBlock(dim) 
        # Feedforward layer
        self.ccm = FC(dim, ffn_scale) 

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class AdaptiveMamba(nn.Module):
    def __init__(self, dim=48, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Sequential(
            nn.Conv2d(3, dim // upscaling_factor, 3, 1, 1),
            nn.PixelUnshuffle(upscaling_factor)
        )
        out_dim = upscaling_factor * dim
        self.feats = nn.Sequential(*[AdaptiveBlock(out_dim, ffn_scale=2.0) for _ in range(8)])
        self.to_img = nn.Sequential(
            nn.Conv2d(out_dim, 3 * (upscaling_factor ** 2), 3, 1, 1),  # Modified to output 3 channels
            nn.PixelShuffle(upscaling_factor)
        )
        

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

