from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx


class DropPath(nnx.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        # super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def __call__(self, x, training: bool = True, rng: Optional[jax.Array] = None):
        if self.drop_prob == 0. or not training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Block(nnx.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    JAX UPDATE: Currently assuming that inputs will be (N, H, W, C)
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, *, rngs: nnx.Rngs):
        # super().__init__()
        self.dwconv = nnx.Conv(
            in_features=dim, 
            out_features=dim, 
            kernel_size=(7, 7), 
            padding=3, 
            feature_group_count=dim,
            rngs=rngs,
        ) # depthwise conv
        self.norm = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.pwconv1 = nnx.Linear(dim, 4 * dim, rngs=rngs) # pointwise/1x1 convs, implemented with linear layers
        self.act = lambda x: jax.nn.gelu(x, approximate=False) # Use exact GELU to match PyTorch
        self.pwconv2 = nnx.Linear(4 * dim, dim, rngs=rngs)
        self.gamma = nnx.Parameter(layer_scale_init_value * jnp.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path)

    def __call__(self, x, training: bool = True, rng: Optional[jax.Array] = None):
        inp = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.value * x
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = inp + self.drop_path(x)
        return x


class ConvNeXt(nnx.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., *,rngs: nnx.Rngs,
                 ):
        # super().__init__()

        self.downsample_layers = nnx.List() # stem and 3 intermediate downsampling conv layers
        stem = nnx.Sequential(
            nnx.Conv(in_features=in_chans, out_features=dims[0], kernel_size=(4, 4), strides=(4, 4), rngs=rngs),
            nnx.LayerNorm(dims[0], epsilon=1e-6, rngs=rngs),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nnx.Sequential(
                    nnx.LayerNorm(dims[i], epsilon=1e-6, rngs=rngs),
                    nnx.Conv(
                        in_features=dims[i],
                        out_features=dims[i + 1],
                        kernel_size=(2, 2),
                        strides=(2, 2),
                        rngs=rngs,
                    ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nnx.List() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = list(jnp.linspace(0, drop_path_rate, sum(depths)))
        cur = 0
        for i in range(4):
            stage = nnx.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, rngs=rngs) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nnx.LayerNorm(dims[-1], epsilon=1e-6, rngs=rngs) # final norm layer
        self.head = nnx.Linear(dims[-1], num_classes, rngs=rngs)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(jnp.mean(x, axis=(1, 2))) # global average pooling, (N, H, W, C) -> (N, C)
        x = self.head(x)
        return x
