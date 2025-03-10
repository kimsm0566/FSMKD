import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional
from torchvision.ops.misc import MLP
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param



class MLP(torch.nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU

class VisionTransformer(nn.Module):
    """Vision Transformer with Head, Body, Tail structure."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        # === Head ===
        self.head = nn.Conv2d(
            in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Class token and positional embedding
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.seq_length = seq_length + 1

        # === Body ===
        self.body = Encoder(
            seq_length=self.seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # === Tail ===
        tail_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            tail_layers["tail"] = nn.Linear(hidden_dim, num_classes)
        else:
            tail_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            tail_layers["act"] = nn.Tanh()
            tail_layers["tail"] = nn.Linear(representation_size, num_classes)

        self.tail = nn.Sequential(tail_layers)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the Head (Patch Embedding)."""
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size and w == self.image_size, f"Expected {self.image_size}x{self.image_size} images.")
        
        # Patchify and project input
        x = self.conv_proj(x)  # (n, hidden_dim, h/p, w/p)
        
        # Flatten spatial dimensions and permute to (batch_size, seq_length, hidden_dim)
        x = x.flatten(2).permute(0, 2, 1)  # (n, hidden_dim * h/p * w/p)

        # Add class token and positional embedding
        batch_class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x += self.pos_embedding
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_head(x)
        x = self.body(x)
        x = self.tail(x)

def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
    ) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 28)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


class Vit_Model(nn.Module):
    def __init__(self, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, num_classes, weights=None, progress=True):
        super().__init__()
        model = _vision_transformer(
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            weights=weights,
            progress=progress,
            num_classes=num_classes
        )
        self.head = model.head
        self.body = model.body
        self.tail = model.tail
        image_size = 28
        seq_length = (image_size // patch_size) ** 2

        # Class token and positional embedding
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length + 1, hidden_dim).normal_(std=0.02))
        self.seq_length = seq_length + 1

    def forward(self, x):
        x = self.head(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 클래스 토큰 추가
        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        
        x = x + self.pos_embedding
        x = self.body(x)
        x = self.tail(x[:, 0])
        return x
    
def VIT_client(args):
    model = Vit_Model(
            patch_size=4,
            num_layers=1,  # 클라이언트는 1-layer Transformer 사용
            num_heads=4,
            hidden_dim=64,
            mlp_dim=256,
            num_classes=10,  # 중간 표현(hidden representation) 크기
            weights=None,
            progress=True
        ).to(args.device)  # 디바이스로 이동
    return model

def VIT_server(args):
    model = Vit_Model(
            patch_size=4,
            num_layers=12,  # 클라이언트는 1-layer Transformer 사용
            num_heads=4,
            hidden_dim=64,
            mlp_dim=256,
            num_classes=10,  # 중간 표현(hidden representation) 크기
            weights=None,
            progress=True
                    ).to(args.device)  # 디바이스로 이동
    return model