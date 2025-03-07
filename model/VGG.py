import torch
import torch.nn as nn
import random
from typing import List, Union, Dict, cast


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 64
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "G_BODY": [ "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C_BODY1": ["M", 128, "M", 256, "M", 512, "M"],
    "C_BODY2": ["M", 128, "M", 256, 256, "M", 512, "M", 512, "M"],
}

class VGG_Client(nn.Module):
    def __init__(
        self,  body: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        in_channels = 3
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = body
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.tail = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=dropout),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=dropout),
                    nn.Linear(4096, num_classes),
                )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.tail(x)
        return x
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
class VGG_Server(nn.Module):
    def __init__(
        self,body: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        in_channels = 3
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = body
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.tail = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=dropout),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=dropout),
                    nn.Linear(4096, num_classes),
                )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.tail(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
      

        
        
def VGG_client(args):
    num_classes = args.num_classes
    client_set = ["C_BODY1","C_BODY2"]
    cfg = random.choice(client_set)
    model = VGG_Client(make_layers(cfgs[cfg]), num_classes=num_classes)
    return model

def VGG_server(args):
    num_classes = args.num_classes
    model = VGG_Server(make_layers(cfgs["G_BODY"]), num_classes=num_classes)
    return model
