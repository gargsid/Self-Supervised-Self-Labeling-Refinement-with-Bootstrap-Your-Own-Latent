import torch
from torch import nn
import torch.nn.functional as F
import copy
from torchvision import models


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.LayerNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = Conv2d(inplanes, planes, 3, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv3x3(planes, planes, 3)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # out = self.conv1(x)
        out = self.conv1.forward(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        out = self.conv2.forward(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.LayerNorm
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv1x1(inplanes, width)
        self.conv1 = Conv2d(inplanes, width,3)
        self.bn1 = norm_layer(width)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv2 = Conv2d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # self.conv3 = conv1x1(width, planes * self.expansion)
        self.conv3 = Conv2d(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # out = self.conv1(x)
        out = self.conv1.forward(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        out = self.conv2.forward(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        out = self.conv3.forward(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.LayerNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        # self.conv1 = nn.Conv2d(
        #     3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        # )

        self.conv1 = Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        x = self.conv1.forward(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load('/content/drive/MyDrive/resnet18.pt')
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=True, progress=True, device="gpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )



class ByolBackBone(nn.Module):
    def __init__(self, dropout=0.15):
        super().__init__()

        self.dropout = dropout
        self.backbone = self._get_backbone()

    def _get_backbone(self):
        
        model = resnet18(pretrained=False)
        modules = list(model.children())[:-2] # delete the layer4, avgpool and fc layer.
        backbone = nn.Sequential(*modules)
        dropout_layer = nn.Dropout(self.dropout)
        mdls_with_drpt = []
        backbone_modules = backbone.children()
        for i, m in enumerate(backbone_modules):
            if isinstance(m, nn.Sequential):
                subm = list(m.children())
                for j, sm in enumerate(subm):
                    mdls_with_drpt.append(sm)
                    if i >= 4:
                        mdls_with_drpt.append(dropout_layer)
            else:
                mdls_with_drpt.append(m)
                if i >= 4:
                    mdls_with_drpt.append(dropout_layer)

        backbone_with_dropout = nn.Sequential(*mdls_with_drpt)
        return backbone_with_dropout

    def forward(self, x):
        return self.backbone(x)

class MLPHead(nn.Module):
    def __init__(self, in_channels, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size)
        )

        self._init_weight()

    def forward(self, x):
        return self.net(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print("init decoder")
                nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ProjectorHead(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.out_channels = out_features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = MLPHead(in_features, hidden_size, out_features)

    def forward(self, x):
        x_pooled = self.avg_pool(x)
        h = x_pooled.view(x_pooled.shape[0], x_pooled.shape[1])   # removing the last dimension
        return self.projection(h)

class BYOL(nn.Module):
    def __init__(self, target_momentum=0.996):
        super().__init__()

        self.online_network = ByolBackBone()
        self.target_network = ByolBackBone()

        # Projection Head
        self.online_projector = ProjectorHead(512, 512, 512)
        self.target_projector = ProjectorHead(512, 512, 512)

        # Predictor Head
        self.predictor = MLPHead(self.online_projector.out_channels, 512, 512)

        self.m = target_momentum

        for param_o, param_t in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    # @torch.no_grad()
    def update_target_network(self):
        for param_o, param_t in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_t.data = self.m * param_t.data + (1 - self.m) * param_o.data

        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data = self.m * param_t.data + (1 - self.m) * param_o.data

    # @staticmethod
    def regression_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)  #L2-normalize
        y_norm = F.normalize(y, dim=1)  #L2-normalize
        loss = 2 - 2 * (x_norm * y_norm).sum(dim=-1)  #dot product
        return loss.mean()

    def forward(self, xo, xt):
        online_out = self.online_network(xo)
        online_out = self.online_projector(online_out)
        online_out = self.predictor(online_out)

        target_out = self.target_network(xt)
        target_out = self.target_projector(target_out)

        l1 = self.regression_loss(online_out, target_out)

        online_out = self.online_network(xt)
        online_out = self.online_projector(online_out)
        online_out = self.predictor(online_out)

        target_out = self.target_network(xo)
        target_out = self.target_projector(target_out)

        l2 = self.regression_loss(online_out, target_out)

        return l1 + l2 

class EvalByol(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size 
        self.num_classes = num_classes 

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.byol_backbone = ByolBackBone()
        self.classifier = MLPHead(512, hidden_size, num_classes)

        for param in self.byol_backbone.parameters():
            param.requires_grad = False 
        
    def forward(self, x):
        x = self.byol_backbone(x)
        x_pooled = self.avg_pool(x)
        x = x_pooled.view(x_pooled.shape[0], x_pooled.shape[1])
        x = self.classifier(x)
        return x
     
# backbone = models.resnet18(pretrained=True)

# print("resnet-18 backbone")
# print(backbone)

# model = BYOL()
# eval_model = nn.Sequential(*[copy.deepcopy(model.online_network), MLPHead(512, 512, 10)])
# eval_model[0].requires_grad = False

# print(eval_model)
# print(eval_model[0])