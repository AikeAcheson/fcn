import torchvision
from torch import nn
from Utils.model import bilinear_kernel


def get_FCN(num_classes):
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))  # 1 x 1 convolution
    net.add_module('transpose_conv',
                   nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))
    W = bilinear_kernel(num_classes, num_classes, 64)  # Init transpose_conv
    net.transpose_conv.weight.data.copy_(W)
    return net
