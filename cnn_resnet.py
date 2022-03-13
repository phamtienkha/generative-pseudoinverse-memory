import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

#=======================================================================================================================

class ReShape(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 1, 28, 28)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

#=======================================================================================================================

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                          'bn': nn.BatchNorm2d(out_channels) }))

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

#=======================================================================================================================

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super(EncoderBlock, self).__init__()
        self.encoder_block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=4, stride=2, padding=2),
                                           ResNetBasicBlock(in_channels=out_channels, out_channels=out_channels),
                                           )

    def forward(self, x):
        return self.encoder_block(x)

class Encoder(nn.Module):
    def __init__(self, original_channel, base_channel, input_encoded_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(EncoderBlock(in_channels=original_channel, out_channels=base_channel),
                                     EncoderBlock(in_channels=base_channel, out_channels=base_channel),
                                     EncoderBlock(in_channels=base_channel, out_channels=base_channel),
                                     Flatten(),
                                     nn.Linear(400, input_encoded_size),
                                     )

    def forward(self, x):
        return self.encoder(x)

#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================

class TransposedConv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size

transposed_conv3x3 = partial(TransposedConv2dAuto, kernel_size=3, bias=False)


class TransposedResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=transposed_conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.ConvTranspose2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def transposed_conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                          'bn': nn.BatchNorm2d(out_channels) }))

class TransposedResNetBasicBlock(TransposedResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

#=======================================================================================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DecoderBlock, self).__init__()
        self.decoder_block = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                              kernel_size=kernel_size, stride=2, padding=padding),
                                           TransposedResNetBasicBlock(in_channels=out_channels, out_channels=out_channels)
                                           )

    def forward(self, x):
        return self.decoder_block(x)

class Decoder(nn.Module):
    def __init__(self, original_channel, latent_size, base_channel):
        super().__init__()
        self.decoder = nn.Sequential(
            DecoderBlock(in_channels=latent_size, out_channels=base_channel, kernel_size=4, padding=0),
            DecoderBlock(in_channels=base_channel, out_channels=base_channel, kernel_size=4, padding=1),
            DecoderBlock(in_channels=base_channel, out_channels=base_channel, kernel_size=3, padding=1),
            DecoderBlock(in_channels=base_channel, out_channels=original_channel, kernel_size=4, padding=2),
        )

    def forward(self, x):
        return self.decoder(x.unsqueeze(-1).unsqueeze(-1))

#=======================================================================================================================

if __name__ == '__main__':
    dummy = torch.ones((512, 1, 28, 28))
    layer = Encoder(original_channel=dummy.shape[1], base_channel=16, input_encoded_size=256)
    out = layer(dummy)
    print(out.shape)

    dummy = torch.ones((16, 100))
    layer = Decoder(latent_size=dummy.shape[1], base_channel=16, original_channel=1)
    out = layer(dummy)
    print(out.shape)
