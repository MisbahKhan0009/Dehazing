"""
U-Net Generator for Echocardiography Dehazing
Implements a U-Net architecture with skip connections for image-to-image translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Attention mechanism for focusing on important features"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetGenerator(nn.Module):
    """
    U-Net Generator with attention mechanism for echocardiography dehazing
    """

    def __init__(self, n_channels=1, n_classes=1, bilinear=False, use_attention=True):
        super(UNetGenerator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Attention blocks
        if use_attention:
            self.att1 = AttentionBlock(F_g=1024//factor, F_l=512, F_int=256)
            self.att2 = AttentionBlock(F_g=512, F_l=256, F_int=128)
            self.att3 = AttentionBlock(F_g=256, F_l=128, F_int=64)
            self.att4 = AttentionBlock(F_g=128, F_l=64, F_int=32)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # Output activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with attention
        if self.use_attention:
            x4_att = self.att1(g=x5, x=x4)
            x = self.up1(x5, x4_att)

            x3_att = self.att2(g=x, x=x3)
            x = self.up2(x, x3_att)

            x2_att = self.att3(g=x, x=x2)
            x = self.up3(x, x2_att)

            x1_att = self.att4(g=x, x=x1)
            x = self.up4(x, x1_att)
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        logits = self.outc(x)
        return self.tanh(logits)


class ResidualBlock(nn.Module):
    """Residual block for generator refinement"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class EnhancedUNetGenerator(nn.Module):
    """
    Enhanced U-Net with residual connections and multi-scale features
    """

    def __init__(self, n_channels=1, n_classes=1, num_residual_blocks=6):
        super(EnhancedUNetGenerator, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # Store input for residual connection
        input_x = x

        # Forward pass
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residual_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)

        # Add residual connection from input
        return x + input_x


def test_generators():
    """Test function for generator architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test UNetGenerator
    unet = UNetGenerator(n_channels=1, n_classes=1,
                         use_attention=True).to(device)
    test_input = torch.randn(1, 1, 256, 256).to(device)

    with torch.no_grad():
        output = unet(test_input)
        print(f"UNet Generator output shape: {output.shape}")

    # Test EnhancedUNetGenerator
    enhanced_unet = EnhancedUNetGenerator(n_channels=1, n_classes=1).to(device)
    with torch.no_grad():
        output = enhanced_unet(test_input)
        print(f"Enhanced UNet Generator output shape: {output.shape}")

    # Count parameters
    unet_params = sum(p.numel() for p in unet.parameters())
    enhanced_params = sum(p.numel() for p in enhanced_unet.parameters())

    print(f"UNet Generator parameters: {unet_params:,}")
    print(f"Enhanced UNet Generator parameters: {enhanced_params:,}")


if __name__ == "__main__":
    test_generators()
