"""
PatchGAN Discriminator for Echocardiography Dehazing
Implements multi-scale discriminators for realistic image generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with optional batch normalization"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size, stride, padding, bias=not use_bn)]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for image-to-image translation
    Returns a patch-based output for each input image
    """

    def __init__(self, input_channels=1, ndf=64, n_layers=3, use_sigmoid=False):
        super(PatchGANDiscriminator, self).__init__()

        layers = []

        # First layer (no batch norm)
        layers.append(ConvBlock(input_channels, ndf, use_bn=False))

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(ConvBlock(ndf * nf_mult_prev, ndf * nf_mult))

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, stride=1))

        # Output layer
        final_layer = [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        if use_sigmoid:
            final_layer.append(nn.Sigmoid())

        layers.extend(final_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConditionalPatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN that takes both input (noisy) and output (clean/generated) images
    """

    def __init__(self, input_channels=1, output_channels=1, ndf=64, n_layers=3):
        super(ConditionalPatchGANDiscriminator, self).__init__()

        # Combine input and output channels
        combined_channels = input_channels + output_channels

        layers = []

        # First layer (no batch norm)
        layers.append(ConvBlock(combined_channels, ndf, use_bn=False))

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(ConvBlock(ndf * nf_mult_prev, ndf * nf_mult))

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, stride=1))

        # Output layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1,
                      kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, input_img, target_img):
        # Concatenate input and target images
        combined = torch.cat([input_img, target_img], dim=1)
        return self.model(combined)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator with different patch sizes
    """

    def __init__(self, input_channels=1, output_channels=1, ndf=64, n_layers=3, num_scales=3):
        super(MultiScaleDiscriminator, self).__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for i in range(num_scales):
            # Create discriminator for each scale
            disc = ConditionalPatchGANDiscriminator(
                input_channels=input_channels,
                output_channels=output_channels,
                ndf=ndf,
                n_layers=n_layers
            )
            self.discriminators.append(disc)

        # Downsampling layer for multi-scale
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, input_img, target_img):
        results = []

        input_downsampled = input_img
        target_downsampled = target_img

        for i in range(self.num_scales):
            # Apply discriminator at current scale
            result = self.discriminators[i](
                input_downsampled, target_downsampled)
            results.append(result)

            # Downsample for next scale (except for last iteration)
            if i < self.num_scales - 1:
                input_downsampled = self.downsample(input_downsampled)
                target_downsampled = self.downsample(target_downsampled)

        return results


class SpectralNormConv2d(nn.Module):
    """Convolutional layer with spectral normalization for stable training"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpectralNormConv2d, self).__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)


class SpectralNormDiscriminator(nn.Module):
    """
    Discriminator with spectral normalization for improved training stability
    """

    def __init__(self, input_channels=1, output_channels=1, ndf=64, n_layers=3):
        super(SpectralNormDiscriminator, self).__init__()

        combined_channels = input_channels + output_channels

        layers = []

        # First layer
        layers.append(SpectralNormConv2d(
            combined_channels, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            layers.append(SpectralNormConv2d(ndf * nf_mult_prev,
                          ndf * nf_mult, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        layers.append(SpectralNormConv2d(ndf * nf_mult_prev,
                      ndf * nf_mult, 4, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output layer
        layers.append(SpectralNormConv2d(ndf * nf_mult, 1, 4, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, input_img, target_img):
        combined = torch.cat([input_img, target_img], dim=1)
        return self.model(combined)


class SelfAttentionBlock(nn.Module):
    """Self-attention mechanism for discriminator"""

    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.in_channels = in_channels

        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute query, key, value
        query = self.query(x).view(
            batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        # Compute attention
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Add residual connection
        out = self.gamma * out + x

        return out


class AttentionDiscriminator(nn.Module):
    """
    Discriminator with self-attention for capturing long-range dependencies
    """

    def __init__(self, input_channels=1, output_channels=1, ndf=64, n_layers=3):
        super(AttentionDiscriminator, self).__init__()

        combined_channels = input_channels + output_channels

        # Initial layers
        self.initial = nn.Sequential(
            nn.Conv2d(combined_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Convolutional layers
        layers = []
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            layers.append(nn.Conv2d(ndf * nf_mult_prev,
                          ndf * nf_mult, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            # Add self-attention at middle layer
            if n == n_layers // 2:
                layers.append(SelfAttentionBlock(ndf * nf_mult))

        self.conv_layers = nn.Sequential(*layers)

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        self.final = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)
        )

    def forward(self, input_img, target_img):
        combined = torch.cat([input_img, target_img], dim=1)
        x = self.initial(combined)
        x = self.conv_layers(x)
        return self.final(x)


def test_discriminators():
    """Test function for discriminator architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test inputs
    input_img = torch.randn(2, 1, 256, 256).to(device)
    target_img = torch.randn(2, 1, 256, 256).to(device)

    # Test PatchGAN Discriminator
    patch_disc = PatchGANDiscriminator(input_channels=1).to(device)
    with torch.no_grad():
        output = patch_disc(input_img)
        print(f"PatchGAN Discriminator output shape: {output.shape}")

    # Test Conditional PatchGAN Discriminator
    cond_disc = ConditionalPatchGANDiscriminator(
        input_channels=1, output_channels=1).to(device)
    with torch.no_grad():
        output = cond_disc(input_img, target_img)
        print(
            f"Conditional PatchGAN Discriminator output shape: {output.shape}")

    # Test Multi-Scale Discriminator
    multi_disc = MultiScaleDiscriminator(
        input_channels=1, output_channels=1, num_scales=3).to(device)
    with torch.no_grad():
        outputs = multi_disc(input_img, target_img)
        print(
            f"Multi-Scale Discriminator outputs: {[out.shape for out in outputs]}")

    # Count parameters
    patch_params = sum(p.numel() for p in patch_disc.parameters())
    cond_params = sum(p.numel() for p in cond_disc.parameters())
    multi_params = sum(p.numel() for p in multi_disc.parameters())

    print(f"PatchGAN Discriminator parameters: {patch_params:,}")
    print(f"Conditional PatchGAN Discriminator parameters: {cond_params:,}")
    print(f"Multi-Scale Discriminator parameters: {multi_params:,}")


if __name__ == "__main__":
    test_discriminators()
