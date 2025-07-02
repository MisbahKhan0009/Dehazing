"""
Custom Loss Functions for Echocardiography Dehazing GAN
Includes medical-specific losses like CNR and structural similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GANLoss(nn.Module):
    """Standard GAN loss with different options"""

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features"""

    def __init__(self, feature_layers=None, use_input_norm=True):
        super(PerceptualLoss, self).__init__()

        if feature_layers is None:
            feature_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']

        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm

        # Load pre-trained VGG19
        vgg = torch.hub.load('pytorch/vision:v0.10.0',
                             'vgg19', pretrained=True).features
        self.vgg = vgg.eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # VGG normalization
        if use_input_norm:
            self.register_buffer('mean', torch.tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_img, target_img):
        # Convert grayscale to RGB if needed
        if input_img.size(1) == 1:
            input_img = input_img.repeat(1, 3, 1, 1)
        if target_img.size(1) == 1:
            target_img = target_img.repeat(1, 3, 1, 1)

        # Normalize inputs
        if self.use_input_norm:
            input_img = (input_img - self.mean) / self.std
            target_img = (target_img - self.mean) / self.std

        # Extract features
        input_features = self.extract_features(input_img)
        target_features = self.extract_features(target_img)

        # Compute loss
        loss = 0
        for input_feat, target_feat in zip(input_features, target_features):
            loss += F.mse_loss(input_feat, target_feat)

        return loss / len(input_features)

    def extract_features(self, x):
        features = []
        layer_names = []

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            layer_name = f'layer_{i}'

            if isinstance(layer, nn.ReLU):
                # Map layer indices to names
                if i == 1:
                    layer_name = 'relu1_1'
                elif i == 6:
                    layer_name = 'relu2_1'
                elif i == 11:
                    layer_name = 'relu3_1'
                elif i == 20:
                    layer_name = 'relu4_1'
                elif i == 29:
                    layer_name = 'relu5_1'

                if layer_name in self.feature_layers:
                    features.append(x)

        return features


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(
            img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window,
                           padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
            ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim_value = self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average)
        return 1 - ssim_value  # Return loss (1 - SSIM)


class CNRLoss(nn.Module):
    """
    Contrast-to-Noise Ratio Loss for medical images
    Uses ROI annotations to compute CNR in specific regions
    """

    def __init__(self, epsilon=1e-8):
        super(CNRLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, img, roi_mask=None):
        """
        Compute CNR loss
        Args:
            img: Input image (B, C, H, W)
            roi_mask: ROI mask (B, 1, H, W) or None
        """
        if roi_mask is None:
            # If no ROI mask, compute global CNR
            mean_signal = img.mean()
            std_noise = img.std()
        else:
            # Compute CNR within ROI regions
            roi_mask = roi_mask.bool()

            # Signal: mean intensity within ROI
            roi_pixels = img[roi_mask]
            if roi_pixels.numel() > 0:
                mean_signal = roi_pixels.mean()
            else:
                mean_signal = img.mean()

            # Noise: standard deviation outside ROI
            bg_mask = ~roi_mask
            bg_pixels = img[bg_mask]
            if bg_pixels.numel() > 0:
                std_noise = bg_pixels.std()
            else:
                std_noise = img.std()

        # CNR = |signal| / (noise + epsilon)
        cnr = torch.abs(mean_signal) / (std_noise + self.epsilon)

        # Return negative CNR as loss (higher CNR is better)
        return -cnr


class EdgePreservationLoss(nn.Module):
    """Edge preservation loss using Sobel filters"""

    def __init__(self):
        super(EdgePreservationLoss, self).__init__()

        # Sobel filters
        self.register_buffer('sobel_x', torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32))
        self.register_buffer('sobel_y', torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32))

    def forward(self, input_img, target_img):
        # Compute gradients
        input_grad_x = F.conv2d(input_img, self.sobel_x, padding=1)
        input_grad_y = F.conv2d(input_img, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target_img, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_img, self.sobel_y, padding=1)

        # Compute edge magnitude
        input_edges = torch.sqrt(input_grad_x**2 + input_grad_y**2 + 1e-8)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)

        # L1 loss on edge maps
        return F.l1_loss(input_edges, target_edges)


class TotalVariationLoss(nn.Module):
    """Total Variation Loss for smoothness regularization"""

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        batch_size, channels, height, width = img.size()

        # Compute differences
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()

        # Normalize by image size
        return (tv_h + tv_w) / (batch_size * channels * height * width)


class CombinedLoss(nn.Module):
    """
    Combined loss function for echocardiography dehazing
    Combines multiple loss functions with adjustable weights
    """

    def __init__(self,
                 lambda_l1=100.0,
                 lambda_perceptual=1.0,
                 lambda_ssim=1.0,
                 lambda_cnr=0.1,
                 lambda_edge=10.0,
                 lambda_tv=0.01,
                 use_roi=True):
        super(CombinedLoss, self).__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_cnr = lambda_cnr
        self.lambda_edge = lambda_edge
        self.lambda_tv = lambda_tv
        self.use_roi = use_roi

        # Initialize loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = SSIMLoss()
        self.cnr_loss = CNRLoss()
        self.edge_loss = EdgePreservationLoss()
        self.tv_loss = TotalVariationLoss()

    def forward(self, generated, target, roi_mask=None):
        total_loss = 0
        loss_dict = {}

        # L1 reconstruction loss
        l1_loss_val = self.l1_loss(generated, target)
        total_loss += self.lambda_l1 * l1_loss_val
        loss_dict['l1'] = l1_loss_val.item()

        # Perceptual loss
        if self.lambda_perceptual > 0:
            perceptual_loss_val = self.perceptual_loss(generated, target)
            total_loss += self.lambda_perceptual * perceptual_loss_val
            loss_dict['perceptual'] = perceptual_loss_val.item()

        # SSIM loss
        if self.lambda_ssim > 0:
            ssim_loss_val = self.ssim_loss(generated, target)
            total_loss += self.lambda_ssim * ssim_loss_val
            loss_dict['ssim'] = ssim_loss_val.item()

        # CNR loss
        if self.lambda_cnr > 0 and self.use_roi:
            cnr_loss_val = self.cnr_loss(generated, roi_mask)
            total_loss += self.lambda_cnr * cnr_loss_val
            loss_dict['cnr'] = cnr_loss_val.item()

        # Edge preservation loss
        if self.lambda_edge > 0:
            edge_loss_val = self.edge_loss(generated, target)
            total_loss += self.lambda_edge * edge_loss_val
            loss_dict['edge'] = edge_loss_val.item()

        # Total variation loss
        if self.lambda_tv > 0:
            tv_loss_val = self.tv_loss(generated)
            total_loss += self.lambda_tv * tv_loss_val
            loss_dict['tv'] = tv_loss_val.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class GradientPenaltyLoss(nn.Module):
    """Gradient Penalty for WGAN-GP"""

    def __init__(self, lambda_gp=10.0):
        super(GradientPenaltyLoss, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, discriminator, real_images, fake_images, input_images):
        batch_size = real_images.size(0)
        device = real_images.device

        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        interpolates = alpha * real_images + (1 - alpha) * fake_images
        interpolates.requires_grad_(True)

        # Discriminator output for interpolated images
        disc_interpolates = discriminator(input_images, interpolates)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return self.lambda_gp * gradient_penalty


def test_losses():
    """Test function for loss functions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test inputs
    generated = torch.randn(2, 1, 256, 256).to(device)
    target = torch.randn(2, 1, 256, 256).to(device)
    roi_mask = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

    # Test individual losses
    print("Testing individual loss functions...")

    # GAN Loss
    gan_loss = GANLoss('lsgan').to(device)
    disc_output = torch.randn(2, 1, 30, 30).to(device)
    gan_loss_val = gan_loss(disc_output, True)
    print(f"GAN Loss: {gan_loss_val.item():.4f}")

    # SSIM Loss
    ssim_loss = SSIMLoss().to(device)
    ssim_loss_val = ssim_loss(generated, target)
    print(f"SSIM Loss: {ssim_loss_val.item():.4f}")

    # CNR Loss
    cnr_loss = CNRLoss().to(device)
    cnr_loss_val = cnr_loss(generated, roi_mask)
    print(f"CNR Loss: {cnr_loss_val.item():.4f}")

    # Edge Loss
    edge_loss = EdgePreservationLoss().to(device)
    edge_loss_val = edge_loss(generated, target)
    print(f"Edge Loss: {edge_loss_val.item():.4f}")

    # Combined Loss
    combined_loss = CombinedLoss().to(device)
    total_loss, loss_dict = combined_loss(generated, target, roi_mask)
    print(f"Combined Loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")


if __name__ == "__main__":
    test_losses()
