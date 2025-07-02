"""
Evaluation metrics for Echocardiography Dehazing
Includes medical-specific metrics like CNR, gCNR, and KS test
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


def calculate_metrics(predicted, target, device='cuda'):
    """
    Wrapper function to calculate metrics between predicted and target images

    Args:
        predicted: Predicted clean images
        target: Target clean images
        device: Device to perform calculations on

    Returns:
        Dictionary of calculated metrics
    """
    calculator = MetricsCalculator(device)
    return calculator.calculate_metrics(predicted, target)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for echocardiography dehazing
    """

    def __init__(self, device='cuda'):
        self.device = device

        # Initialize LPIPS model
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
        except:
            self.lpips_model = None
            print("Warning: LPIPS model not available")

    def denormalize_tensor(self, tensor, mean=0.5, std=0.5):
        """Denormalize tensor from [-1, 1] to [0, 1]"""
        return tensor * std + mean

    def tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    def calculate_psnr(self, generated, target, data_range=1.0):
        """
        Calculate Peak Signal-to-Noise Ratio

        Args:
            generated: Generated image tensor (B, C, H, W)
            target: Target image tensor (B, C, H, W)
            data_range: Data range of images

        Returns:
            PSNR value in dB
        """
        # Convert to numpy
        generated_np = self.tensor_to_numpy(generated)
        target_np = self.tensor_to_numpy(target)

        psnr_values = []
        for i in range(generated_np.shape[0]):
            # Extract single image and convert to 2D if needed
            gen_img = generated_np[i]
            tar_img = target_np[i]

            if len(gen_img.shape) == 3:
                gen_img = gen_img.squeeze()
                tar_img = tar_img.squeeze()

            psnr_val = psnr(tar_img, gen_img, data_range=data_range)
            psnr_values.append(psnr_val)

        return np.mean(psnr_values)

    def calculate_ssim(self, generated, target, data_range=1.0):
        """
        Calculate Structural Similarity Index

        Args:
            generated: Generated image tensor (B, C, H, W)
            target: Target image tensor (B, C, H, W)
            data_range: Data range of images

        Returns:
            SSIM value between 0 and 1
        """
        # Convert to numpy
        generated_np = self.tensor_to_numpy(generated)
        target_np = self.tensor_to_numpy(target)

        ssim_values = []
        for i in range(generated_np.shape[0]):
            # Extract single image
            gen_img = generated_np[i]
            tar_img = target_np[i]

            if len(gen_img.shape) == 3:
                gen_img = gen_img.squeeze()
                tar_img = tar_img.squeeze()

            ssim_val = ssim(tar_img, gen_img, data_range=data_range)
            ssim_values.append(ssim_val)

        return np.mean(ssim_values)

    def calculate_lpips(self, generated, target):
        """
        Calculate Learned Perceptual Image Patch Similarity

        Args:
            generated: Generated image tensor (B, C, H, W)
            target: Target image tensor (B, C, H, W)

        Returns:
            LPIPS distance
        """
        if self.lpips_model is None:
            return float('nan')

        # Convert grayscale to RGB if needed
        if generated.size(1) == 1:
            generated = generated.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        # Ensure tensors are in [-1, 1] range
        generated = torch.clamp(generated, -1, 1)
        target = torch.clamp(target, -1, 1)

        with torch.no_grad():
            lpips_distance = self.lpips_model(generated, target)

        return lpips_distance.mean().item()

    def calculate_cnr(self, image, roi_mask=None, background_mask=None):
        """
        Calculate Contrast-to-Noise Ratio

        Args:
            image: Image tensor (B, C, H, W)
            roi_mask: ROI mask tensor (B, 1, H, W) or None
            background_mask: Background mask tensor (B, 1, H, W) or None

        Returns:
            CNR value
        """
        cnr_values = []

        for i in range(image.size(0)):
            img = image[i].squeeze()

            if roi_mask is not None and roi_mask[i].sum() > 0:
                # Use provided ROI mask
                roi = roi_mask[i].squeeze().bool()
                signal_mean = img[roi].mean().item()
                signal_std = img[roi].std().item()

                # Background is inverse of ROI
                bg = ~roi
                if bg.sum() > 0:
                    noise_std = img[bg].std().item()
                else:
                    noise_std = img.std().item()
            else:
                # Use global statistics
                signal_mean = img.mean().item()
                signal_std = img.std().item()
                noise_std = signal_std

            # CNR = |signal_mean| / noise_std
            cnr = abs(signal_mean) / (noise_std + 1e-8)
            cnr_values.append(cnr)

        return np.mean(cnr_values)

    def calculate_gcnr(self, image, roi_mask=None):
        """
        Calculate Generalized Contrast-to-Noise Ratio

        Args:
            image: Image tensor (B, C, H, W)
            roi_mask: ROI mask tensor (B, 1, H, W) or None

        Returns:
            gCNR value
        """
        gcnr_values = []

        for i in range(image.size(0)):
            img = image[i].squeeze()
            img_np = self.tensor_to_numpy(img)

            if roi_mask is not None and roi_mask[i].sum() > 0:
                roi = roi_mask[i].squeeze().bool()
                roi_np = self.tensor_to_numpy(roi)

                # Extract ROI and background pixels
                roi_pixels = img_np[roi_np]
                bg_pixels = img_np[~roi_np]

                if len(roi_pixels) > 1 and len(bg_pixels) > 1:
                    # Calculate overlap coefficient
                    # gCNR = 1 - overlap_coefficient
                    min_val = min(roi_pixels.min(), bg_pixels.min())
                    max_val = max(roi_pixels.max(), bg_pixels.max())

                    # Create histograms
                    bins = np.linspace(min_val, max_val, 50)
                    hist_roi, _ = np.histogram(
                        roi_pixels, bins=bins, density=True)
                    hist_bg, _ = np.histogram(
                        bg_pixels, bins=bins, density=True)

                    # Calculate overlap
                    overlap = np.sum(np.minimum(hist_roi, hist_bg)) * \
                        (max_val - min_val) / len(bins)
                    gcnr = 1 - overlap
                else:
                    gcnr = 0
            else:
                # Use global contrast measure
                gcnr = img_np.std() / (img_np.mean() + 1e-8)

            gcnr_values.append(gcnr)

        return np.mean(gcnr_values)

    def calculate_ks_test(self, generated, target, roi_mask=None):
        """
        Calculate Kolmogorov-Smirnov test statistic

        Args:
            generated: Generated image tensor (B, C, H, W)
            target: Target image tensor (B, C, H, W)
            roi_mask: ROI mask tensor (B, 1, H, W) or None

        Returns:
            KS test statistic and p-value
        """
        ks_stats = []
        p_values = []

        for i in range(generated.size(0)):
            gen_img = self.tensor_to_numpy(generated[i].squeeze())
            tar_img = self.tensor_to_numpy(target[i].squeeze())

            if roi_mask is not None and roi_mask[i].sum() > 0:
                roi = self.tensor_to_numpy(roi_mask[i].squeeze().bool())
                gen_pixels = gen_img[roi].flatten()
                tar_pixels = tar_img[roi].flatten()
            else:
                gen_pixels = gen_img.flatten()
                tar_pixels = tar_img.flatten()

            # Perform KS test
            ks_stat, p_val = stats.ks_2samp(gen_pixels, tar_pixels)
            ks_stats.append(ks_stat)
            p_values.append(p_val)

        return np.mean(ks_stats), np.mean(p_values)

    def calculate_edge_preservation(self, generated, target):
        """
        Calculate edge preservation metric using Sobel filters

        Args:
            generated: Generated image tensor (B, C, H, W)
            target: Target image tensor (B, C, H, W)

        Returns:
            Edge preservation score
        """
        # Sobel filters
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                               dtype=torch.float32, device=generated.device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                               dtype=torch.float32, device=generated.device)

        # Calculate gradients
        gen_grad_x = F.conv2d(generated, sobel_x, padding=1)
        gen_grad_y = F.conv2d(generated, sobel_y, padding=1)
        tar_grad_x = F.conv2d(target, sobel_x, padding=1)
        tar_grad_y = F.conv2d(target, sobel_y, padding=1)

        # Edge magnitude
        gen_edges = torch.sqrt(gen_grad_x**2 + gen_grad_y**2 + 1e-8)
        tar_edges = torch.sqrt(tar_grad_x**2 + tar_grad_y**2 + 1e-8)

        # Correlation between edge maps
        gen_edges_flat = gen_edges.view(gen_edges.size(0), -1)
        tar_edges_flat = tar_edges.view(tar_edges.size(0), -1)

        correlations = []
        for i in range(gen_edges_flat.size(0)):
            gen_flat = gen_edges_flat[i]
            tar_flat = tar_edges_flat[i]

            # Calculate correlation coefficient
            gen_mean = gen_flat.mean()
            tar_mean = tar_flat.mean()

            numerator = ((gen_flat - gen_mean) * (tar_flat - tar_mean)).sum()
            denominator = torch.sqrt(((gen_flat - gen_mean)**2).sum() *
                                     ((tar_flat - tar_mean)**2).sum())

            if denominator > 1e-8:
                correlation = numerator / denominator
                correlations.append(correlation.item())
            else:
                correlations.append(0.0)

        return np.mean(correlations)

    def calculate_metrics(self, generated, target, roi_mask=None, denormalize=True):
        """
        Calculate all metrics for a batch of images

        Args:
            generated: Generated images (B, C, H, W)
            target: Target images (B, C, H, W)
            roi_mask: ROI masks (B, 1, H, W) or None
            denormalize: Whether to denormalize images from [-1,1] to [0,1]

        Returns:
            Dictionary of metric values
        """
        if denormalize:
            generated = self.denormalize_tensor(generated)
            target = self.denormalize_tensor(target)

        # Clamp values to valid range
        generated = torch.clamp(generated, 0, 1)
        target = torch.clamp(target, 0, 1)

        metrics = {}

        # Basic metrics
        metrics['psnr'] = self.calculate_psnr(generated, target)
        metrics['ssim'] = self.calculate_ssim(generated, target)
        metrics['lpips'] = self.calculate_lpips(
            generated * 2 - 1, target * 2 - 1)  # LPIPS expects [-1,1]

        # Medical metrics
        metrics['cnr'] = self.calculate_cnr(generated, roi_mask)
        metrics['gcnr'] = self.calculate_gcnr(generated, roi_mask)

        # Statistical metrics
        ks_stat, ks_p = self.calculate_ks_test(generated, target, roi_mask)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_p_value'] = ks_p

        # Edge preservation
        metrics['edge_preservation'] = self.calculate_edge_preservation(
            generated, target)

        # L1 and L2 errors
        metrics['l1_error'] = F.l1_loss(generated, target).item()
        metrics['l2_error'] = F.mse_loss(generated, target).item()

        return metrics

    def print_metrics(self, metrics, title="Metrics"):
        """Print metrics in a formatted way"""
        print(f"\n{title}:")
        print("-" * 40)
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric_name:>15}: {value:.4f}")
            else:
                print(f"{metric_name:>15}: {value}")


class MetricsLogger:
    """Logger for tracking metrics during training"""

    def __init__(self):
        self.metrics_history = {}
        self.best_metrics = {}

    def log_metrics(self, metrics, epoch, phase='train'):
        """Log metrics for a specific epoch and phase"""
        key = f"{phase}_epoch_{epoch}"
        self.metrics_history[key] = metrics.copy()

        # Update best metrics
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                best_key = f"best_{metric_name}"
                if best_key not in self.best_metrics:
                    self.best_metrics[best_key] = {
                        'value': value, 'epoch': epoch}
                else:
                    # Higher is better for most metrics except losses and KS statistic
                    is_better = value > self.best_metrics[best_key]['value']
                    if metric_name in ['l1_error', 'l2_error', 'ks_statistic', 'lpips']:
                        is_better = value < self.best_metrics[best_key]['value']

                    if is_better:
                        self.best_metrics[best_key] = {
                            'value': value, 'epoch': epoch}

    def get_best_metric(self, metric_name):
        """Get the best value for a specific metric"""
        key = f"best_{metric_name}"
        return self.best_metrics.get(key, None)

    def save_metrics(self, filepath):
        """Save metrics history to file"""
        import json

        # Convert to JSON-serializable format
        save_data = {
            'history': self.metrics_history,
            'best': self.best_metrics
        }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

    def load_metrics(self, filepath):
        """Load metrics history from file"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.metrics_history = data.get('history', {})
        self.best_metrics = data.get('best', {})


def test_metrics():
    """Test metrics calculation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    batch_size = 2
    height, width = 256, 256

    generated = torch.randn(batch_size, 1, height, width).to(device)
    target = torch.randn(batch_size, 1, height, width).to(device)
    roi_mask = torch.randint(
        0, 2, (batch_size, 1, height, width)).float().to(device)

    # Initialize metrics calculator
    calculator = MetricsCalculator(device)

    # Calculate metrics
    metrics = calculator.calculate_all_metrics(generated, target, roi_mask)

    # Print results
    calculator.print_metrics(metrics, "Test Metrics")

    # Test metrics logger
    logger = MetricsLogger()
    logger.log_metrics(metrics, epoch=1, phase='train')

    print(f"\nBest PSNR: {logger.get_best_metric('psnr')}")


if __name__ == "__main__":
    test_metrics()
