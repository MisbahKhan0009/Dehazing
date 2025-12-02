import io
import os
import base64
from pathlib import Path
from typing import Tuple

from flask import Flask, request, jsonify, send_file, render_template_string
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Config
# ---------------------------
# Absolute path provided by user for best checkpoint
DEFAULT_CKPT = r"C:\\Users\\mkhan\\Documents\\Projects\\CSE498\\Dehazing\\Dehazing\\checkpoints\\gan_dehazing\\best.pt"
IN_CHANNELS = 3
BASE_CHANNELS = 64
PAD_MULTIPLE = 16

# Prefer GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable TF32 on CUDA devices for speed (optional)
if DEVICE.type == 'cuda':
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

# ---------------------------
# Model definition (matches notebook)
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='in'):
        super().__init__()
        Norm = nn.InstanceNorm2d if norm=='in' else nn.BatchNorm2d
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm='in'):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, norm)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, norm='in'):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, norm)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, norm)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64, bilinear=True, norm='in'):
        super().__init__()
        self.inc = DoubleConv(in_channels, base, norm)
        self.down1 = Down(base, base*2, norm)
        self.down2 = Down(base*2, base*4, norm)
        self.down3 = Down(base*4, base*8, norm)
        self.down4 = Down(base*8, base*8, norm)
        self.up1 = Up(base*16, base*4, bilinear, norm)
        self.up2 = Up(base*8, base*2, bilinear, norm)
        self.up3 = Up(base*4, base, bilinear, norm)
        self.up4 = Up(base*2, base, bilinear, norm)
        self.outc = OutConv(base, out_channels)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.tanh(x)

# ---------------------------
# Utilities
# ---------------------------
def _pad_to_multiple(arr: np.ndarray, base: int) -> Tuple[np.ndarray, int, int]:
    h, w = arr.shape[:2]
    nh = (h + base - 1)//base*base
    nw = (w + base - 1)//base*base
    pad_h = nh - h
    pad_w = nw - w
    arr_pad = np.pad(arr, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
    return arr_pad, pad_h, pad_w

@torch.no_grad()
def run_denoise(model: nn.Module, pil_img: Image.Image) -> Image.Image:
    model.eval()
    arr = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
    h, w, _ = arr.shape
    arr_pad, ph, pw = _pad_to_multiple(arr, PAD_MULTIPLE)
    t = torch.from_numpy(arr_pad.transpose(2,0,1)).float().unsqueeze(0)
    t = t * 2.0 - 1.0  # [-1,1]
    t = t.to(DEVICE)
    with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
        out = model(t)
    out = (out.clamp(-1,1) + 1.0) * 0.5
    out = out.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    out = out[:h, :w, :]
    out = (out * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out)

# ---------------------------
# App init and model loading
# ---------------------------
app = Flask(__name__)

GEN = UNetGenerator(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS, base=BASE_CHANNELS).to(DEVICE)
# Channels-last can improve speed on Ampere+ GPUs
if DEVICE.type == 'cuda':
    GEN = GEN.to(memory_format=torch.channels_last)

# If multiple GPUs, DataParallel for inference (optional)
if DEVICE.type == 'cuda' and torch.cuda.device_count() > 1:
    GEN = nn.DataParallel(GEN)

# Load weights
ckpt_path = os.environ.get('GAN_DENOISING_CKPT', DEFAULT_CKPT)
if not os.path.isfile(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
state = torch.load(ckpt_path, map_location=DEVICE)
# Notebook saved under key 'gen'
state_dict = state.get('gen', state)
# Unwrap DP if needed
_target = GEN.module if isinstance(GEN, nn.DataParallel) else GEN
_target.load_state_dict(state_dict)

# ---------------------------
# Routes
# ---------------------------
INDEX_HTML = """
<!doctype html>
<title>GAN Denoiser</title>
<h1>Upload a noisy image</h1>
<form method=post enctype=multipart/form-data action="/infer">
  <input type=file name=image accept="image/*">
  <select name="response" aria-label="Response type">
    <option value="file">Return image file (PNG)</option>
    <option value="json">Return JSON (base64)</option>
  </select>
  <input type=submit value=Upload>
</form>
"""

@app.get('/')
def index():
    return render_template_string(INDEX_HTML)

@app.get('/health')
def health():
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'gpus': torch.cuda.device_count() if DEVICE.type=='cuda' else 0,
        'checkpoint': str(ckpt_path)
    })

@app.post('/infer')
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'no file field named "image"'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'failed to read image: {e}'}), 400

    out_img = run_denoise(GEN, img)

    resp_type = (request.form.get('response') or request.args.get('response') or 'file').lower()
    if resp_type == 'json':
        buf = io.BytesIO()
        out_img.save(buf, format='PNG')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        return jsonify({'image_base64': b64, 'format': 'PNG'})
    else:
        buf = io.BytesIO()
        out_img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png', as_attachment=False, download_name='denoised.png')


if __name__ == '__main__':
    # Example run: python backend/app.py
    port = int(os.environ.get('PORT', '5000'))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
