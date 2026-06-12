"""
Flux2-style latent diffusion on Flickr8k
=========================================
Improvements over v1:
  1. Latent-space training via frozen SD VAE (256×256 → 32×32×4)
  2. Rectified flow / flow-matching loss instead of DDPM
  3. Frozen CLIP ViT-B/32 replaces custom text encoder
  4. Two backbone options: UNet (stable) or DiT (scalable) via BACKBONE config
  5. Effective 256×256 resolution training
  6. One-time VAE latent cache — VAE runs only once, not every epoch

For SDXL/FLUX LoRA fine-tuning instead of training from scratch:
  → use diffusers train_dreambooth_lora_sdxl.py or train_text_to_image_lora.py
  → requires only ~500 images and converges in < 1 hour on 1 GPU
"""

import os, math, copy, glob, random, warnings, json, time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import CLIPTokenizer, CLIPModel
from diffusers import AutoencoderKL

import sys as _sys
import matplotlib
# Set backend before any other matplotlib import.
# In --no_gui mode use non-interactive Agg so no display is needed.
matplotlib.use("Agg" if "--no_gui" in _sys.argv else "TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ─────────────────────────────────────────────────────────────────────

IMG_DIR      = "./flickr8k/Images"
CAPTIONS_CSV = "./flickr8k/captions.txt"
LATENT_CACHE = "latents_cache.pt"    # pre-computed VAE latents (created once)

# swap "openai/clip-vit-large-patch14" for richer 768-dim embeddings
CLIP_MODEL   = "openai/clip-vit-base-patch32"
CLIP_DIM     = 512
CLIP_SEQ_LEN = 77

VAE_MODEL    = "stabilityai/sd-vae-ft-mse"
VAE_SCALE    = 0.18215               # SD latent scaling constant
IMG_SIZE     = 256                   # real image size fed into VAE
LATENT_SIZE  = 32                    # VAE: 256 / 8 = 32
LATENT_CH    = 4                     # VAE latent channels

# ── choose backbone ──────────────────────────────────────────────────────────
BACKBONE     = "unet"   # "unet" | "dit"

# UNet settings
UNET_CH      = [128, 256, 512]

# DiT settings
DIT_DIM      = 512
DIT_DEPTH    = 8
DIT_HEADS    = 8
DIT_PATCH    = 2                     # 32/2 = 16 → 256 patches

# ── training ─────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
LR           = 1e-4
EPOCHS       = 200
FLOW_STEPS   = 50                    # Euler ODE steps at inference
CFG_DROPOUT  = 0.15
CFG_SCALE    = 3.0                   # bump to 7.5 after ~50 epochs
EMA_DECAY    = 0.9999
VIZ_EVERY    = 50
SAVE_EVERY   = 10

CLIP_LOSS_WEIGHT = 0.05   # auxiliary CLIP semantic loss weight
CLIP_LOSS_EVERY  = 20    # compute every N steps (VAE decode + CLIP forward is expensive)
CLIP_LOSS_BATCH  = 4     # samples used for CLIP loss per compute

LOG_FILE         = "training_log.jsonl"   # one JSON object per line
LOG_EVERY        = 10                     # log every N steps

EMBED_DIM    = 512                   # time embedding dim
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── Frozen CLIP text encoder ───────────────────────────────────────────────────

class FrozenCLIP(nn.Module):
    """CLIP ViT-B/32 — text encoder for cross-attention + image encoder for auxiliary loss."""
    def __init__(self, model_id=CLIP_MODEL):
        super().__init__()
        self.tok   = CLIPTokenizer.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, device):
        """Returns last-hidden-state sequence (B, 77, 512) for cross-attention."""
        tok = self.tok(texts, padding="max_length", max_length=CLIP_SEQ_LEN,
                       truncation=True, return_tensors="pt")
        out = self.model.text_model(input_ids=tok.input_ids.to(device),
                                    attention_mask=tok.attention_mask.to(device))
        return out.last_hidden_state  # (B, 77, 512)

    @torch.no_grad()
    def encode_text_pooled(self, texts, device):
        """Returns L2-normalised pooled text embedding (B, 512) for CLIP loss."""
        tok  = self.tok(texts, padding="max_length", max_length=CLIP_SEQ_LEN,
                        truncation=True, return_tensors="pt")
        out  = self.model.text_model(input_ids=tok.input_ids.to(device),
                                     attention_mask=tok.attention_mask.to(device))
        feats = self.model.text_projection(out.pooler_output)
        return F.normalize(feats, dim=-1)

    def encode_images_for_loss(self, imgs):
        """
        Returns L2-normalised pooled image embedding (B, 512).
        imgs: (B,3,H,W) in [-1,1].  Gradients flow through so the UNet learns.
        """
        x = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
        x = x * 0.5 + 0.5  # [-1,1] → [0,1]
        mean = torch.tensor([0.48145466, 0.4578275,  0.40821073],
                             device=imgs.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=imgs.device).view(1, 3, 1, 1)
        out   = self.model.vision_model(pixel_values=(x - mean) / std)
        feats = self.model.visual_projection(out.pooler_output)
        return F.normalize(feats, dim=-1)

    def null_embed(self, B, device):
        return self.encode([""] * B, device)

    def forward(self, texts, device):
        return self.encode(texts, device)


# ── Frozen VAE ────────────────────────────────────────────────────────────────

class FrozenVAE(nn.Module):
    def __init__(self, model_id=VAE_MODEL):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x):
        """x: (B,3,H,W) ∈ [-1,1]  →  latents (B,4,H/8,W/8)"""
        return self.vae.encode(x).latent_dist.sample() * VAE_SCALE

    @torch.no_grad()
    def decode(self, z):
        """z: (B,4,h,w)  →  images (B,3,H,W) ∈ [-1,1]"""
        return self.vae.decode(z / VAE_SCALE).sample

    def decode_grad(self, z):
        """Same as decode() but keeps the compute graph so gradients flow back through z."""
        return self.vae.decode(z / VAE_SCALE).sample


# ── Latent cache ──────────────────────────────────────────────────────────────

def precompute_latents(vae, csv_path, img_dir, save_path, batch_size=32):
    """Encode all unique images with the frozen VAE once and save to disk."""
    if os.path.isfile(save_path):
        print(f"Latent cache found: {save_path}")
        return
    print("Pre-computing VAE latents (one-time — may take a few minutes)…")
    vae.eval().to(DEVICE)
    df = pd.read_csv(csv_path); df.columns = df.columns.str.strip()
    unique = [f for f in df["image"].unique()
              if os.path.isfile(os.path.join(img_dir, f))]
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    cache, imgs_b, names_b = {}, [], []
    for i, fname in enumerate(unique):
        imgs_b.append(tfm(Image.open(os.path.join(img_dir, fname)).convert("RGB")))
        names_b.append(fname)
        if len(imgs_b) == batch_size or i == len(unique) - 1:
            lats = vae.encode(torch.stack(imgs_b).to(DEVICE)).cpu()
            for n, l in zip(names_b, lats):
                cache[n] = l
            imgs_b, names_b = [], []
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(unique)}")
    torch.save(cache, save_path)
    print(f"Saved {len(cache)} latents → {save_path}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        df = pd.read_csv(csv_path); df.columns = df.columns.str.strip()
        df["exists"] = df["image"].apply(
            lambda f: os.path.isfile(os.path.join(img_dir, f)))
        self.df      = df[df["exists"]].reset_index(drop=True)
        self.img_dir = img_dir
        self.cache   = {}   # filled by load_latent_cache()
        self.tfm = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def load_latent_cache(self, path):
        if os.path.isfile(path):
            self.cache = torch.load(path, map_location="cpu")
            print(f"Loaded {len(self.cache)} cached latents")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fname = row["image"]
        if fname in self.cache:
            lat = self.cache[fname]         # (4,32,32) — already encoded
        else:
            img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
            lat = self.tfm(img)             # (3,256,256) — will be encoded on GPU
        return lat, row["caption"]


# ── Rectified flow scheduler ──────────────────────────────────────────────────

class RectifiedFlow:
    """
    Forward:   x_t = (1−t)·x0 + t·ε,  ε ~ N(0,I),  t ∈ [0,1]
    Target:    v   = ε − x0
    Sampling:  Euler ODE from t=1 (noise) → t=0 (image)
               x_{t−dt} = x_t − dt · v(x_t, t)
    """

    def forward_process(self, x0, t):
        noise  = torch.randn_like(x0)
        tv     = t.view(-1, 1, 1, 1)
        x_t    = (1 - tv) * x0 + tv * noise
        return x_t, noise - x0             # noisy sample, velocity target

    @torch.no_grad()
    def sample(self, model, shape, ctx, steps=FLOW_STEPS, cfg_scale=1.0):
        use_cfg    = cfg_scale > 1.0
        null_ctx   = torch.zeros_like(ctx) if use_cfg else None
        x          = torch.randn(shape, device=DEVICE)
        dt         = 1.0 / steps

        for i in range(steps):
            t_val = 1.0 - i * dt
            t     = torch.full((shape[0],), t_val, device=DEVICE)
            v     = model(x, t, ctx)
            if use_cfg:
                vu = model(x, t, null_ctx)
                v  = (vu + cfg_scale * (v - vu)).clamp(-5, 5)
            else:
                v  = v.clamp(-5, 5)
            x = (x - dt * v).clamp(-4, 4)  # stability guard

        return x.clamp(-1, 1)


# ── Shared building blocks ────────────────────────────────────────────────────

def sinusoidal_embedding(t, dim):
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args  = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
        self.dim  = dim

    def forward(self, t):
        t_int = (t * 999).clamp(0, 999).long() if (t.is_floating_point() and t.max() <= 1.0) else t.long()
        return self.proj(sinusoidal_embedding(t_int, self.dim))


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, nhead=8):
        super().__init__()
        assert query_dim % nhead == 0
        self.nhead = nhead
        self.scale = (query_dim // nhead) ** -0.5
        self.q   = nn.Linear(query_dim,   query_dim, bias=False)
        self.k   = nn.Linear(context_dim, query_dim, bias=False)
        self.v   = nn.Linear(context_dim, query_dim, bias=False)
        self.out = nn.Linear(query_dim,   query_dim)

    def forward(self, x, ctx):
        B, N, C = x.shape; h = self.nhead
        q = self.q(x).view(B, N,  h, C//h).transpose(1, 2)
        k = self.k(ctx).view(B, -1, h, C//h).transpose(1, 2)
        v = self.v(ctx).view(B, -1, h, C//h).transpose(1, 2)
        out = (F.scaled_dot_product_attention(q, k, v)
               .transpose(1, 2).contiguous().view(B, N, C))
        return self.out(out)


# ── UNet backbone ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch, time_dim, ctx_dim, nhead=8):
        super().__init__()
        g = min(32, ch // 4)
        self.norm1     = nn.GroupNorm(g, ch)
        self.conv1     = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2     = nn.GroupNorm(g, ch)
        self.conv2     = nn.Conv2d(ch, ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, ch * 2)   # scale + shift (adaGN)
        self.xattn     = CrossAttention(ch, ctx_dim, nhead)
        self.xnorm     = nn.LayerNorm(ch)

    def forward(self, x, t_emb, ctx):
        B, C, H, W = x.shape
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb)).view(B, C*2, 1, 1).chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        x = x + h
        flat = x.view(B, C, H*W).transpose(1, 2)
        flat = flat + self.xattn(self.xnorm(flat), ctx)
        return flat.transpose(1, 2).view(B, C, H, W)


class DownBlock(nn.Module):
    def __init__(self, ic, oc, td, cd):
        super().__init__()
        self.conv_in = nn.Conv2d(ic, oc, 3, padding=1)
        self.res     = ResBlock(oc, td, cd)
        self.down    = nn.Conv2d(oc, oc, 4, stride=2, padding=1)

    def forward(self, x, t, c):
        x = F.silu(self.conv_in(x)); x = self.res(x, t, c)
        return self.down(x), x          # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, ic, sc, oc, td, cd):
        super().__init__()
        self.up      = nn.ConvTranspose2d(ic, ic, 4, stride=2, padding=1)
        self.conv_in = nn.Conv2d(ic + sc, oc, 3, padding=1)
        self.res     = ResBlock(oc, td, cd)

    def forward(self, x, skip, t, c):
        x = F.silu(self.conv_in(torch.cat([self.up(x), skip], 1)))
        return self.res(x, t, c)


class UNet(nn.Module):
    """
    Latent UNet: 4-ch input, CLIP context (77×512), velocity-field output.
    Channels 4 → 128 → 256 → 512 → mid → 256 → 128 → 4
    """
    def __init__(self, in_ch=LATENT_CH, ctx_dim=CLIP_DIM,
                 td=EMBED_DIM, ch=UNET_CH):
        super().__init__()
        self.time_embed = TimeEmbed(td)
        self.ctx_proj   = nn.Linear(CLIP_DIM, ctx_dim)

        self.d1 = DownBlock(in_ch,  ch[0], td, ctx_dim)
        self.d2 = DownBlock(ch[0],  ch[1], td, ctx_dim)
        self.d3 = DownBlock(ch[1],  ch[2], td, ctx_dim)
        self.mid = ResBlock(ch[2], td, ctx_dim)
        self.u3 = UpBlock(ch[2], ch[2], ch[1], td, ctx_dim)
        self.u2 = UpBlock(ch[1], ch[1], ch[0], td, ctx_dim)
        self.u1 = UpBlock(ch[0], ch[0], ch[0], td, ctx_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(min(32, ch[0]//4), ch[0]), nn.SiLU(),
            nn.Conv2d(ch[0], in_ch, 3, padding=1),
        )

    def forward(self, x, t, ctx):
        te  = self.time_embed(t)
        ctx = self.ctx_proj(ctx)
        x, s1 = self.d1(x, te, ctx)
        x, s2 = self.d2(x, te, ctx)
        x, s3 = self.d3(x, te, ctx)
        x      = self.mid(x, te, ctx)
        x      = self.u3(x, s3, te, ctx)
        x      = self.u2(x, s2, te, ctx)
        x      = self.u1(x, s1, te, ctx)
        return self.out(x)


# ── DiT backbone ──────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """Self-attn + cross-attn + FFN, conditioned via adaLN from time embedding."""
    def __init__(self, dim, nhead, ctx_dim, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.sattn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.xattn = CrossAttention(dim, ctx_dim, nhead)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        # adaLN: time_emb → 6 × dim  (shift+scale for each of 3 sub-layers)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, t_emb, ctx):
        s1, g1, s2, g2, s3, g3 = self.ada(t_emb).chunk(6, dim=-1)
        # each si/gi: (B, dim)
        def mod(h, shift, scale):
            return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        h = mod(self.norm1(x), s1, g1)
        x = x + self.sattn(h, h, h, need_weights=False)[0]
        h = mod(self.norm2(x), s2, g2)
        x = x + self.xattn(h, ctx)
        h = mod(self.norm3(x), s3, g3)
        x = x + self.ffn(h)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer: patchify latents → N transformer blocks → unpatchify.
    Default: patch=2, dim=512, depth=8 → ~45M params on 32×32×4 latents.
    """
    def __init__(self, lat_size=LATENT_SIZE, lat_ch=LATENT_CH,
                 patch=DIT_PATCH, dim=DIT_DIM, depth=DIT_DEPTH,
                 nhead=DIT_HEADS, ctx_dim=CLIP_DIM):
        super().__init__()
        self.patch    = patch
        self.lat_size = lat_size
        self.lat_ch   = lat_ch
        n_patches     = (lat_size // patch) ** 2
        patch_dim     = lat_ch * patch * patch

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)
        self.time_embed  = TimeEmbed(dim)
        self.ctx_proj    = nn.Linear(CLIP_DIM, dim)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, nhead, dim) for _ in range(depth)])
        self.norm_out  = nn.LayerNorm(dim)
        self.proj_out  = nn.Linear(dim, patch_dim)

    def patchify(self, x):
        B, C, H, W = x.shape; p = self.patch
        return (x.reshape(B, C, H//p, p, W//p, p)
                  .permute(0, 2, 4, 1, 3, 5)
                  .reshape(B, (H//p)*(W//p), C*p*p))

    def unpatchify(self, x):
        B, N, _ = x.shape; p = self.patch; h = self.lat_size // p; C = self.lat_ch
        return (x.reshape(B, h, h, C, p, p)
                  .permute(0, 3, 1, 4, 2, 5)
                  .reshape(B, C, self.lat_size, self.lat_size))

    def forward(self, x, t, ctx):
        te  = self.time_embed(t)
        ctx = self.ctx_proj(ctx)
        x   = self.patch_embed(self.patchify(x)) + self.pos_embed
        for blk in self.blocks:
            x = blk(x, te, ctx)
        return self.unpatchify(self.proj_out(self.norm_out(x)))


def build_model():
    if BACKBONE == "dit":
        m = DiT()
    else:
        m = UNet()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Backbone: {BACKBONE.upper()} | params: {n/1e6:.1f}M")
    return m


# ── Live visualisation ─────────────────────────────────────────────────────────

class LiveViz:
    N_SAMPLES = 4

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 5), facecolor="#111")
        self.fig.canvas.manager.set_window_title("Flux2-Flickr Training")
        gs = gridspec.GridSpec(1, self.N_SAMPLES + 1,
                               width_ratios=[2.5] + [1]*self.N_SAMPLES,
                               wspace=0.25)

        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_loss.set_facecolor("#1a1a2e")
        self.ax_loss.set_title("Training Loss (rectified flow)", color="white", fontsize=9)
        self.ax_loss.tick_params(colors="white")
        for sp in self.ax_loss.spines.values():
            sp.set_color("#444")
        self.loss_line, = self.ax_loss.plot([], [], color="#00d4ff", lw=0.8, alpha=0.4, label="raw")
        self.ema_line,  = self.ax_loss.plot([], [], color="#ff6b35", lw=2.0,           label="EMA")
        self.ax_loss.legend(facecolor="#222", labelcolor="white", fontsize=7, loc="upper right")
        self.ax_loss.set_xlabel("Step",     color="white", fontsize=8)
        self.ax_loss.set_ylabel("MSE Loss", color="white", fontsize=8)

        self.sample_axes = []
        for i in range(self.N_SAMPLES):
            ax = self.fig.add_subplot(gs[0, i + 1])
            ax.set_facecolor("#1a1a1a"); ax.set_axis_off()
            ax.text(0.5, 0.5, "generating…", ha="center", va="center",
                    color="#555", fontsize=8, transform=ax.transAxes)
            self.sample_axes.append(ax)

        self.steps = []; self.losses = []; self.ema_vals = []
        self._ema_alpha = 0.02
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.1)
        self.fig.canvas.draw(); plt.pause(0.01)

    def update_loss(self, step, loss):
        self.steps.append(step); self.losses.append(loss)
        ema = ((1 - self._ema_alpha)*self.ema_vals[-1] + self._ema_alpha*loss
               if self.ema_vals else loss)
        self.ema_vals.append(ema)
        self.loss_line.set_data(self.steps, self.losses)
        self.ema_line.set_data(self.steps, self.ema_vals)
        self.ax_loss.relim(); self.ax_loss.autoscale_view()

    def update_samples(self, images, captions):
        for i in range(self.N_SAMPLES):
            ax = self.sample_axes[i]
            ax.cla(); ax.set_axis_off(); ax.set_facecolor("#1a1a1a")
            if i < len(images):
                img = images[i].cpu().permute(1, 2, 0).float().numpy()
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
                img = (img * 0.5 + 0.5).clip(0.0, 1.0)
                ax.imshow(img, aspect="auto", interpolation="bilinear")
                ax.set_title(captions[i] if i < len(captions) else "",
                             color="white", fontsize=7, pad=4,
                             bbox=dict(boxstyle="round,pad=0.25", fc="#000000bb", ec="none"))
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def set_info(self, epoch, total_epochs, lr):
        self.fig.suptitle(
            f"Epoch {epoch}/{total_epochs}  |  LR {lr:.2e}  |  "
            f"CFG ×{CFG_SCALE}  |  {BACKBONE.upper()}  |  {IMG_SIZE}×{IMG_SIZE}→latent {LATENT_SIZE}×{LATENT_SIZE}",
            color="white", fontsize=8)

    def flush(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff(); plt.show()


# ── CLI progress (--no_gui mode) ───────────────────────────────────────────────

class CLIProgress:
    """
    CLI training display for --no_gui mode.
    Uses a single overwriting status line (\r) — no Live panels, no stacking.
    Sample generation is skipped entirely (controlled in the training loop).
    """
    _WIDTH = 110   # chars to clear on each \r

    def __init__(self):
        from rich.console import Console
        self.console    = Console(highlight=False)
        self._s         = dict(epoch=0, total=EPOCHS, step=0,
                               loss=0.0, ema=0.0, lr=LR, best=float("inf"))
        self.ema_vals   = []
        self._ema_alpha = 0.02
        self.console.print(
            f"[bold magenta]Flux2-Flickr[/]  "
            f"[dim]{BACKBONE.upper()} · CLIP · VAE · Rectified Flow · "
            f"{DEVICE.upper()} · CFG ×{CFG_SCALE}[/]\n"
            "─" * self._WIDTH
        )

    def update_loss(self, step, loss):
        self._s["step"] = step
        self._s["loss"] = loss
        ema = ((1 - self._ema_alpha) * self.ema_vals[-1] + self._ema_alpha * loss
               if self.ema_vals else loss)
        self.ema_vals.append(ema)
        self._s["ema"] = ema
        if loss < self._s["best"]:
            self._s["best"] = loss

    def update_samples(self, images, captions):
        pass   # sample generation is disabled in --no_gui mode

    def set_info(self, epoch, total_epochs, lr):
        self._s["epoch"] = epoch
        self._s["total"] = total_epochs
        self._s["lr"]    = lr

    def flush(self):
        s = self._s
        line = (f"\r  epoch {s['epoch']:>3}/{s['total']}"
                f"  step {s['step']:>8,}"
                f"  loss {s['loss']:.5f}"
                f"  ema {s['ema']:.5f}"
                f"  lr {s['lr']:.2e}"
                f"  best {s['best']:.5f}   ")
        _sys.stdout.write(line.ljust(self._WIDTH))
        _sys.stdout.flush()

    def log(self, msg):
        """Print a permanent line (epoch end, checkpoint) without breaking the status line."""
        _sys.stdout.write("\r" + " " * self._WIDTH + "\r")
        _sys.stdout.flush()
        self.console.print(msg)

    def close(self):
        _sys.stdout.write("\n")
        self.console.print("─" * self._WIDTH)
        self.console.print("[bold green]✓  Training complete.[/]")


# ── Training ──────────────────────────────────────────────────────────────────

def train(no_gui=False):
    print(f"Device: {DEVICE}  |  Backbone: {BACKBONE}")

    # ── frozen pretrained models ──────────────────────────────────────────────
    print("Loading CLIP…")
    clip = FrozenCLIP().to(DEVICE)
    print("Loading VAE…")
    vae  = FrozenVAE().to(DEVICE)

    # ── one-time latent pre-computation ───────────────────────────────────────
    precompute_latents(vae, CAPTIONS_CSV, IMG_DIR, LATENT_CACHE)

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = Flickr8kDataset(CAPTIONS_CSV, IMG_DIR)
    dataset.load_latent_cache(LATENT_CACHE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=False, drop_last=True)
    print(f"Dataset: {len(dataset)} pairs")

    # ── trainable model ───────────────────────────────────────────────────────
    model     = build_model().to(DEVICE)
    ema_model = copy.deepcopy(model); ema_model.eval()

    flow      = RectifiedFlow()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS * len(loader))

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch, step, best_loss = 1, 0, float("inf")
    ckpts = sorted(glob.glob("checkpoint_epoch*.pt"))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location=DEVICE)
        try:
            model.load_state_dict(ck["model"])
            ema_model.load_state_dict(ck.get("ema_model", ck["model"]))
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = ck["epoch"] + 1
            step        = ck.get("step", 0)
            best_loss   = ck.get("loss", float("inf"))
            for _ in range(step): lr_sched.step()
            print(f"Resumed {ckpts[-1]} (epoch {ck['epoch']}, step {step})")
        except Exception as e:
            print(f"Could not resume: {e} — starting fresh")

    # ── viz ───────────────────────────────────────────────────────────────────
    viz = CLIProgress() if no_gui else LiveViz()
    fixed_prompts = [
        "a dog running on the beach",
        "a child playing in a park",
        "two people hiking on a mountain trail",
        "a cat sitting on a wooden fence",
    ]

    # ── train loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for lats, captions in loader:
            # encode if raw images (no cache hit)
            if lats.shape[1] == 3:
                lats = vae.encode(lats.to(DEVICE))
            else:
                lats = lats.to(DEVICE, dtype=torch.float32)

            # CLIP text encoding with CFG dropout
            drop  = torch.rand(len(captions)) < CFG_DROPOUT
            texts = ["" if d else c for d, c in zip(drop.tolist(), captions)]
            with torch.no_grad():
                ctx = clip.encode(texts, DEVICE)   # (B, 77, 512)

            # rectified flow loss
            t            = torch.rand(lats.shape[0], device=DEVICE)
            x_t, v_tgt   = flow.forward_process(lats, t)
            v_pred        = model(x_t, t, ctx)
            loss          = F.mse_loss(v_pred, v_tgt)
            flow_loss_val = loss.item()
            clip_loss_val = None

            # CLIP auxiliary loss — "does the predicted image match the text?"
            # Gradients flow: CLIP image encoder → VAE decoder → x0_pred → v_pred → UNet
            if step % CLIP_LOSS_EVERY == 0 and CLIP_LOSS_WEIGHT > 0:
                n    = min(CLIP_LOSS_BATCH, lats.shape[0])
                tv   = t[:n].view(-1, 1, 1, 1)
                # rectified flow x0 prediction: x0 = x_t - t * v
                x0_pred  = (x_t[:n] - tv * v_pred[:n]).clamp(-4, 4)
                imgs_pred = vae.decode_grad(x0_pred)          # (n,3,H,W) in [-1,1]
                img_feats = clip.encode_images_for_loss(imgs_pred)  # gradients flow here
                with torch.no_grad():
                    # use original captions (not CFG-dropped texts)
                    txt_feats = clip.encode_text_pooled(list(captions[:n]), DEVICE)
                clip_loss     = 1.0 - (img_feats * txt_feats).sum(dim=-1).mean()
                clip_loss_val = clip_loss.item()
                loss = loss + CLIP_LOSS_WEIGHT * clip_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_sched.step()

            with torch.no_grad():
                for pe, p in zip(ema_model.parameters(), model.parameters()):
                    pe.data.lerp_(p.data, 1.0 - EMA_DECAY)

            epoch_loss += loss.item()
            step += 1
            viz.update_loss(step, loss.item())
            viz.set_info(epoch, EPOCHS, lr_sched.get_last_lr()[0])

            if step % LOG_EVERY == 0:
                with open(LOG_FILE, "a") as _lf:
                    _lf.write(json.dumps({
                        "step":       step,
                        "epoch":      epoch,
                        "flow_loss":  flow_loss_val,
                        "clip_loss":  clip_loss_val,
                        "total_loss": loss.item(),
                        "lr":         lr_sched.get_last_lr()[0],
                        "ts":         time.time(),
                    }) + "\n")

            # sample generation only in GUI mode (expensive, nothing to show otherwise)
            if step % VIZ_EVERY == 0 and not no_gui:
                ema_model.eval()
                with torch.no_grad():
                    ctx_viz  = clip.encode(fixed_prompts, DEVICE)
                    lats_gen = flow.sample(
                        ema_model,
                        shape=(len(fixed_prompts), LATENT_CH, LATENT_SIZE, LATENT_SIZE),
                        ctx=ctx_viz, cfg_scale=CFG_SCALE)
                    imgs_gen = vae.decode(lats_gen)
                viz.update_samples([imgs_gen[i] for i in range(len(fixed_prompts))],
                                   fixed_prompts)
                ema_model.train()

            viz.flush()

        avg = epoch_loss / len(loader)
        epoch_msg = f"Epoch {epoch:>3}/{EPOCHS} | loss {avg:.5f} | lr {lr_sched.get_last_lr()[0]:.2e}"
        if no_gui:
            viz.log(epoch_msg)
        else:
            print(epoch_msg)

        if epoch % SAVE_EVERY == 0 or avg < best_loss:
            best_loss = min(best_loss, avg)
            p = f"checkpoint_epoch{epoch:04d}.pt"
            torch.save({"epoch": epoch, "step": step,
                        "model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": avg}, p)
            ckpt_msg = f"  → saved {p}"
            if no_gui:
                viz.log(ckpt_msg)
            else:
                print(ckpt_msg)

    viz.close()


# ── Inference ─────────────────────────────────────────────────────────────────

def generate(prompt, checkpoint, n=4, cfg=CFG_SCALE):
    clip = FrozenCLIP().to(DEVICE)
    vae  = FrozenVAE().to(DEVICE)
    model = build_model().to(DEVICE)
    ck    = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(ck.get("ema_model", ck["model"]))
    model.eval()
    flow = RectifiedFlow()
    with torch.no_grad():
        ctx  = clip.encode([prompt] * n, DEVICE)
        lats = flow.sample(model, (n, LATENT_CH, LATENT_SIZE, LATENT_SIZE),
                           ctx, cfg_scale=cfg)
        imgs = vae.decode(lats)
    _, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, img in zip(axes, imgs):
        ax.imshow((img.cpu().permute(1,2,0).numpy()*0.5+0.5).clip(0,1))
        ax.axis("off")
    plt.suptitle(prompt); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Flux2-style latent diffusion on Flickr8k")
    p.add_argument("--no_gui",     action="store_true",
                   help="CLI-only mode: rich terminal dashboard instead of matplotlib window")
    p.add_argument("--generate",   metavar="PROMPT", default=None,
                   help="Generate images from a text prompt")
    p.add_argument("--checkpoint", metavar="PATH",   default=None,
                   help="Checkpoint to load for generation")
    p.add_argument("--n",          type=int, default=4,
                   help="Number of images to generate (default 4)")
    p.add_argument("--cfg",        type=float, default=CFG_SCALE,
                   help=f"CFG guidance scale (default {CFG_SCALE})")
    args = p.parse_args()

    if args.generate:
        if not args.checkpoint:
            p.error("--generate requires --checkpoint")
        generate(args.generate, args.checkpoint, n=args.n, cfg=args.cfg)
    else:
        train(no_gui=args.no_gui)
