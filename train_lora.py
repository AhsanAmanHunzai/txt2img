"""
train_lora.py — Fine-tune Stable Diffusion 1.5 with LoRA on Flickr8k
=====================================================================
Uses the same captions.txt format as train.py but fine-tunes a
pre-trained SD model instead of training from scratch.

Commercial licensing:
  SD 1.5   → CreativeML Open RAIL-M  (commercial use of outputs allowed)
  SD 2.1   → CreativeML OpenRAIL++   (same, swap MODEL_ID below)
  FLUX schnell → Apache 2.0          (fully open, swap MODEL_ID below,
                                       needs ~12 GB VRAM + different pipeline)

Why this works when train.py doesn't:
  SD 1.5 was trained on 2 billion image-caption pairs and already knows
  what dogs, mountains, sunflowers look like.  LoRA just steers it toward
  your dataset's style without forgetting any of that prior knowledge.
  Expect clear images after 2–4 epochs (~20 minutes on a T4).

Usage:
  python train_lora.py                                  # train
  python train_lora.py --generate "a dog on a mountain"
  python train_lora.py --generate "a sunflower field" --lora lora_epoch0004.pt
  python train_lora.py --epochs 10 --rank 16            # longer run, richer LoRA
"""

import os, glob, math, random, warnings, time, json, sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import (StableDiffusionPipeline, DDPMScheduler,
                        UNet2DConditionModel, AutoencoderKL)
from transformers import CLIPTokenizer, CLIPTextModel
import argparse

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID     = "runwayml/stable-diffusion-v1-5"
# Swap to one of these for different licences / quality:
#   "stabilityai/stable-diffusion-2-1-base"   # OpenRAIL++, same VRAM
#   "black-forest-labs/FLUX.1-schnell"         # Apache-2.0, needs 12 GB VRAM

IMG_DIR      = "./flickr8k/Images"
CAPTIONS_CSV = "./flickr8k/captions.txt"
IMG_SIZE     = 512        # SD was pre-trained at 512×512
BATCH_SIZE   = 4          # reduce to 2 if OOM
LR           = 1e-4
EPOCHS       = 20
LORA_RANK    = 8          # 4–32: higher = more expressive, more params
LORA_ALPHA   = 16         # scaling = alpha / rank; keep alpha = 2 × rank
SAVE_EVERY   = 2
LOG_FILE     = "lora_training_log.jsonl"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

torch.manual_seed(SEED)
random.seed(SEED)


# ── LoRA layer ─────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps an existing frozen Linear and adds a trainable low-rank delta:
        output = W·x  +  (B @ A) · x · (alpha / rank)
    B is zero-initialised → at init the layer is identical to the original.
    """
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.scale  = alpha / rank
        d_in, d_out = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, d_in)  * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    # diffusers accesses these on the wrapped linear
    @property
    def weight(self): return self.linear.weight
    @property
    def bias(self):   return self.linear.bias


def inject_lora(unet: nn.Module, rank: int, alpha: float):
    """
    Injects LoRA into every Q/K/V/Out projection of every attention block
    (both self-attention and cross-attention to text).
    Returns the list of LoRA layers so we can pass only them to the optimiser.
    """
    lora_layers = []
    for _name, module in unet.named_modules():
        if not hasattr(module, "to_q"):
            continue
        if not isinstance(module.to_q, nn.Linear):
            continue
        for attr in ("to_q", "to_k", "to_v"):
            orig = getattr(module, attr)
            lora = LoRALinear(orig, rank, alpha)
            setattr(module, attr, lora)
            lora_layers.append(lora)
        # to_out is a ModuleList([Linear, Dropout])
        if hasattr(module, "to_out") and isinstance(module.to_out[0], nn.Linear):
            orig = module.to_out[0]
            lora = LoRALinear(orig, rank, alpha)
            module.to_out[0] = lora
            lora_layers.append(lora)
    return lora_layers


def lora_state_dict(lora_layers):
    """Collect only LoRA A/B weights (small — ~50 MB for rank 8)."""
    state = {}
    for i, layer in enumerate(lora_layers):
        state[f"lora.{i}.A"] = layer.lora_A.data.cpu()
        state[f"lora.{i}.B"] = layer.lora_B.data.cpu()
    return state


def load_lora_state_dict(lora_layers, state):
    for i, layer in enumerate(lora_layers):
        layer.lora_A.data.copy_(state[f"lora.{i}.A"].to(layer.lora_A.device))
        layer.lora_B.data.copy_(state[f"lora.{i}.B"].to(layer.lora_B.device))


# ── Dataset ────────────────────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=512):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df["exists"] = df["image"].apply(
            lambda f: os.path.isfile(os.path.join(img_dir, f)))
        self.df      = df[df["exists"]].reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB")
        return self.tfm(img), row["caption"]


# ── Text encoding ──────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_text(tokenizer, text_encoder, texts, device):
    tok = tokenizer(texts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt")
    return text_encoder(
        input_ids=tok.input_ids.to(device),
        attention_mask=tok.attention_mask.to(device)
    ).last_hidden_state


# ── Training ───────────────────────────────────────────────────────────────────

def train(epochs=EPOCHS, rank=LORA_RANK, alpha=LORA_ALPHA, no_gui=False):
    print(f"Device: {DEVICE}  |  LoRA rank {rank}  |  Model: {MODEL_ID}")

    # ── load pretrained pieces ────────────────────────────────────────────────
    print("Loading tokenizer + text encoder…")
    tokenizer     = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder  = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    text_encoder  = text_encoder.to(DEVICE).eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    print("Loading VAE…")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    vae = vae.to(DEVICE).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    print("Loading UNet…")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    unet = unet.to(DEVICE)
    for p in unet.parameters():          # freeze all base weights
        p.requires_grad_(False)

    # ── inject LoRA ───────────────────────────────────────────────────────────
    lora_layers = inject_lora(unet, rank=rank, alpha=alpha)
    lora_params = [p for l in lora_layers for p in (l.lora_A, l.lora_B)]
    n_params = sum(p.numel() for p in lora_params)
    print(f"LoRA injected: {len(lora_layers)} layers  |  "
          f"{n_params/1e6:.2f}M trainable params  "
          f"(base UNet frozen)")

    # ── noise scheduler (same as SD training) ────────────────────────────────
    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = Flickr8kDataset(CAPTIONS_CSV, IMG_DIR, IMG_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, drop_last=True)
    print(f"Dataset: {len(dataset)} pairs  |  {len(loader)} steps/epoch")

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=1e-2)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5 * len(loader), T_mult=2, eta_min=1e-6)

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch, step, best_loss = 1, 0, float("inf")
    ckpts = sorted(glob.glob("lora_epoch*.pt"))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location=DEVICE)
        load_lora_state_dict(lora_layers, ck["lora"])
        optimizer.load_state_dict(ck["optimizer"])
        if "lr_sched" in ck:
            lr_sched.load_state_dict(ck["lr_sched"])
        start_epoch = ck["epoch"] + 1
        step        = ck.get("step", 0)
        best_loss   = ck.get("loss", float("inf"))
        print(f"Resumed {ckpts[-1]}  (epoch {ck['epoch']}, step {step})")

    # ── train loop ─────────────────────────────────────────────────────────────
    unet.train()
    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0

        for imgs, captions in loader:
            imgs = imgs.to(DEVICE)

            # encode images to latents (same VAE SD used during training)
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            # add noise at a random timestep
            noise       = torch.randn_like(latents)
            timesteps   = torch.randint(0, scheduler.config.num_train_timesteps,
                                        (latents.shape[0],), device=DEVICE).long()
            noisy_lats  = scheduler.add_noise(latents, noise, timesteps)

            # text conditioning
            text_emb = encode_text(tokenizer, text_encoder, list(captions), DEVICE)

            # UNet predicts the noise (standard DDPM objective)
            noise_pred = unet(noisy_lats, timesteps,
                              encoder_hidden_states=text_emb).sample
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            lr_sched.step()

            epoch_loss += loss.item()
            step += 1

            # progress
            if step % 20 == 0:
                lr_now = lr_sched.get_last_lr()[0]
                sys.stdout.write(
                    f"\r  ep {epoch:>3}/{epochs}  "
                    f"step {step:>7,}  "
                    f"loss {loss.item():.5f}  "
                    f"lr {lr_now:.2e}   ")
                sys.stdout.flush()

            # log
            if step % 10 == 0:
                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps({
                        "step": step, "epoch": epoch,
                        "loss": loss.item(),
                        "lr":   lr_sched.get_last_lr()[0],
                        "ts":   time.time(),
                    }) + "\n")

        avg = epoch_loss / len(loader)
        best_loss = min(best_loss, avg)
        sys.stdout.write("\n")
        print(f"Epoch {epoch:>3}/{epochs}  avg loss {avg:.5f}  "
              f"best {best_loss:.5f}  lr {lr_sched.get_last_lr()[0]:.2e}")

        if epoch % SAVE_EVERY == 0 or avg <= best_loss:
            p = f"lora_epoch{epoch:04d}.pt"
            torch.save({
                "epoch":     epoch,
                "step":      step,
                "lora":      lora_state_dict(lora_layers),
                "optimizer": optimizer.state_dict(),
                "lr_sched":  lr_sched.state_dict(),
                "loss":      avg,
            }, p)
            print(f"  → saved {p}")

    print("Training complete.")


# ── Generation ──────────────────────────────────────────────────────────────────

def generate(prompt, lora_path, n=4, guidance=7.5, steps=30, seed=None):
    print(f"Loading pipeline from {MODEL_ID}…")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    pipe.safety_checker = None   # remove for clean outputs during dev

    if lora_path:
        print(f"Loading LoRA weights from {lora_path}…")
        unet = pipe.unet
        for p in unet.parameters():
            p.requires_grad_(False)
        lora_layers = inject_lora(unet, rank=LORA_RANK, alpha=LORA_ALPHA)
        ck = torch.load(lora_path, map_location=DEVICE)
        load_lora_state_dict(lora_layers, ck["lora"])
        print(f"  LoRA epoch {ck['epoch']}, loss {ck['loss']:.5f}")

    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(seed)

    images = pipe(
        [prompt] * n,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images

    # save grid
    w, h   = images[0].size
    cols   = min(n, 4)
    rows   = math.ceil(n / cols)
    grid   = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    out = "generated.png"
    grid.save(out)
    print(f"Saved {n} images → {out}")
    return images


# ── Entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SD 1.5 + LoRA fine-tuning on Flickr8k")
    p.add_argument("--generate",  metavar="PROMPT", default=None)
    p.add_argument("--lora",      metavar="PATH",   default=None,
                   help="LoRA checkpoint to load for generation")
    p.add_argument("--epochs",    type=int,   default=EPOCHS)
    p.add_argument("--rank",      type=int,   default=LORA_RANK)
    p.add_argument("--alpha",     type=float, default=LORA_ALPHA)
    p.add_argument("--n",         type=int,   default=4,
                   help="Number of images to generate (default 4)")
    p.add_argument("--cfg",       type=float, default=7.5,
                   help="Guidance scale (default 7.5)")
    p.add_argument("--steps",     type=int,   default=30,
                   help="Diffusion steps (default 30)")
    p.add_argument("--seed",      type=int,   default=None)
    p.add_argument("--no_gui",    action="store_true")
    args = p.parse_args()

    if args.generate:
        generate(args.generate, args.lora, n=args.n,
                 guidance=args.cfg, steps=args.steps, seed=args.seed)
    else:
        train(epochs=args.epochs, rank=args.rank, alpha=args.alpha,
              no_gui=args.no_gui)
