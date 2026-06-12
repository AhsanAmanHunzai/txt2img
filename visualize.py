"""
visualize.py — Training health dashboard for txt2img training logs.

Usage:
  python visualize.py                          # read training_log.jsonl, show plot
  python visualize.py --file my_log.jsonl      # custom log file
  python visualize.py --save dashboard.png     # save to PNG instead of showing
  python visualize.py --watch                  # auto-refresh every 30s (saves to PNG)
  python visualize.py --watch --interval 60    # custom refresh interval (seconds)
"""

import json
import time
import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_log(path):
    if not os.path.isfile(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def ema(values, alpha=0.02):
    out = []
    for v in values:
        out.append(v if not out else (1 - alpha) * out[-1] + alpha * v)
    return out


def style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#888", labelsize=7)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_color("#2d333b")


# ── Dashboard ─────────────────────────────────────────────────────────────────

def build_dashboard(entries, save_path=None):
    if not entries:
        print("No log entries found — nothing to plot yet.")
        return

    steps       = [e["step"]       for e in entries]
    flow_losses = [e["flow_loss"]   for e in entries]
    total_losses= [e["total_loss"]  for e in entries]
    lrs         = [e["lr"]          for e in entries]
    epochs_list = [e["epoch"]       for e in entries]
    timestamps  = [e.get("ts", 0)   for e in entries]

    clip_steps  = [e["step"]       for e in entries if e.get("clip_loss") is not None]
    clip_vals   = [e["clip_loss"]  for e in entries if e.get("clip_loss") is not None]

    epoch_buckets = defaultdict(list)
    for e in entries:
        epoch_buckets[e["epoch"]].append(e["flow_loss"])
    ep_nums = sorted(epoch_buckets)
    ep_avgs = [float(np.mean(epoch_buckets[ep])) for ep in ep_nums]
    ep_mins = [float(np.min(epoch_buckets[ep]))  for ep in ep_nums]

    flow_ema    = ema(flow_losses, alpha=0.02)
    clip_ema    = ema(clip_vals,   alpha=0.05) if clip_vals else []

    # ── timing estimate ───────────────────────────────────────────────────────
    steps_per_sec = None
    eta_str = "n/a"
    if len(timestamps) >= 2 and timestamps[-1] and timestamps[0]:
        elapsed = timestamps[-1] - timestamps[0]
        n_steps = steps[-1] - steps[0]
        if elapsed > 0 and n_steps > 0:
            steps_per_sec = n_steps / elapsed

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
    fig.suptitle("Training Health Dashboard", color="white", fontsize=13,
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                           left=0.06, right=0.97, top=0.93, bottom=0.10)

    # ── 1. Flow loss (raw + EMA) ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)
    ax1.plot(steps, flow_losses, color="#00d4ff", lw=0.5, alpha=0.25, label="raw")
    ax1.plot(steps, flow_ema,    color="#ff6b35", lw=1.8,              label="EMA")
    if any(tl != fl for tl, fl in zip(total_losses, flow_losses)):
        ax1.plot(steps, total_losses, color="#a0f0a0", lw=0.8, alpha=0.5,
                 label="total (flow + CLIP)")
    best_idx   = int(np.argmin(flow_ema))
    ax1.axvline(steps[best_idx], color="#ffd700", lw=0.8, linestyle="--", alpha=0.6)
    ax1.annotate(f"best EMA\n{flow_ema[best_idx]:.5f}",
                 xy=(steps[best_idx], flow_ema[best_idx]),
                 xytext=(10, 10), textcoords="offset points",
                 color="#ffd700", fontsize=6, arrowprops=dict(arrowstyle="->", color="#ffd700"))
    ax1.set_title("Flow Loss (MSE on velocity field)")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7, loc="upper right")

    # ── 2. Stats summary panel ────────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[0, 2])
    style_ax(ax_stats)
    ax_stats.set_axis_off()
    current_epoch = epochs_list[-1]
    current_ema   = flow_ema[-1]
    best_ema_val  = min(flow_ema)
    best_epoch    = ep_nums[int(np.argmin(ep_avgs))] if ep_avgs else "?"
    sps_str       = f"{steps_per_sec:.1f} steps/s" if steps_per_sec else "n/a"

    lines = [
        ("Current epoch",   f"{current_epoch}"),
        ("Total steps",     f"{steps[-1]:,}"),
        ("Current EMA",     f"{current_ema:.5f}"),
        ("Best EMA",        f"{best_ema_val:.5f}"),
        ("Best epoch",      f"{best_epoch}"),
        ("Speed",           sps_str),
        ("CLIP loss pts",   f"{len(clip_vals)}"),
    ]
    y = 0.92
    ax_stats.text(0.5, 1.02, "Quick Stats", ha="center", va="top",
                  color="white", fontsize=9, fontweight="bold",
                  transform=ax_stats.transAxes)
    for label, val in lines:
        ax_stats.text(0.05, y, label, color="#888", fontsize=8,
                      transform=ax_stats.transAxes)
        ax_stats.text(0.97, y, val, color="#00d4ff", fontsize=8,
                      ha="right", transform=ax_stats.transAxes)
        y -= 0.13

    # ── 3. CLIP loss ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    if clip_vals:
        ax2.scatter(clip_steps, clip_vals, color="#c77dff", s=6, alpha=0.5, label="raw")
        ax2.plot(clip_steps, clip_ema,     color="#ff9ef5", lw=1.5,         label="EMA")
        ax2.axhline(0, color="#444", lw=0.5, linestyle="--")
        ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)
    else:
        ax2.text(0.5, 0.5, "No CLIP loss yet\n(runs every 20 steps)",
                 ha="center", va="center", color="#555", fontsize=8,
                 transform=ax2.transAxes)
    ax2.set_title("CLIP Semantic Loss (1 − cos_sim)")
    ax2.set_xlabel("Step"); ax2.set_ylabel("CLIP loss")

    # ── 4. Per-epoch avg loss ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    ax3.plot(ep_nums, ep_avgs, color="#ffd700", lw=1.8, marker="o",
             markersize=3, label="avg loss")
    ax3.fill_between(ep_nums, ep_mins, ep_avgs, alpha=0.15, color="#ffd700",
                     label="min–avg range")
    best_ep_idx = int(np.argmin(ep_avgs))
    ax3.axhline(ep_avgs[best_ep_idx], color="#ff4444", lw=0.8, linestyle="--",
                label=f"best {ep_avgs[best_ep_idx]:.4f} @ ep {ep_nums[best_ep_idx]}")
    ax3.set_title("Per-Epoch Average Loss")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Avg Loss")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)

    # ── 5. Learning rate ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4)
    ax4.plot(steps, lrs, color="#4fc3f7", lw=1.2)
    ax4.set_title("Learning Rate Schedule")
    ax4.set_xlabel("Step"); ax4.set_ylabel("LR")
    ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # ── 6. Loss distribution histogram ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5)
    ax5.hist(flow_losses, bins=60, color="#00d4ff", alpha=0.7, edgecolor="none")
    ax5.axvline(np.median(flow_losses), color="#ffd700", lw=1.2,
                linestyle="--", label=f"median {np.median(flow_losses):.4f}")
    ax5.set_title("Loss Distribution (all steps)")
    ax5.set_xlabel("Flow Loss"); ax5.set_ylabel("Count")
    ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)

    # ── 7. Loss volatility (rolling std) ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6)
    window = max(20, len(flow_losses) // 40)
    if len(flow_losses) >= window:
        roll_std = [float(np.std(flow_losses[max(0, i-window):i+1]))
                    for i in range(len(flow_losses))]
        ax6.plot(steps, roll_std, color="#ff9800", lw=1.0)
        ax6.set_title(f"Loss Volatility (rolling std, w={window})")
        ax6.set_xlabel("Step"); ax6.set_ylabel("Std Dev")
    else:
        ax6.text(0.5, 0.5, "Not enough data yet", ha="center", va="center",
                 color="#555", fontsize=8, transform=ax6.transAxes)
        ax6.set_title("Loss Volatility")

    # ── 8. Steps per epoch bar ────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    style_ax(ax7)
    ep_counts = {ep: len(v) for ep, v in epoch_buckets.items()}
    if ep_counts:
        ax7.bar(list(ep_counts.keys()), list(ep_counts.values()),
                color="#7eb8f7", alpha=0.8, width=0.7)
        ax7.set_title("Logged Steps per Epoch")
        ax7.set_xlabel("Epoch"); ax7.set_ylabel("Log entries")
    else:
        ax7.set_title("Logged Steps per Epoch")

    # ── timestamp footer ──────────────────────────────────────────────────────
    last_ts = timestamps[-1]
    ts_str  = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts)) if last_ts else "?"
    fig.text(0.5, 0.01, f"Last log entry: {ts_str}  |  {len(entries)} total records",
             ha="center", color="#555", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[{time.strftime('%H:%M:%S')}] Saved → {save_path}  ({len(entries)} records)")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Training health dashboard")
    parser.add_argument("--file",     default="training_log.jsonl",
                        help="Path to the JSONL log file (default: training_log.jsonl)")
    parser.add_argument("--save",     default=None, metavar="PNG",
                        help="Save dashboard to this PNG path instead of showing")
    parser.add_argument("--watch",    action="store_true",
                        help="Auto-refresh mode: re-render on each interval")
    parser.add_argument("--interval", type=int, default=30,
                        help="Refresh interval in seconds for --watch (default: 30)")
    args = parser.parse_args()

    if args.watch:
        out = args.save or "dashboard.png"
        print(f"Watch mode — refreshing every {args.interval}s → {out}")
        print("Press Ctrl+C to stop.\n")
        matplotlib.use("Agg")
        try:
            while True:
                entries = load_log(args.file)
                build_dashboard(entries, save_path=out)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        entries = load_log(args.file)
        build_dashboard(entries, save_path=args.save)


if __name__ == "__main__":
    main()
