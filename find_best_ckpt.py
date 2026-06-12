import glob, torch

ckpts = sorted(glob.glob("checkpoint_epoch*.pt"))
if not ckpts:
    print("No checkpoints found in current directory.")
    exit(1)

best_path, best_loss = None, float("inf")
for path in ckpts:
    ck = torch.load(path, map_location="cpu")
    loss = ck.get("loss", float("inf"))
    epoch = ck.get("epoch", "?")
    step  = ck.get("step", "?")
    marker = ""
    if loss < best_loss:
        best_loss = loss
        best_path = path
        marker = "  ← best so far"
    print(f"  {path}  |  epoch {epoch}  |  step {step:,}  |  loss {loss:.5f}{marker}")

print(f"\nBest checkpoint: {best_path}  (loss {best_loss:.5f})")
print(f"\nGenerate with:")
print(f"  python train.py --generate \"a sunflower field\" --checkpoint {best_path} --cfg 7.5")
