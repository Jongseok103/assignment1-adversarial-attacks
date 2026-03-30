import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, x):
    model.eval()
    logits = model(x)
    return logits.argmax(dim=1)


def attack_success_rate_targeted(model, loader, attack_fn, device, eps, max_samples=100):
    model.eval()

    total = 0
    success = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        target = (y + 1) % 10
        x_adv = attack_fn(model, x, target, eps)
        pred = predict(model, x_adv)

        success += (pred == target).sum().item()
        total += x.size(0)

        if total >= max_samples:
            break

    return success / total


def attack_success_rate_untargeted(model, loader, attack_fn, device, eps, max_samples=100):
    model.eval()

    total = 0
    success = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        x_adv = attack_fn(model, x, y, eps)
        pred = predict(model, x_adv)

        success += (pred != y).sum().item()
        total += x.size(0)

        if total >= max_samples:
            break

    return success / total


def save_attack_visualization(x, x_adv, pred_clean, pred_adv, filepath, is_grayscale=True, perturb_scale=5.0):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    x_cpu = x.detach().cpu()[0]
    x_adv_cpu = x_adv.detach().cpu()[0]
    perturb = x_adv_cpu - x_cpu

    if is_grayscale:
        img_clean = x_cpu.squeeze(0).numpy()
        img_adv = x_adv_cpu.squeeze(0).numpy()
        img_pert = perturb.squeeze(0).numpy()
        cmap = "gray"
    else:
        img_clean = x_cpu.permute(1, 2, 0).numpy()
        img_adv = x_adv_cpu.permute(1, 2, 0).numpy()
        img_pert = perturb.permute(1, 2, 0).numpy()
        cmap = None

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_clean, cmap=cmap)
    axes[0].set_title(f"Original\npred={pred_clean}")
    axes[0].axis("off")

    axes[1].imshow(img_adv, cmap=cmap)
    axes[1].set_title(f"Adversarial\npred={pred_adv}")
    axes[1].axis("off")

    axes[2].imshow((img_pert * perturb_scale + 0.5).clip(0, 1), cmap=cmap)
    axes[2].set_title(f"Perturbation x{perturb_scale}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()