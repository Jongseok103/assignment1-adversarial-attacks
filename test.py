import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

from models import MNISTCNN, build_resnet18_for_cifar10
from train import fit, evaluate
from attacks import (
    fgsm_targeted,
    fgsm_untargeted,
    pgd_targeted,
    pgd_untargeted,
)
from utils import (
    set_seed,
    predict,
    attack_success_rate_targeted,
    attack_success_rate_untargeted,
    save_attack_visualization,
)

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

MNIST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "mnist_cnn.pt")
CIFAR_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "cifar10_resnet18.pt")
CSV_PATH = os.path.join(RESULTS_DIR, "attack_success_rates.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def pgd_targeted_wrapper(model, x, target, eps, k=40, eps_step=0.01):
    return pgd_targeted(model, x, target, k=k, eps=eps, eps_step=eps_step)


def pgd_untargeted_wrapper(model, x, label, eps, k=40, eps_step=0.01):
    return pgd_untargeted(model, x, label, k=k, eps=eps, eps_step=eps_step)


def save_examples(
    model,
    loader,
    attack_name,
    attack_fn,
    dataset_name,
    device,
    is_grayscale=True,
    targeted=False,
    eps=0.2,
    n_examples=5,
    class_names=None,
):
    saved = 0
    model.eval()

    dataset_result_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(dataset_result_dir, exist_ok=True)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        for i in range(x.size(0)):
            x_one = x[i:i+1]
            y_one = y[i:i+1]

            pred_clean = predict(model, x_one)

            if targeted:
                target = (y_one + 1) % 10
                x_adv = attack_fn(model, x_one, target, eps)
            else:
                x_adv = attack_fn(model, x_one, y_one, eps)

            pred_adv = predict(model, x_adv)

            eps_str = f"{eps:.4f}".replace(".", "p")

            filepath = os.path.join(
                dataset_result_dir,
                f"{dataset_name}_{attack_name}_eps_{eps_str}_{saved:02d}.png"
            )

            save_attack_visualization(
                x_one,
                x_adv,
                pred_clean.item(),
                pred_adv.item(),
                filepath=filepath,
                is_grayscale=is_grayscale,
                class_names=class_names,
                eps=eps,
            )

            saved += 1
            if saved >= n_examples:
                return

def load_or_train_mnist(device):
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    model = MNISTCNN().to(device)

    if os.path.exists(MNIST_CKPT_PATH):
        print(f"Loading MNIST checkpoint from: {MNIST_CKPT_PATH}")
        state_dict = torch.load(MNIST_CKPT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("MNIST checkpoint not found. Training from scratch...")
        fit(model, train_loader, test_loader, device, epochs=5, lr=1e-3)
        torch.save(model.state_dict(), MNIST_CKPT_PATH)
        print(f"Saved MNIST checkpoint to: {MNIST_CKPT_PATH}")

    model.eval()
    return model, train_loader, test_loader


def load_or_train_cifar10(device):
    # train에는 augmentation 추가
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # test는 단순 ToTensor만 사용
    test_transform = transforms.ToTensor()

    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    model = build_resnet18_for_cifar10().to(device)

    if os.path.exists(CIFAR_CKPT_PATH):
        print(f"Loading CIFAR-10 checkpoint from: {CIFAR_CKPT_PATH}")
        state_dict = torch.load(CIFAR_CKPT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("CIFAR-10 checkpoint not found. Training ResNet18 from scratch...")
        fit(model, train_loader, test_loader, device, epochs=50, lr=1e-3)
        torch.save(model.state_dict(), CIFAR_CKPT_PATH)
        print(f"Saved CIFAR-10 checkpoint to: {CIFAR_CKPT_PATH}")

    model.eval()
    return model, train_loader, test_loader


def run_mnist(device, all_results):
    print("\n===== MNIST =====")
    model, _, test_loader = load_or_train_mnist(device)

    criterion = torch.nn.CrossEntropyLoss()
    _, clean_acc = evaluate(model, test_loader, criterion, device)
    print(f"MNIST clean accuracy: {clean_acc:.4f}")

    eps_list = [0.1, 0.2, 0.3, 0.31, 0.32, 0.33, 0.34] 

    for eps in eps_list:
        fgsm_t = attack_success_rate_targeted(
            model, test_loader, fgsm_targeted, device, eps=eps, max_samples=256
        )
        fgsm_u = attack_success_rate_untargeted(
            model, test_loader, fgsm_untargeted, device, eps=eps, max_samples=256
        )
        pgd_t = attack_success_rate_targeted(
            model,
            test_loader,
            lambda m, x, t, e: pgd_targeted_wrapper(m, x, t, e, k=40, eps_step=0.01),
            device,
            eps=eps,
            max_samples=256,
        )
        pgd_u = attack_success_rate_untargeted(
            model,
            test_loader,
            lambda m, x, y, e: pgd_untargeted_wrapper(m, x, y, e, k=40, eps_step=0.01),
            device,
            eps=eps,
            max_samples=256,
        )

        all_results.append({
            "dataset": "MNIST",
            "eps": eps,
            "FGSM_targeted": fgsm_t,
            "FGSM_untargeted": fgsm_u,
            "PGD_targeted": pgd_t,
            "PGD_untargeted": pgd_u,
            "clean_accuracy": clean_acc,
        })

        print(
            f"[MNIST][eps={eps}] "
            f"FGSM-T={fgsm_t:.4f}, FGSM-U={fgsm_u:.4f}, "
            f"PGD-T={pgd_t:.4f}, PGD-U={pgd_u:.4f}"
        )

    save_examples(
        model, test_loader, "fgsm_targeted", fgsm_targeted,
        "mnist", device, is_grayscale=True, targeted=True
    )
    save_examples(
        model, test_loader, "fgsm_untargeted", fgsm_untargeted,
        "mnist", device, is_grayscale=True, targeted=False
    )
    save_examples(
        model,
        test_loader,
        "pgd_targeted",
        lambda m, x, t, e: pgd_targeted_wrapper(m, x, t, e, k=40, eps_step=0.01),
        "mnist",
        device,
        is_grayscale=True,
        targeted=True,
        eps=0.2,
    )
    save_examples(
        model,
        test_loader,
        "pgd_untargeted",
        lambda m, x, y, e: pgd_untargeted_wrapper(m, x, y, e, k=40, eps_step=0.01),
        "mnist",
        device,
        is_grayscale=True,
        targeted=False,
        eps=0.2,
    )


def run_cifar10(device, all_results):
    print("\n===== CIFAR-10 =====")
    model, _, test_loader = load_or_train_cifar10(device)

    criterion = torch.nn.CrossEntropyLoss()
    _, clean_acc = evaluate(model, test_loader, criterion, device)
    print(f"CIFAR-10 clean accuracy: {clean_acc:.4f}")

    eps_list = [2/255, 4/255, 8/255, 16/255]

    for eps in eps_list:
        fgsm_t = attack_success_rate_targeted(
            model, test_loader, fgsm_targeted, device, eps=eps, max_samples=256
        )
        fgsm_u = attack_success_rate_untargeted(
            model, test_loader, fgsm_untargeted, device, eps=eps, max_samples=256
        )
        pgd_t = attack_success_rate_targeted(
            model,
            test_loader,
            lambda m, x, t, e: pgd_targeted_wrapper(m, x, t, e, k=20, eps_step=2/255),
            device,
            eps=eps,
            max_samples=256,
        )
        pgd_u = attack_success_rate_untargeted(
            model,
            test_loader,
            lambda m, x, y, e: pgd_untargeted_wrapper(m, x, y, e, k=20, eps_step=2/255),
            device,
            eps=eps,
            max_samples=256,
        )

        all_results.append({
            "dataset": "CIFAR-10",
            "eps": eps,
            "FGSM_targeted": fgsm_t,
            "FGSM_untargeted": fgsm_u,
            "PGD_targeted": pgd_t,
            "PGD_untargeted": pgd_u,
            "clean_accuracy": clean_acc,
        })

        print(
            f"[CIFAR10][eps={eps:.5f}] "
            f"FGSM-T={fgsm_t:.4f}, FGSM-U={fgsm_u:.4f}, "
            f"PGD-T={pgd_t:.4f}, PGD-U={pgd_u:.4f}"
        )
        

    save_examples(
        model,
        test_loader,
        "fgsm_targeted",
        fgsm_targeted,
        "cifar10",
        device,
        is_grayscale=False,
        targeted=True,
        eps=8/255,
        class_names=CIFAR10_CLASSES,
    )
    save_examples(
        model,
        test_loader,
        "fgsm_untargeted",
        fgsm_untargeted,
        "cifar10",
        device,
        is_grayscale=False,
        targeted=False,
        eps=8/255,
        class_names=CIFAR10_CLASSES,
    )
    save_examples(
        model,
        test_loader,
        "pgd_targeted",
        lambda m, x, t, e: pgd_targeted_wrapper(m, x, t, e, k=20, eps_step=2/255),
        "cifar10",
        device,
        is_grayscale=False,
        targeted=True,
        class_names=CIFAR10_CLASSES,
        eps=8/255,
    )
    save_examples(
        model,
        test_loader,
        "pgd_untargeted",
        lambda m, x, y, e: pgd_untargeted_wrapper(m, x, y, e, k=20, eps_step=2/255),
        "cifar10",
        device,
        is_grayscale=False,
        targeted=False,
        class_names=CIFAR10_CLASSES,
        eps=8/255,
    )


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_results = []

    run_mnist(device, all_results)
    run_cifar10(device, all_results)

    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)

    print("\nAll experiments finished.")
    print(f"Saved results to: {RESULTS_DIR}/")
    print(f"Saved CSV to: {CSV_PATH}")


if __name__ == "__main__":
    main()