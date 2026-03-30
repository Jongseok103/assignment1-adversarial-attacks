import torch
import torch.nn.functional as F


def fgsm_targeted(model, x, target, eps):
    model.eval()

    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, target)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv - eps * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def fgsm_untargeted(model, x, label, eps):
    model.eval()

    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, label)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def pgd_targeted(model, x, target, k, eps, eps_step):
    model.eval()

    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, target)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv - eps_step * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = x_adv.detach()

    return x_adv


def pgd_untargeted(model, x, label, k, eps, eps_step):
    model.eval()

    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, label)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + eps_step * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = x_adv.detach()

    return x_adv