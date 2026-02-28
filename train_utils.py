import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import torchprofile
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np


def calculate_metrics(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total


def get_comp_cost(model, input_size, device, is_transformer=False, vocab_size=None):
    """
    Calculate FLOPs using torchprofile.
    For Transformer: torchprofile may undercount due to optimized SDPA kernels.
    An additional analytical correction is applied in run_experiment().
    """
    if is_transformer:
        # [batch, seq_len] format to match batch_first=True
        inputs = torch.randint(0, vocab_size, input_size).to(device)
    else:
        inputs = torch.randn(input_size).to(device)

    try:
        flops = torchprofile.profile_macs(model, inputs)
    except Exception as e:
        print(f"torchprofile failed: {e}")
        flops = 0
    return flops


def analytical_transformer_flops(seq_len, d_model, nlayers):
    """
    Analytical FLOPs estimate for Transformer attention layers.
    Per layer: FLOPs ≈ 2 * L^2 * d  (for scaled dot-product attention)
    Total across all layers.
    """
    return 2 * (seq_len**2) * d_model * nlayers


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    grad_norms = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients (critical for Mish/Swish)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Record gradient norm AFTER clipping (reflects actual update magnitude)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        grad_norms.append(total_norm)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    end_time = time.time()
    epoch_time = end_time - start_time

    avg_grad_norm = np.mean(grad_norms)
    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    return (
        running_loss / total,
        100 * correct / total,
        epoch_time,
        avg_grad_norm,
        peak_mem,
    )


def run_experiment(
    model_fn, dataset_fn, dataset_name, activation, epochs=30, batch_size=128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment: {dataset_name} with {activation} on {device}")

    torch.manual_seed(42)

    vocab_size = None
    input_sample_size = None

    if dataset_name == "CIFAR10":
        train_loader, val_loader, test_loader = dataset_fn(batch_size=batch_size)
        input_sample_size = (1, 3, 32, 32)
    else:  # AGNews
        train_loader, val_loader, test_loader, vocab_size = dataset_fn(
            batch_size=batch_size, vocab_size=20000
        )
        # [batch, seq_len] — matches batch_first=True
        input_sample_size = (1, 100)

    print(
        f"Dataset loaders prepared. Train: {len(train_loader)}, "
        f"Val: {len(val_loader)}, Test: {len(test_loader)} batches"
    )

    # Instantiate model
    if dataset_name == "CIFAR10":
        model = model_fn(activation=activation, num_classes=10).to(device)
    else:
        model = model_fn(
            ntoken=vocab_size,
            d_model=256,
            nhead=4,
            d_hid=512,
            nlayers=2,
            num_classes=4,
            activation=activation,
        ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Separate optimiser configs per task:
    # - CIFAR-10 (ResNet): higher LR, standard weight decay
    # - AG News (Transformer): lower LR, stronger weight decay for stability
    if dataset_name == "CIFAR10":
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # CosineAnnealing decays LR smoothly — better than StepLR for 30 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Calculate FLOPs
    flops = 0
    try:
        flops = get_comp_cost(
            model,
            input_sample_size,
            device,
            is_transformer=(dataset_name == "AGNews"),
            vocab_size=(vocab_size if dataset_name == "AGNews" else None),
        )
        # Add analytical attention FLOPs for Transformer (torchprofile misses SDPA)
        if dataset_name == "AGNews":
            attn_flops = analytical_transformer_flops(
                seq_len=100, d_model=256, nlayers=2
            )
            flops += attn_flops
            print(f"Transformer FLOPs (profile + analytical attention): {flops:,}")
        else:
            print(f"Model FLOPs: {flops:,}")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")

    results = []
    total_training_time = 0

    for epoch in range(epochs):
        train_loss, train_acc, epoch_time, avg_grad_norm, peak_mem = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = calculate_metrics(model, val_loader, criterion, device)
        scheduler.step()

        total_training_time += epoch_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"TrL: {train_loss:.4f}, TrA: {train_acc:.2f}% | "
            f"VaL: {val_loss:.4f}, VaA: {val_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s | GradNorm: {avg_grad_norm:.4f}"
        )

        results.append(
            {
                "dataset": dataset_name,
                "activation": activation,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time": epoch_time,
                "total_training_time": total_training_time,
                "avg_grad_norm": avg_grad_norm,
                "peak_memory_mb": peak_mem,
                "flops": flops,
            }
        )

    # Final evaluation on test set
    test_loss, test_acc = calculate_metrics(model, test_loader, criterion, device)
    print(f"Final Test Result — Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    results[-1]["test_loss"] = test_loss
    results[-1]["test_acc"] = test_acc

    return results, model.state_dict()


if __name__ == "__main__":
    pass
