from typing import Tuple

import glob
import functools
import os
import subprocess

import matplotlib.pylab as plt
import torch
import torch.utils.data as data_utils
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.tensorboard as torch_tb

from pytorch_resnet_cifar10 import resnet


@functools.lru_cache(1)
def git_HEAD_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf8").strip()


def model_accuracy(
    model: nn.Module, device: torch.device, testloader: data_utils.DataLoader
) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def data_loaders() -> Tuple[data_utils.DataLoader, data_utils.DataLoader]:
    batch_size = 128
    num_workers = 2

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_loader = data_utils.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = data_utils.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def save_checkpoint(
    checkpoints_dir: str,
    epoch: int,
    i_batch_cumulative: int,
    model: nn.Module,
    optimizer,
):
    torch.save(
        {
            "epoch": epoch,
            "i_batch_cumulative": i_batch_cumulative,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(checkpoints_dir, f"{epoch:03d}.pt"),
    )


def resume_checkpoint(
    device: torch.device, checkpoints_dir: str, model: nn.Module, optimizer
) -> Tuple[int, int]:
    # Find the chronologically latest checkpoint
    try:
        checkpoint_path = sorted(glob.glob(os.path.join(checkpoints_dir, "*.pt")))[-1]
    except IndexError:
        # No checkpoints found.
        return 0, 0

    d = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(d["model_state_dict"])
    optimizer.load_state_dict(d["optimizer_state_dict"])
    return d["epoch"], d["i_batch_cumulative"]


def train_and_eval(resume: bool = True):
    checkpoints_dir = os.path.join("checkpoints", git_HEAD_hash())
    os.makedirs(checkpoints_dir, exist_ok=True)
    tb_writer = torch_tb.SummaryWriter(checkpoints_dir, flush_secs=1000, max_queue=1000)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = resnet.resnet20()
    model.to(device)

    train_loader, val_loader = data_loaders()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epoch, i_batch_cumulative = 0, 0
    if resume:
        epoch, i_batch_cumulative = resume_checkpoint(device, checkpoints_dir, model, optimizer)
        epoch += 1

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch, 100):
        model.train()

        for i_batch, (images, labels) in enumerate(train_loader):
            i_batch_cumulative += 1
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)

            # Take an optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % 10 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch, i_batch, float(loss)))
                tb_writer.add_scalar("batch_loss", float(loss), i_batch_cumulative)

        model.eval()
        save_checkpoint(checkpoints_dir, epoch, i_batch_cumulative, model, optimizer)
        eval_accuracy = model_accuracy(model, device, val_loader)
        tb_writer.add_scalar("eval_accuracy", 100 * eval_accuracy, i_batch_cumulative)
        tb_writer.add_scalar("eval_accuracy_at_epoch", 100 * eval_accuracy, epoch)
        print("Eval accuracy: %d %%" % (100 * eval_accuracy))

    print("Finished training")


if __name__ == "__main__":
    train_and_eval()
