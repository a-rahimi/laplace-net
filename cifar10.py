#!/usr/bin/env python3

from typing import Optional, Tuple

import argparse
import fsspec
import logging
import os
import sys

import matplotlib.pylab as plt
import torch
import torch.utils.tensorboard as torch_tb  # Import this before torchvision. Otherwise you get a segfault (https://github.com/pytorch/pytorch/issues/30651)
import torch.utils.data as data_utils
from torch import nn
import torchvision
import torchvision.transforms as transforms
import watchtower

from pytorch_resnet_cifar10 import resnet
import conductance

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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
    batch_size = 256
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
    with fsspec.open(os.path.join(checkpoints_dir, f"{epoch:03d}.pt"), "wb") as f:
        torch.save(
            {
                "epoch": epoch,
                "i_batch_cumulative": i_batch_cumulative,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f,
        )


def latest_checkpoint(checkpoints_dir: str) -> Optional[str]:
    checkpoints = [
        fname
        for fname in fsspec.open(checkpoints_dir).fs.find(checkpoints_dir)
        if fname.endswith(".pt")
    ]

    if checkpoints:
        # Return the lexicographically latest checkpoint.
        return "s3://" + max(checkpoints)


def resume_checkpoint(
    device: torch.device, checkpoints_dir: str, model: nn.Module, optimizer
) -> Tuple[int, int]:
    # Find the chronologically latest checkpoint
    checkpoint_path = latest_checkpoint(checkpoints_dir)
    if not checkpoint_path:
        # No checkpoints found.
        return 0, 0

    with fsspec.open(checkpoint_path, "rb") as f:
        d = torch.load(f, map_location=device)
    model.load_state_dict(d["model_state_dict"])
    optimizer.load_state_dict(d["optimizer_state_dict"])
    return d["epoch"], d["i_batch_cumulative"]


def train_and_eval(checkpoints_dir: str, resume: bool = True):
    tb_writer = torch_tb.SummaryWriter(checkpoints_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = conductance.PoissonNet()
    # model = resnet.resnet20()
    model.to(device)

    train_loader, val_loader = data_loaders()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    epoch, i_batch_cumulative = 0, 0
    if resume:
        epoch, i_batch_cumulative = resume_checkpoint(
            device, checkpoints_dir, model, optimizer
        )
        epoch += 1

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch, epoch + 100):
        log.info("-----Epoch %d-----", epoch)
        model.train()

        for i_batch, (images, labels) in enumerate(train_loader):
            i_batch_cumulative += 1
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)

            # Take an optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % 20 == 0:
                log.info("[%d, %5d] loss: %.3f", epoch, i_batch, float(loss))
                tb_writer.add_scalar("batch_loss", float(loss), i_batch_cumulative)

        model.eval()
        save_checkpoint(checkpoints_dir, epoch, i_batch_cumulative, model, optimizer)
        eval_accuracy = model_accuracy(model, device, val_loader)
        tb_writer.add_scalar("eval_accuracy", 100 * eval_accuracy, i_batch_cumulative)
        tb_writer.add_scalar("eval_accuracy_at_epoch", 100 * eval_accuracy, epoch)
        log.info("Eval accuracy at epoch %d %d %%", epoch, 100 * eval_accuracy)

    log.info("Finished training")


def setup_logging(experiment: str):
    # Set up environment variables to write tensorboard event files to S3.
    os.environ["AWS_REGION"] = "us-west-2"
    os.environ["S3_ENDPOINT"] = "https://s3-us-west-2.amazonaws.com"

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%H:%M:%S"
    )
    # The default handler goes to stderr. Only let warnings and errors go to stderr
    logging.root.handlers[0].setLevel(logging.WARNING)

    # Set up another handler that logs only low priority events to stdout.
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(lambda record: record.levelno <= logging.INFO)
    logging.root.addHandler(handler)

    handler = watchtower.CloudWatchLogHandler(
        log_group="experiments", stream_name=experiment
    )
    handler.setLevel(logging.DEBUG)
    logging.root.addHandler(handler)

    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--pdb", default=False, action="store_true")
    args = parser.parse_args()

    setup_logging(args.experiment)
    try:
        train_and_eval("s3://tensorboard-log/new/" + args.experiment)
    except Exception as e:
        # If the exception has a stderr field, add it to the log message.
        # CalledProcessError has such a field, and it captures the stderr of
        # the failed subprocess.
        log.error(
            "Exception caught: %s", getattr(e, "stderr", b"").decode(), exc_info=e
        )
        if args.pdb:
            import pdb

            pdb.post_mortem()
        else:
            raise