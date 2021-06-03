import os
import subprocess

import matplotlib.pylab as plt
import torch
import torch.utils.data as data_utils
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models
import torch.utils.tensorboard as torch_tb


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


def train_and_eval():
    os.makedirs("checkpoints", exist_ok=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 64
    num_workers = 2
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = data_utils.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = data_utils.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    tb_writer = torch_tb.SummaryWriter(
        "checkpoints/" + git_HEAD_hash(), flush_secs=1000, max_queue=1000
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10, bias=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    i_batch_cumulative = 0
    for epoch in range(100):
        running_loss = 0.0
        model.train()

        for i_batch, (images, labels) in enumerate(trainloader, 0):
            i_batch_cumulative += 1
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)

            # Take an optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Average and print the loss of the last few batches.
            running_loss += loss.item()
            if i_batch % 10 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch, i_batch, running_loss / 10))
                tb_writer.add_scalar("batch_loss", running_loss, i_batch_cumulative)
                running_loss = 0.0

        model.eval()
        torch.save(model.state_dict(), f"checkpoints/{epoch:03d}")
        eval_accuracy = model_accuracy(model, device, testloader)
        tb_writer.add_scalar("eval_accuracy", 100 * eval_accuracy, i_batch_cumulative)
        print("Eval accuracy: %d %%" % (100 * eval_accuracy))

    print("Finished training")


if __name__ == "__main__":
    train_and_eval()
