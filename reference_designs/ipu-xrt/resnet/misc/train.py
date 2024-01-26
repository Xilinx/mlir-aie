import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import resnet_cifar
import resnet_projected
from utils import count_parameters
from datetime import date
import sys
from torch.utils.data import DataLoader
from utils import ImageNetKaggle

today = date.today()
name_append = today.strftime("%m_%d_%Y")

project = "SingleBottleneck_x3_projected_with_bn"
dataset_name = "CIFAR"


if dataset_name == "CIFAR":
    num_classes = 10  # CIFAR
elif dataset_name == "Imagenet":
    num_classes = 1000  # Imagenet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# net = resnet.DoubleBottleneckModel(num_classes).to(device)
# net = resnet_cifar.ResNet18_CIFAR(num_classes).to(device)
# net = resnet_cifar.SingleBottleneckModel_CIFAR(num_classes).to(device)

# net = resnet_projected.SingleBottleneckModel_projected(num_classes).to(device)


lr_rate = 0.1  # default
lr_rate = 0.0001

net = resnet_projected.ResNet50_projected(num_classes).to(device)
net = resnet_projected.ResNet18_projected(num_classes).to(device)
net = resnet_projected.QuantSingleBottleneckModel2x_projected(num_classes).to(device)
net = resnet_projected.SingleBottleneckModel2x_projected(num_classes).to(device)
net = resnet_projected.ResNet50_projected_IMAGENET(num_classes).to(device)
# net=resnet_projected.Complete_SingleBottleneckModel2x_projected(num_classes).to(device)


# net=resnet_projected.Bottleneck_x2_NOQUANT_BREAK_CONV1(num_classes).to(device)

# net=resnet_projected.ResNet50_AIEConv2d_projected_NOQUANT(num_classes).to(device)
# net=resnet_projected.Complete_SingleBottleneckModel2x_projected_NOQUANT(num_classes).to(device)
net = resnet_projected.Complete_SingleBottleneckModel3x_projected_NOQUANT(
    num_classes
).to(device)
# net=resnet_projected.Complete_SingleBottleneckModel1x_projected_NOQUANT(num_classes).to(device)
# net=resnet_projected.Complete_SingleConvModel1x_projected_NOQUANT(num_classes).to(device)


net = resnet_projected.SingleBottleneck_x1_projected_with_bn(num_classes).to(device)
net = resnet_projected.SingleBottleneck_x1_plain_with_bn(num_classes).to(device)
net = resnet_projected.SingleBottleneck_x2_projected_with_bn(num_classes).to(device)

net = resnet_projected.SingleBottleneck_x3_projected_with_bn(num_classes).to(device)


workdir = (
    "weights_"
    + dataset_name
    + "_"
    + project
    + "_"
    + str(lr_rate)
    + "_"
    + str(name_append)
)

os.makedirs(workdir, exist_ok=True)

if dataset_name == "CIFAR":
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data_torchvision", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data_torchvision", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )


elif dataset_name == "Imagenet":
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = ImageNetKaggle("/group/rad/imagenet", "train", train_transform)
    val_dataset = ImageNetKaggle("/group/rad/imagenet", "val", val_transform)
    trainloader = DataLoader(
        train_dataset,
        batch_size=32,  # may need to reduce this depending on your GPU
        num_workers=8,  # may need to reduce this depending on your num of CPUs and RAM
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    testloader = DataLoader(
        val_dataset,
        batch_size=32,  # may need to reduce this depending on your GPU
        num_workers=8,  # may need to reduce this depending on your num of CPUs and RAM
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
original_stdout = sys.stdout  # Save a reference to the original standard output

with open(workdir + "/model.txt", "w") as f:
    sys.stdout = f  # Change the standard output to the file we created.
    count_parameters(net)
sys.stdout = original_stdout  # Reset the standard output to its original value

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

EPOCHS = 501


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print("\nEpoch : %d Test Acc : %.3f" % (epoch, 100.0 * correct / total))
        print("--------------------------------------------------------------")

    # Save checkpoint.


def train():
    net.train()
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(
            total=len(trainloader),
            desc="[0/{}]".format(len(trainloader.sampler)),
            ascii=True,
            leave=True,
            ncols=100,
            bar_format="{l_bar}{bar}| [{elapsed}{postfix}]",
        )

        chunks = 0
        with progress_bar:
            if epoch % 50 == 0:
                test(epoch)
                torch.save(
                    net.state_dict(),
                    os.path.join(
                        workdir, dataset_name + "_" + project + "_weight_%s.tar" % epoch
                    ),
                )
            for i, inp in enumerate(trainloader):
                chunks += inp[0].shape[0]
                inputs, labels = inp
                inputs, labels = inputs.to("cpu"), labels.to("cpu")
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar.set_postfix(loss="%.4f" % (running_loss / (i + 1)))
                progress_bar.set_description(
                    "[{}/{}]".format(chunks, len(trainloader.sampler))
                )
                progress_bar.update()

            avg_loss = sum(losses) / len(losses)
            scheduler.step(avg_loss)

    print("Training Done")


train()
print("Weights at: ", workdir)
