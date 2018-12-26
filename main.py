import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from validation import ClassificationValidator
from models import cifar10_resnet
from models.mnist_lenet import Net

from pytrainer.utils import weight_init
from pytrainer.callbacks import LearningRateScheduler
from pytrainer.lr_scheduler import Step_decay
from pytrainer import Trainer


if __name__ == '__main__':
    dataset_name = 'cifar10'
    if dataset_name == "cifar10":
       #=========================================================================
        # CIFAR
        model = cifar10_resnet.ResNet18()

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(root='CIFAR10', train=True, download=True, transform=transform_train)


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(root='CIFAR10', train=False, download=True, transform=transform_test)

        train_batch_size = 512
        test_batch_size = 512
        # =========================================================================
    if dataset_name == "mnist":
        # =========================================================================
        # MNIST
        model = Net()
        trainset = datasets.MNIST('mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

        testset = datasets.MNIST('mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        train_batch_size = 512
        test_batch_size = 1000
        # =========================================================================
    # Dataloader
    train_loader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        testset,
        batch_size=test_batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Classification validator
    validator = ClassificationValidator(val_loader, save_best=True, top5=False)

    # Model weights init
    weight_init(model)

    # Classification
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    lr_callback = LearningRateScheduler(Step_decay([15000, 25000]), type="batch")


    #==========================================================================
    # Report parameters
    log_every_n_steps = 500
    val_every_n_steps = 1000
    save_steps = 5000
    tag = dataset_name


    # ==========================================================================
    # Start processes

    t = Trainer()
    t.compile(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        validation=validator,
        callbacks=[lr_callback]
    )
    t.fit(
        train_loader,
        report_steps=log_every_n_steps,
        val_steps=val_every_n_steps,
        save_steps=save_steps,
        tag=tag,
    )
