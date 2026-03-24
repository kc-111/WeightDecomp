import torchvision
from datasets import load_dataset

torchvision.datasets.MNIST(root="./data", download=True)
torchvision.datasets.CIFAR10(root="./data", download=True)
torchvision.datasets.CIFAR100(root="./data", download=True)
load_dataset("wikitext", "wikitext-103-raw-v1").save_to_disk("./data/wikitext-103")