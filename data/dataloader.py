from operator import imod
from xmlrpc.client import TRANSPORT_ERROR
from sklearn.utils import shuffle
from sympy import root
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform= transform)

traindataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform= transform)

traindataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)