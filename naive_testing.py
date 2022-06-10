import math
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

data = datasets.MNIST(
    root = '~/Documents/Code/Gradient_Matching/data',
    train = True,
    transform = ToTensor(), 
    download = True,            
)

test_data = datasets.MNIST(
    root = '~/Documents/Code/Gradient_Matching/data',
    train = False,
    transform = ToTensor(), 
    download = True,            
)

batch_size = 7000
# data = ConcatDataset([train_data, test_data])
data_loader = DataLoader(data, batch_size=batch_size)

# Split the indices in a stratified way
indices = np.arange(len(data))
train_indices, test_indices = train_test_split(indices, train_size=100*10, stratify=data.targets, random_state=42)

# Warp into Subsets and DataLoaders
train_subset = Subset(data, train_indices)
subset_loader = DataLoader(train_subset, batch_size=1)
fullset_loader = DataLoader(train_subset, batch_size=1000)


class SimpleCNN(nn.Module):
    "Defines a simple convolutional neural net"
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1/math.sqrt(5), 1/math.sqrt(5))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def loss_fn(predictions, targets):
    return F.cross_entropy(predictions, targets)

def pick_subset():
    "Accept a dataloader object and return a dataloader object that is the subset of the input"
    pass

def gradient_closure(full_grad, subset_grad):
    "Returns the cosine distance between two gradient vectors calculated from different passes"
    cosine_distances = []
    cosine = nn.CosineSimilarity()
    for gradient in subset_grad:
        dist = cosine(full_grad, gradient)
        cosine_distances.append(dist)

    return cosine_distances

def forward_pass_grads(model, dataloader):
    "Returns the gradient of the loss wrt the final layer parameters after one forward pass"
    gradients = []
    for _, (input, target) in enumerate(dataloader):
        output = model(input)
        loss = loss_fn(output, target)

        grad = torch.autograd.grad(loss, model.fc2.weight, retain_graph=True)[0]
        gradients.append(grad)
    
    return torch.flatten(torch.mean(torch.stack(gradients, dim=1)))

def subset_forward_pass_grads(model, dataloader):
    "Returns the gradient of the loss wrt the final layer parameters after one forward pass"
    sample_grads = []
    for _, (input, target) in enumerate(dataloader):
        output = model(input)
        loss = loss_fn(output, target)

        grad = torch.autograd.grad(loss, model.fc2.weight, retain_graph=True)[0]
        sample_grads.append(grad)

    for i, tensor in enumerate(sample_grads):
        sample_grads[i] = torch.flatten(tensor)

    return sample_grads

# def compute_grad(model, sample, target):
    
    # sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    # target = target.unsqueeze(0)

    # prediction = model(sample)
    # loss = loss_fn(prediction, target)

    # return torch.autograd.grad(loss, model.fc2.weight)


# def compute_sample_grads(model, data, targets):
    # """ manually process each sample with per sample gradient """
    # sample_grads = [compute_grad(model, data[i], targets[i]) for i in range(batch_size)]
    # sample_grads = zip(*sample_grads)
    # sample_grads = [torch.stack(shards) for shards in sample_grads]
    # return sample_grads

# per_sample_grads = compute_sample_grads(model, data, targets)



# [x]Write general forward pass
# [x]Write custom sample wise forward pass
# [x]Write sampler to pass to custom forward pass
# [x]Write weight initializer
# []Compare gradient
# []Party!!!
