import random
import math
from itertools import chain
from collections import Counter
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List

data = datasets.ImageNet(
    root = '~/Documents/Code/Gradient_Matching/data',
    split = 'train',
    transform = ToTensor(), 
)

test_data = datasets.ImageNet(
    root = '~/Documents/Code/Gradient_Matching/data',
    split = 'test',
    transform = ToTensor(), 
)

batch_size = 6000
# data = ConcatDataset([train_data, test_data])
data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

# Split the indices in a stratified way
indices = np.arange(len(data))
train_indices, test_indices = train_test_split(indices, train_size=100*10, stratify=data.targets, random_state=42)

# Warp into Subsets and DataLoaders
train_subset = Subset(data, train_indices)
subset_loader = DataLoader(train_subset, batch_size=1, shuffle=False)

fullset_loader = DataLoader(train_subset, batch_size=1000, shuffle=False)

test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False)

class SimpleCNN(nn.Module):
    """Defines a simple convolutional neural net"""
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


def pick_subset(mean_gradient: torch.Tensor, sample_grads: List[torch.Tensor]) -> List[int]:
    """Accept mean gradient and sample wise gradients to calculate  and return the indices of top 1000 datapoints (with highest cosine distances)"""
    list_g = gradient_closure(mean_gradient, sample_grads)
    list_g_indices = [int(data.targets[i]) for i in data.targets]

    zipped_list = zip(list_g, list_g_indices, list(range(60000)))
    zipped_list = sorted(zipped_list, reverse=True)

    top_1000 = []
    for label in range(10):
        dummy = 100
        for tup in zipped_list:
            if dummy == 0:
                break
            if tup[1] == label:
                top_1000.append(tup)
                dummy-=1

    top_1000 = sorted(top_1000, reverse=True)
    top_1000_indices = [i[2] for i in top_1000]

    return top_1000_indices


def gradient_closure(full_grad, subset_grad):
    """Returns the cosine distance between two gradient vectors calculated from different passes"""
    cosine_distances = []
    cosine = nn.CosineSimilarity(dim=0)
    for gradient in subset_grad:
        dist = cosine(full_grad, gradient)
        cosine_distances.append(float(dist))

    return cosine_distances


def subset_forward_pass_grads(model: SimpleCNN, dataloader: DataLoader) -> List[torch.Tensor]:
    """Returns the gradient of the loss wrt the final layer parameters after one forward pass"""
    sample_grads = []
    for _, (input, target) in enumerate(dataloader):
        output = model(input)
        loss = loss_fn(output, target)

        grad = torch.autograd.grad(loss, model.fc2.weight, retain_graph=True)[0]
        sample_grads.append(grad)

    for i, tensor in enumerate(sample_grads):
        sample_grads[i] = torch.flatten(tensor)

    return sample_grads


def mean_gradient(model: SimpleCNN, data_loader: DataLoader) -> torch.Tensor :
    """Returns the mean gradient of the last layer after one epoch"""
    gradients = []
    for input, target in data_loader:
        output = model(input)
        loss = loss_fn(output, target)
        grad_ = torch.autograd.grad(loss, model.fc2.weight, retain_graph=True)[0]
        gradients.append(grad_)

    mean_gradient = torch.flatten(torch.mean(torch.stack(gradients), dim=0))

    return mean_gradient


def full_model_train(model, dataloader, optimizer):

    """Trains given model on given dataloader created from full dataset:
    :model: An instantiation of SimpleCNN
    :dataloader: DataLoader object created full dataset
    """

    model.train()
    # for epoch in range(50):
    optimizer.zero_grad()
    for x, y in dataloader:
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
    optimizer.step()
        # print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

    return model


def subset_train(model, dataloader, optimizer, epochs=30):

    """Trains given model on given dataloader created from a subset of full dataset:
    :model: An instantiation of SimpleCNN
    :dataloader: DataLoader object created from Subset of full dataset
    """

    model.train()
    for epoch in range(epochs):
        for x, y in dataloader:
            output = model(x)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}', end="\r")

    return model


def test(model, test_dataloader=test_dataloader):
    """Return metrics of accuracy for given models"""
    correct = 0
    total = 0

    # Variables to calculate label wise accuracy
    correct_pred = {classname: 0 for classname in range(10)}
    total_pred = {classname: 0 for classname in range(10)}

    with torch.no_grad():
        for x, y in test_dataloader:
            output = model(x)
            
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            for label, prediction in zip(y, predicted):
                if label == prediction:
                    correct_pred[int(label)] += 1
                total_pred[int(label)] += 1
            
    accuracy = 100*(correct/total)
    classwise_accuracy = {i:100*correct_pred[i]/total_pred[i] for i in range(10)}

    return [accuracy, classwise_accuracy]


def with_new_init():
    """
    Returns a list of lists that are top 1000 indices with 50 different initializations of the model
    """

    list_of_lists = [] #Stores lists of top 100 indices lists for rank aggregatio later

    for epoch in range(50):
        torch.manual_seed(epoch)
        fullset_model = SimpleCNN()
        mean_grad = mean_gradient(fullset_model, fullset_loader)
        sample_wise_gradients = subset_forward_pass_grads(fullset_model, subset_loader)

        top_1000 = pick_subset(mean_grad, sample_wise_gradients)
        list_of_lists.append(top_1000)

        print(f"Epoch: {epoch} of finding subset while reinitializing full model", end='\r')

    return list_of_lists


def with_train():
    """
    Returns a list of lists where each list is top 1000 indices using cosine distance of mean gradient obtained after each epoch of full model training
    """
    # Initializing full dataset model
    fullset_model = SimpleCNN()
    fullset_optimizer = torch.optim.Adam(params=fullset_model.parameters(), lr=0.01)
    list_of_lists = [] #Stores lists of top 100 indices lists for rank aggregatio later

    for epoch in range(50):
        mean_grad = mean_gradient(fullset_model, fullset_loader)
        sample_wise_gradients = subset_forward_pass_grads(fullset_model, subset_loader)

        top_1000 = pick_subset(mean_grad, sample_wise_gradients)
        list_of_lists.append(top_1000)

        # torch.manual_seed(epoch)
        # t1_model = SimpleCNN()
        # t1_subset = Subset(data, top_1000)
        # t1_loader = DataLoader(t1_subset, batch_size=1000, shuffle=False)
        # t1_optimizer = torch.optim.Adam(t1_model.parameters(), lr=0.01)

        # t1_model = subset_train(t1_model, t1_loader, t1_optimizer)

        # t1_accuracy, t1_class_accuracy = test(t1_model)

        fullset_model = full_model_train(fullset_model, dataloader=fullset_loader, optimizer=fullset_optimizer)

        print(f"Epoch: {epoch} of finding subset while training full model", end='\r')
        # print(f"--------------EPOCH: {epoch}--------------")
        # print(f"Cosine distance based subset accuracy: {t1_accuracy}")
        # print(f"Cosine distance based classwise subset accuracy: {t1_class_accuracy}")
        # print(f"Random subset accuracy: {random_accuracy}")
        # print(f"Random subset classwise accuracy: {random_class_accuracy}")
        # print(f"Fullset accuracy: {fullset_accuracy}")
        # print(f"Fullset classwise accuracy: {fullset_class_accuracy}")
        # print()
        
        # with open("output.txt", "a") as f:
            # f.write(f"--------------EPOCH: {epoch}--------------\n")
            # f.write(f"Cosine distance based subset accuracy: {t1_accuracy} \n")
            # f.write(f"Cosine distance based classwise subset accuracy: {t1_class_accuracy}\n")
            # f.write(f"Random subset accuracy: {random_accuracy}\n")
            # f.write(f"Random subset classwise accuracy: {random_class_accuracy}\n")
            # f.write(f"Fullset accuracy: {fullset_accuracy}\n")
            # f.write(f"Fullset classwise accuracy: {fullset_class_accuracy}\n")
            # f.write("\n")

    return list_of_lists

def rank_aggregation(list_of_lists):
    freq_dict = Counter(chain.from_iterable((list_of_lists)))
    ordered_freq_list_indices = freq_dict.most_common()

    index_class_dict = {i:int(data.targets[i]) for i in data.targets}

    super_list = []

    for class_no in range(10):
        count = 0
        for item in ordered_freq_list_indices:
            if class_no == index_class_dict[item[0]]:
                super_list.append(item[0])
                count += 1

            if count == 100:
                break
    return super_list


big_random_acc = []
big_random_acc_classwise = []

# Changing initialization as well as train set for the random baseline testing
for test_no in range(10):
    torch.manual_seed(random.randint(1,500))
    random_model = SimpleCNN()
    random_indices, random_test_indices = train_test_split(indices, train_size=100*10, stratify=data.targets, random_state=random.randint(1,500))
    random_subset = Subset(data, random_indices)
    random_loader = DataLoader(random_subset, batch_size=1000, shuffle=False)
    random_optimizer = torch.optim.Adam(random_model.parameters(), lr=0.01)

    print(f"Iteration {test_no+1} of training on random subset", end='\r')
    random_model = subset_train(random_model, random_loader, random_optimizer, 20)
    random_accuracy, random_class_accuracy = test(random_model)
    big_random_acc.append(random_accuracy)
    big_random_acc_classwise.append(random_class_accuracy)

random_accuracy = sum(big_random_acc) / len(big_random_acc)
random_class_accuracy = {}
for i in range(10):
    random_class_accuracy[i] = sum(k[i] for k in big_random_acc_classwise) / 10

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", required=True, help = "With different initialization(1)/With train(2)")
args = parser.parse_args()

if int(args.number) == 1:
    list_of_lists = with_new_init()
    top_1000 = rank_aggregation(list_of_lists)

    t1_model = SimpleCNN()
    t1_subset = Subset(data, top_1000)
    t1_loader = DataLoader(t1_subset, batch_size=1000, shuffle=False)
    t1_optimizer = torch.optim.Adam(t1_model.parameters(), lr=0.01)

    t1_model = subset_train(t1_model, t1_loader, t1_optimizer, epochs=50)

    t1_accuracy, t1_class_accuracy = test(t1_model)

    print("Accuracy metrics using new initialization to pick subset:", t1_accuracy)
    print("Classwise accuracy metrics using new initialization to pick subset:", t1_class_accuracy)
    print("Accuracy metrics using random subset:", random_accuracy)
    print("Classwise accuracy metrics using random subset:", random_class_accuracy)

    #Saving list_of_lists for lat comparisons
    with open("new_init_list", "wb") as f:
        pickle.dump(list_of_lists, f)

elif int(args.number) == 2:
    list_of_lists = with_train()
    top_1000 = rank_aggregation(list_of_lists)

    t1_model = SimpleCNN()
    t1_subset = Subset(data, top_1000)
    t1_loader = DataLoader(t1_subset, batch_size=1000, shuffle=False)
    t1_optimizer = torch.optim.Adam(t1_model.parameters(), lr=0.01)

    t1_model = subset_train(t1_model, t1_loader, t1_optimizer, epochs=50)

    t1_accuracy, t1_class_accuracy = test(t1_model)

    print("Accuracy metrics using subset picked while training on full dataset: ", t1_accuracy)
    print("Classwises accuracy metrics using subset picked while training on full dataset: ", t1_accuracy)
    print("Accuracy metrics using random subset:", random_accuracy)
    print("Classwise accuracy metrics using random subset:", random_class_accuracy)

    with open("while_train_list", "wb") as f:
        pickle.dump(list_of_lists, f)

else:
    print("Valid arguments include only '1' and '2'")


fullset_model = SimpleCNN()
fullset_optimizer = torch.optim.Adam(params=fullset_model.parameters(), lr=0.01)
fullset_model = subset_train(fullset_model, dataloader=fullset_loader, optimizer=fullset_optimizer, epochs=50)
fullset_accuracy, fullset_class_accuracy = test(fullset_model)

print("Accuracy on full dataset:", fullset_accuracy)
print("Classwise accuracy on full dataset:", fullset_class_accuracy)
# [x] Add command line option to perform either first or second experiment
# [] Add validation procedure to train loop
# [x] Save the obtained subset indices
# [x] Write a function that perfroms rank aggregation
# [x] Test new subset obtained from frequency based rank aggregation on test set 


