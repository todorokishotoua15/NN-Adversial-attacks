import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import urllib
from PIL import Image
import requests
from timeit import default_timer as timer
from tqdm import tqdm
from torchvision import datasets
from torchmetrics import Accuracy
import numpy as np
import  matplotlib.pyplot as plt

torch.cuda.empty_cache()

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

# Getting Pretrained Inception V3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.to(device)
model.eval()

print("Loading the dataset")
start = timer()

preprocess = transforms.Compose([
    transforms.Resize(299),
    # transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = torchvision.datasets.ImageNet(root="data", split="val", transform=preprocess)
val_dataset_subset = torch.utils.data.Subset(val_dataset, np.random.choice(len(val_dataset), 100, replace=False))
end = timer()
print(f"Time taken to load the dataset : {end-start : .3f}")

val_loader = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)

batch1 = next(iter(val_loader))
# print(f"Shape of the input : {batch1[0].shape}\nDtype of the input : {batch1[0].dtype}\nDevice of the input : {batch1[0].device}")
# print(len(val_dataset), len(val_loader))

accuracy = Accuracy(task="multiclass", num_classes=1000).to(device)
class_to_idx = val_dataset.class_to_idx
# print(class_to_idx)
idx_to_class = {a : b for (b,a) in class_to_idx.items()}

def get_base_acc():
    correct = 0

    val_acc = 0
    cnt = 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(val_loader)):
            X, y = X.to(device), y.to(device)
            # print("here")
            
            logits = model(X)
            labels = torch.argmax(F.softmax(logits, dim=1),dim=1)
            val_acc += accuracy(labels,y)
            cnt += 1
            
           
    return (val_acc.item())/len(val_loader)

# restores the tensors to their original scale
def denorm(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image,0,1)
    return perturbed_image

def test_fgsm_attack(model, device, val_loader, epsilon):
    correct = 0
    adv_examples = []

    for data, target in tqdm(val_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = torch.argmax(F.softmax(output, dim=1),dim=1)

        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output,target)

        model.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        data_denorm = denorm(data)

        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        perturbed_data_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_data)

        output = model(perturbed_data_normalized)

        final_pred = torch.argmax(F.softmax(output, dim=1),dim=1)

        if final_pred.item() == target.item():
            correct += 1

        if epsilon == 0 and len(adv_examples) < 3:
            perturbed_data = perturbed_data.permute(0,2,3,1)
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            data1 = data
            data1 = torch.clamp(data_denorm,0,1)
            # data1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data1)
            data1 = data1.permute(0,2,3,1)
            orig = data1.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex,orig))
        else:
            if len(adv_examples) < 3 and final_pred.item() != target.item():
                perturbed_data = perturbed_data.permute(0,2,3,1)
                data1 = data
                data1 = torch.clamp(data_denorm,0,1)
                # data1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data1)
                data1 = data1.permute(0,2,3,1)
                orig = data1.squeeze().detach().cpu().numpy()
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex,orig))

    final_acc = correct/float(len(val_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(val_loader)} = {final_acc}")

    return final_acc, adv_examples


def plot_fgsm():
    epsilons = [0,0.001,0.002,0.003,0.004,0.005,0.1,0.15,0.2]
    # epsilons = [0.005]
    accs = []
    examples = []
    for eps in epsilons:
        print(f"Epsilon : {eps}")
        acc,ex = test_fgsm_attack(model, device, val_loader, eps)
        print(f"For epsilon : {eps}, accuracy : {acc}")
        accs.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5,5))
    plt.plot(epsilons,accs)
    plt.title("Accuracy vs Epsilon")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.show()
    return examples, epsilons

def plot_adverseries(examples,epsilons):
    cnt = 0
    plt.figure(figsize=(20,20))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), 2*len(examples[0]), cnt)
            plt.xticks([],[])
            plt.yticks([],[])
            if j == 0:
                plt.ylabel(f"Eps : {epsilons[i]}", fontsize=14)
            orig,adv,ex, orimage = examples[i][j]
            plt.title(f"Original : {idx_to_class[orig]}")
            plt.imshow(orimage,cmap="gray")
            cnt+=1
            plt.subplot(len(epsilons), 2*len(examples[0]), cnt)
            plt.title(f"Changed : {idx_to_class[adv]}")
            plt.imshow(ex,cmap="gray")
    plt.tight_layout()
    plt.show()

def Clip(X_new, X, epsilon):
    X_new = torch.min(torch.min(torch.tensor(255),X + epsilon), torch.max(torch.tensor(0), torch.max(X-epsilon,X_new)))
    # print(X_new,X)
    return X_new

def BIM(image,epsilon,alpha,loss_fn,y_true, epochs):
    perturbed_image = image
    # print(epochs)
    for epoch in range(epochs):
        perturbed_image.retain_grad()
        output = model(perturbed_image)
        loss = loss_fn(output, y_true)
        
        loss.backward(retain_graph=True)
        # print(type(perturbed_image))
        data_grad = perturbed_image.grad.data
        # signed_data_grad = data_grad.sign()
        perturbed_image = perturbed_image + alpha*(data_grad.sign())
        perturbed_image = Clip(perturbed_image,image,epsilon)
    return perturbed_image

def test_BIM_attack(model, device, val_loader, epsilon, alpha, epochs):
    correct = 0
    adv_examples = []
    targets = []

    for data, target in tqdm(val_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = torch.argmax(F.softmax(output, dim=1),dim=1)
        target_class = torch.argmin(F.softmax(output, dim=1), dim=1)

        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss

        
        data_denorm = denorm(data)

        perturbed_data = BIM(data_denorm, epsilon, alpha, loss, target, epochs)

        perturbed_data_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_data)

        output = model(perturbed_data_normalized)

        final_pred = torch.argmax(F.softmax(output, dim=1),dim=1)
        

        if final_pred.item() == target.item():
            correct += 1

        if epsilon == 0 and len(adv_examples) < 3:
            perturbed_data = perturbed_data.permute(0,2,3,1)
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            data1 = data
            data1 = torch.clamp(data_denorm,0,1)
            # data1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data1)
            data1 = data1.permute(0,2,3,1)
            orig = data1.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex,orig))
            targets.append((target.item(), final_pred.item()))
        else:
            if len(adv_examples) < 3 and final_pred.item() != target.item() and final_pred.item() == target_class.item():
                perturbed_data = perturbed_data.permute(0,2,3,1)
                data1 = data
                data1 = torch.clamp(data_denorm,0,1)
                # data1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data1)
                data1 = data1.permute(0,2,3,1)
                orig = data1.squeeze().detach().cpu().numpy()
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex,orig))
                targets.append((target.item(), final_pred.item()))


    final_acc = correct/float(len(val_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(val_loader)} = {final_acc}")

    return final_acc, adv_examples,targets

def iterative_least_likey_class_attack(X, alpha, epsilon, epochs, loss_fn, y_target, model):
    perturbed_image = X
    
    for epoch in range(epochs):
        perturbed_image.retain_grad()
        output = model(perturbed_image)
        loss = loss_fn(output, y_target)
        
        loss.backward(retain_graph=True)
        # print(type(perturbed_image))
        data_grad = perturbed_image.grad.data
        signed_data_grad = data_grad.sign()
        perturbed_image = perturbed_image - alpha*signed_data_grad
        perturbed_image = Clip(perturbed_image, X, epsilon)

    return perturbed_image

def test_illca_attack(model, device, val_loader, epsilon, alpha, epochs):
    correct = 0
    adv_examples = []
    targets = []

    for data, target in tqdm(val_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = torch.argmax(F.softmax(output, dim=1),dim=1)

        if init_pred.item() != target.item():
            continue

        target_class = torch.argmin(F.softmax(output, dim=1), dim=1)

        data_denorm = denorm(data)

        loss = F.nll_loss

        perturbed_data = iterative_least_likey_class_attack(data_denorm,  alpha, epsilon, epochs, loss, target_class,model)

        perturbed_data_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_data)

        output = model(perturbed_data_normalized)

        final_pred = torch.argmax(F.softmax(output, dim=1),dim=1)

        if final_pred.item() == target.item():
            correct += 1

        if epsilon == 0 and len(adv_examples) < 3:
            perturbed_data = perturbed_data.permute(0,2,3,1)
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            data1 = data
            data1 = torch.clamp(data_denorm,0,1)
            # data1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data1)
            data1 = data1.permute(0,2,3,1)
            orig = data1.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex,orig))
            targets.append((target.item(), target_class.item(), final_pred.item()))
        else:
            if len(adv_examples) < 3 and final_pred.item() != target.item() and final_pred.item() == target_class.item():
                perturbed_data = perturbed_data.permute(0,2,3,1)
                data1 = data
                data1 = torch.clamp(data_denorm,0,1)
                # data1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data1)
                data1 = data1.permute(0,2,3,1)
                orig = data1.squeeze().detach().cpu().numpy()
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex,orig))
                targets.append((target.item(), target_class.item(), final_pred.item()))

    final_acc = correct/float(len(val_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(val_loader)} = {final_acc}")

    return final_acc, adv_examples, targets


def plot_BIM():
    epsilons = [0.1,0.15,0.2,0.25,0.30]
    # epsilons = [0.0]
    accs = []
    examples = []
    for eps in epsilons:
        print(f"Epsilon : {eps}")
        acc,ex,targets = test_BIM_attack(model, device, val_loader, eps, 1, 20)
        print(f"For epsilon : {eps}, accuracy : {acc}")
        accs.append(acc)
        examples.append(ex)
        print(f"For epsilon : {eps}, Accuracy : {acc}")
        for el in targets:
            print(f"Original image : {el[0]}, adversary : {el[1]}")

    # plt.figure(figsize=(5,5))
    # plt.plot(epsilons,accs)
    # plt.title("Accuracy vs Epsilon")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Epsilon")
    # plt.show()
    return examples, epsilons

def plot_illca():
    epsilons = [0.1,0.15,0.2]
    accs = []
    examples = []
    for eps in epsilons:
        print(f"Epsilon : {eps}")
        acc,ex,targets = test_illca_attack(model, device, val_loader, eps, 1, 20)
        print(f"For epsilon : {eps}, accuracy : {acc}")
        accs.append(acc)
        examples.append(ex)

        print(f"For epsilon : {eps}, Accuracy : {acc}")
        for el in targets:
            print(f"Original image : {el[0]}, least likely : {el[1]}, adversary : {el[2]}")

# examples, epsilons = plot_fgsm()
plot_BIM()
# plot_adverseries(examples,epsilons)