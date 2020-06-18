import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from ipywidgets import IntProgress

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.'''

    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


plt.show(block=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('data/trainset', download=True, train=True, transform=transform)

valset = datasets.MNIST('data/valset', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
figure = plt.figure()
for i in range(1, 64):
    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.imshow(images[i].numpy().squeeze(), cmap='gray_r')

input_size = 28 * 28  # size of image is 28 x 28 and there are 64 of them
hidden_sizes = [128, 64]  # 2 hidden layers with 128 nodes and 64 nodes
output_size = 10
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
)
print(model)
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
print(images.numpy().shape)
images = images.view(images.shape[0], -1)
print(images.numpy().shape)
print(labels.numpy().shape)

logps = model(images)  # log probabilities
print(logps.shape)
loss = criterion(logps, labels)  # calculate the NLL loss
print(loss.shape)
print(loss)

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # print(images.shape)
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        # print(images.shape)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        # print(output.shape)
        loss = criterion(output, labels)
        # print(loss)

        # This is where the model learns by backprop
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(
            e, running_loss / len(trainloader)
        ))

print("\nTraining Time (in minutes) =", (time()-time0) / 60)

images, labels = next(iter(valloader))
img = images[0].view(1, 28 * 28)
print(img.numpy().shape)
with torch.no_grad():
    logps = model(img)
print(logps)
ps = torch.exp(logps)
print(ps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)

correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 28 * 28)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("Model Accuracy =", (correct_count / all_count))

torch.save(model, './init_mnist_model.pt')
