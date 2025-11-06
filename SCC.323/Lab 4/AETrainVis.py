# Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from AEClass import Autoencoder

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)


# Load dataset
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)

# Train the autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

# change code to use Adam optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data.view(-1, 28*28))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Visualise the reconstructions
dataiter = iter(test_loader)
images, _ = next(dataiter)
images = images.to(device)
output = model(images)
output = output.view(-1, 1, 28, 28).cpu().detach().numpy()

# Plot
fig, axes = plt.subplots(2, 10, figsize=(10,2))
for i in range(10):
    axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(output[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('Reconstructed')
plt.show()


latent_vectors = []
labels = []

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        encoded = model.encoder(data.view(-1, 28*28))
        latent_vectors.append(encoded.cpu())
        labels.append(target)
latent_vectors = torch.cat(latent_vectors)
labels = torch.cat(labels)

plt.figure(figsize=(6,6))
plt.scatter(latent_vectors[:,0], latent_vectors[:,1], c=labels, cmap='tab10', s=10)
plt.colorbar()
plt.title("Latent Space Representation of MNIST Digits")
plt.show()

# use trained autoencoder to make comparisons of samples of the testing images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
images, labels = next(iter(test_loader))
img1 = images[0].unsqueeze(0).to(device)
img2 = images[1].unsqueeze(0).to(device)

img1_cpu = img1.squeeze().cpu().numpy().reshape(28, 28)
img2_cpu = img2.squeeze().cpu().numpy().reshape(28, 28)

fig, axes = plt.subplots(1, 2, figsize=(4, 2))
axes[0].imshow(img1_cpu, cmap='gray')
axes[0].set_title(f"Label: {labels[0].item()}")
axes[1].imshow(img2_cpu, cmap='gray')
axes[1].set_title(f"Label: {labels[1].item()}")
plt.show()

with torch.no_grad():
    feat1 = model.encoder(img1.view(-1, 28*28))
    feat2 = model.encoder(img2.view(-1, 28*28))

euclidean_distance = torch.norm(feat1 - feat2).item()
cosine_similarity = F.cosine_similarity(feat1, feat2).item()

print(f"Euclidean distance: {euclidean_distance:.4f}")
print(f"Cosine similarity:  {cosine_similarity:.4f}")