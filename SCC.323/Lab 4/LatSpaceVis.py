# Visualise the latent space
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