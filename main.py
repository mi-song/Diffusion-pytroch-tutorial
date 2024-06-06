import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the UNet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # First encoder block
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Second encoder block
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Third encoder block
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # First decoder block
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        # Second decoder block
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        # Final layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        d1 = self.decoder1(e3)
        d1 = torch.cat((d1, e2), dim=1)  # skip connection

        d2 = self.decoder2(d1)
        d2 = torch.cat((d2, e1), dim=1)  # Skip connection

        out = self.final_layer(d2)
        return out

# Forward Process : adds noise to the input image over time steps
def forward_diffusion(x0, t, betas):
    batch_size = x0.size(0)
    sqrt_alphas_cumprod = torch.cumprod(1 - betas, dim=0) ** 0.5  # Cumulative product of (1 - beta) values
    sqrt_one_minus_alphas_cumprod = (1 - sqrt_alphas_cumprod**2) ** 0.5  # Complement of the cumulative product
    noise = torch.randn_like(x0)  # Generate random noise

    # Select the appropriate cumulative product values for the current time step
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)

    # Return the noisy image and the noise itself
    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# Compute the loss for training: measures how well the model can predict the added noise
def compute_loss(model, x0, t, betas):
    xt, noise = forward_diffusion(x0, t, betas)  # Perform forward diffusion to get noisy image
    pred_noise = model(xt)  # Predict the noise added to the image
    return nn.MSELoss()(pred_noise, noise)  # Compute mean squared error between predicted and actual noise

# Reverse process: removes noise from the noisy image over time steps
def reverse_process(model, xt, t, betas):
    sqrt_recip_alphas_cumprod = (1 / torch.cumprod(1 - betas, dim=0)) ** 0.5  # Reciprocal cumulative product
    sqrt_recipm1_alphas_cumprod = (1 / torch.cumprod(1 - betas, dim=0) - 1) ** 0.5  # Complement of the reciprocal cumulative product
    pred_noise = model(xt)  # Predict the noise in the image
    # Remove the predicted noise and scale by the reciprocal cumulative product
    xt = (xt - sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * pred_noise) / sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
    return xt

def train(model, dataloader, betas, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for x0, _ in dataloader:
            x0 = x0.to('cuda')
            t = torch.randint(0, len(betas), (x0.size(0),), device=x0.device).long()
            loss = compute_loss(model, x0, t, betas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Sample images from the model: generates new images by reversing the diffusion process
def sample(model, betas, num_samples=64):
    model.eval()
    xt = torch.randn((num_samples, 1, 28, 28), device='cuda')  # Initialize with random noise
    with torch.no_grad():
        for t in reversed(range(len(betas))):
            xt = reverse_process(model, xt, t, betas)  # Remove noise at each step
    return xt


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Beta schedule for the diffusion process: linearly spaced values between 0.0001 and 0.02
betas = torch.linspace(0.0001, 0.02, 1000).to('cuda')

model = UNet().to('cuda')
train(model, train_loader, betas, epochs=10)

# Generate samples using the trained model
samples = sample(model, betas)
samples = samples.cpu().numpy()

# Visualize generated samples
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i, 0], cmap='gray')
    ax.axis('off')
plt.show()
