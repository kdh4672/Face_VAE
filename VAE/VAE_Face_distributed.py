import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image # Load 'save_image' Function
import time
start = time.time()
trans = transforms.Compose([
    transforms.Resize((240,180)),transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(root='kong_face',
                                        transform=trans,
                                        )

# ================================================================== #
#                        0. Define Hyper-parameters
# ================================================================== #
# Device Configuration for Where the Tensors Be Operated
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define OS Configuration
sample_dir = './results_face'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
image_size = 240*180
h_dim = 256
h_dim2 = 128
z_dim = 2
num_epochs = 10000
batch_size = 60
learning_rate = 0.001


# In[4]:


# ================================================================== #
#                        2. Define Dataloader
# ================================================================== #
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

# Doesn't Need Test Loader As Well


# In[5]:


# ================================================================== #
#                        3. Define Model
# ================================================================== #
class VAE(nn.Module):
    def __init__(self, image_size=image_size, h_dim=h_dim,h_dim2=h_dim2, z_dim=z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim) # from 784 Nodes(28x28 MNIST Image) to 400 Nodes (h_dim) 
        self.fc2 = nn.Linear(h_dim, h_dim2) # from 400 Nodes (h_dim) to 20 Nodes (Dims of mean of z)
        self.fc3 = nn.Linear(h_dim, h_dim2)# from 400 Nodes (h_dim) to 20 Nodes (Dims of std of z)
        self.fc4 = nn.Linear(h_dim2, z_dim)
        self.fc5 = nn.Linear(h_dim2, z_dim)
        self.fc6 = nn.Linear(z_dim, h_dim2) # from 20 Nodes (reparameterized z=mean+eps*std) to 400 Nodes (h_dim)
        self.fc7 = nn.Linear(h_dim2, h_dim)
        self.fc8 = nn.Linear(h_dim, image_size) # from 400 Nodes (h_dim) to 784 Nodes (Reconstructed 28x28 Image)
        
    # Encoder: Encode Image to Latent Vector z
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h_mu = F.relu(self.fc2(h))
        h_log_var = F.relu(self.fc3(h))
        return self.fc4(h_mu), self.fc5(h_log_var) # 784 -> 256
    
    # Reparameterize z=mean+std to z=mean+esp*std
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Decoder: Decode Reparameterized Latent Vector z to Reconstructed Image
    def decode(self, z):
        h = F.relu(self.fc6(z))
        h = F.relu(self.fc7(h))
        return torch.sigmoid(self.fc8(h))
    
    # Feed Forward the Process and Outputs Estimated (Mean, Std, Reconstructed_Image) at the same time
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

model = VAE().to(device)
model = torch.nn.DataParallel(model)
model.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Total Loss is going to be defined in Training Part as it is a combination of Reconstruction Loss and Regularization Loss



import numpy as np
from numpy import moveaxis

def plt_manifold(trained_model_instance, save_file_path, mean_range=3, n=10, figsize=(100, 100)):
    x1_axis = np.linspace(-mean_range, mean_range, n)
    y1_axis = np.linspace(-mean_range, mean_range, n)
    canvas = np.empty((3,240*n, 180*n))
    print("making_manifold...")
    for i, y1 in enumerate(x1_axis):
        for j, x1 in enumerate(y1_axis):
            z_mean = np.array([[[x1, y1]],[[x1, y1]],[[x1, y1]]] * 1)
            z_mean = torch.tensor(z_mean, device=device).float()
            x_reconst = VAE().to(device).decode(z_mean) ##model.decode (single_gpu에선)
            canvas[:,(n-i-1)*240:(n-i)*240, j*180:(j+1)*180] = x_reconst.view(3,240,180).cpu()
    canvas = moveaxis(canvas, 1, 0)
    canvas = moveaxis(canvas, 1, 2)
    plt.figure(figsize=figsize)
    xi, yi = np.meshgrid(x1_axis, y1_axis)
    plt.imshow(canvas, origin="upper")
    plt.savefig(save_file_path)
    print("making " + save_file_path + " is completed!")
    
    return


for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_loader): # '_' as we don't need label of the input Image
        # Feed Forward
        x = x.to(device).view(-1,3, image_size) # Flatten 2D Image into 1D Nodes
        x_reconst, mu, log_var = model(x)
        
        # Compute the Total Loss
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False) # See the Description below
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        
        # Get Loss, Compute Gradient, Update Parameters
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print Loss for Tracking Training
        if (i+1) % 1 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(train_loader), reconst_loss.item(), kl_div.item()))
            
    # Save Model on Last epoch
    if epoch+1 == num_epochs:
        torch.save(model.state_dict(), './model.pth')
    
    if epoch%100==0:
        save_file_path = "./face_manifold/face_manifold_{}.png".format(epoch)
        with torch.no_grad():
            plt_manifold(model, save_file_path)
        print("time :", time.time() - start)
    
    # Save Generated Image and Reconstructed Image at every Epoch
    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(batch_size,3,z_dim).to(device) # Randomly Sample z (Only Contains Mean)
        out = VAE().to(device).decode(z).view(-1,3, 240, 180) #sigle gpu에선 model로 써도됨
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 3, 240, 180), out.view(-1, 3, 240, 180)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))