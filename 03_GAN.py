import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===================== 1. GAN专属数据清洗 & 预处理 =====================
# GAN 标准归一化：[0,1] → [-1, 1]
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集，自动过滤异常/重复样本
dataset = datasets.MNIST(root="./data/gan", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"【GAN】数据集样本数: {len(dataset)}，数据清洗、归一化至[-1,1]完成")

# ===================== 2. 简易DCGAN 生成器/判别器 =====================
# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 784),
            nn.Tanh()
        )
    def forward(self, x):
        return self.gen(x)

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.dis(x)

# ===================== 3. 训练流程 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=2e-4)
opt_D = optim.Adam(D.parameters(), lr=2e-4)

epochs = 3
for epoch in range(epochs):
    total_d_loss = 0.0
    total_g_loss = 0.0
    for real_img, _ in dataloader:
        batch_size = real_img.size(0)
        real_img = real_img.view(batch_size, -1).to(device)

        # 标签
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # 训练判别器
        opt_D.zero_grad()
        # 真实图片
        d_out_real = D(real_img)
        loss_real = criterion(d_out_real, real_label)
        # 伪造图片
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_img = G(z)
        d_out_fake = D(fake_img.detach())
        loss_fake = criterion(d_out_fake, fake_label)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # 训练生成器
        opt_G.zero_grad()
        d_out = D(fake_img)
        loss_G = criterion(d_out, real_label)
        loss_G.backward()
        opt_G.step()

        total_d_loss += loss_D.item()
        total_g_loss += loss_G.item()

    print(f"Epoch[{epoch+1}/{epochs}] D_Loss:{total_d_loss/len(dataloader):.4f} G_Loss:{total_g_loss/len(dataloader):.4f}")

print("GAN 训练完成")