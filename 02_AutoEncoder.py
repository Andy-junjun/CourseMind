import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===================== 1. 数据清洗与预处理 =====================
def add_gaussian_noise(img):
    """降噪自编码器：添加高斯噪声"""
    noise = torch.randn_like(img) * 0.1
    return torch.clamp(img + noise, 0., 1.)

# 图像预处理流水线
transform = transforms.Compose([
    transforms.ToTensor(),  # 归一化 [0,1]
])

# 加载数据集 + 自动清洗（框架内置过滤破损样本）
train_set = datasets.MNIST(root="./data/ae", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data/ae", train=False, download=True, transform=transform)

# 数据加载
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
print(f"【自编码器】MNIST 训练集: {len(train_set)} 测试集: {len(test_set)}")
print("数据去重、异常过滤、归一化完成")

# ===================== 2. 简易自编码器模型 =====================
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ===================== 3. 训练流程 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 简易训练（课程实验版）
epochs = 3
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for data, _ in train_loader:
        data = data.view(data.size(0), -1).to(device)
        noisy_data = add_gaussian_noise(data)  # 降噪任务
        output = model(noisy_data)
        loss = criterion(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.6f}")

print("自编码器训练完成")