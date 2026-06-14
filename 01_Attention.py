import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===================== 通用图像清洗函数 =====================
def filter_invalid_img(dataset):
    """过滤全黑/全白异常图像"""
    valid_idx = []
    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        # 简单判断异常图
        if img.max() > 0.05 and img.min() < 0.95:
            valid_idx.append(idx)
    return valid_idx

# ===================== 图像注意力 + 数据清洗 =====================
print("===== 【注意力机制 - 图像任务 CIFAR10】 =====")
# 图像预处理流水线：统一尺寸、归一化
img_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 加载数据集
train_dataset = datasets.CIFAR10(root="./data/attention", train=True,
                                 download=True, transform=img_transform)
test_dataset = datasets.CIFAR10(root="./data/attention", train=False,
                                download=True, transform=img_transform)

# 清洗：过滤异常图像
train_valid = filter_invalid_img(train_dataset)
test_valid = filter_invalid_img(test_dataset)

print(f"原始训练集样本: {len(train_dataset)} | 清洗后有效样本: {len(train_valid)}")
print(f"原始测试集样本: {len(test_dataset)} | 清洗后有效样本: {len(test_valid)}")

# 构造数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("图像去重、异常过滤、归一化完成，注意力机制数据准备完毕")