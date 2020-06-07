import torch
from torch import nn

# -----------------------------
# 定义网络
class DNet(nn.Module):
    # 判别器，识别图片，并返回正确率，越真实的图片正确率尽量接近1，否则接近0
    # input:(batch_size, 1, 28, 28)
    # output:(batch_size, 1)
    def __init__(self):
        super(DNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(128)  # , momentum=0.9)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)  # , momentum=0.9)
        self.batch_norm3 = torch.nn.BatchNorm2d(512)  # , momentum=0.9)
        self.leakyrelu = nn.LeakyReLU()
        self.linear = nn.Linear(8192, 1)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batch_norm1(self.conv2(x)))
        x = self.leakyrelu(self.batch_norm2(self.conv3(x)))
        x = self.leakyrelu(self.batch_norm3(self.conv4(x)))
        x = torch.flatten(x).reshape(-1, 8192)
        x = torch.sigmoid(self.linear(x))
        return x


class GNet(nn.Module):
    # 生成器，输入随机噪声（种子），生成尽可能真实的图片
    # input:(batch_size, seed_size)
    # output:(batch_size, 1, 28, 28)
    def __init__(self, input_size):
        super(GNet, self).__init__()
        self.d = 3
        self.linear = nn.Linear(input_size, self.d * self.d * 512)
        self.conv_tranpose1 = nn.ConvTranspose2d(512, 256, 5, 2, 1)
        self.conv_tranpose2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv_tranpose3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv_tranpose4 = nn.ConvTranspose2d(64, 1, 3, 1, 1)
        self.batch_norm1 = torch.nn.BatchNorm2d(512)     # , momentum=0.9)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)     # , momentum=0.9)
        self.batch_norm3 = torch.nn.BatchNorm2d(128)     # , momentum=0.9)
        self.batch_norm4 = torch.nn.BatchNorm2d(64)      # , momentum=0.9)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):                                       # x:[batch_size,100]
        x = self.linear(x).reshape(-1, 512, self.d, self.d)     # [batch_size,512,3,3]
        x = self.relu(self.batch_norm1(x))
        x = self.conv_tranpose1(x)
        x = self.relu(self.batch_norm2(x))
        x = self.conv_tranpose2(x)
        x = self.relu(self.batch_norm3(x))
        x = self.conv_tranpose3(x)
        x = self.relu(self.batch_norm4(x))
        x = self.tanh(self.conv_tranpose4(x))
        return x