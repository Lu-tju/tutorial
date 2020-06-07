""" GAN生成mnist数字 """
# -----------------------------
# 导入模块
# import torch
# from torch import nn
from torchvision import datasets
from models import *

# -----------------------------
# 设置gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# -----------------------------
# 基本参数设置
# 随机生成种子尺寸,用随机噪声来生成图像
input_size = 100
# batch size
batch_size = 200

# -----------------------------
# 数据预处理
dataset = datasets.MNIST('data/', download=True)
data = dataset.data.reshape(-1, 1, 28, 28).float()
# 数据控制到-1到1之间
data = data / (255 / 2) - 1

# -----------------------------
# 载入模型
dmodel = DNet().to(device)
gmodel = GNet(input_size).to(device)

# -----------------------------
# 损失函数和优化器
loss_fun = nn.BCELoss()
# 使用BCE计算损失，如果用MSE的话很难收敛，而且计算也很慢
goptim = torch.optim.Adam(gmodel.parameters(), lr=0.0001)
doptim = torch.optim.Adam(dmodel.parameters(), lr=0.0001)

# -----------------------------
# 训练数据
dmodel.train()
gmodel.train()
li_gloss = []
li_dloss = []

# 目标结果
d_true = torch.ones(batch_size, 1).to(device)
d_fake = torch.zeros(batch_size, 1).to(device)
for epoch in range(30):
    for batch in range(0, 60000, batch_size):
        # mnist集取出的真数据
        batch_data = data[batch:batch + batch_size].to(device)      # [batch_size,1,28,28]
        # 随机生成的种子数据
        fake_data = torch.randn(batch_size, input_size).to(device)  # [batch_size,100]
        # 先用判别器判别真数据
        output_dtrue = dmodel(batch_data)
        # 真数据的判别结果和1越近越好
        loss_dtrue = loss_fun(output_dtrue, d_true)
        # 再用判别器来判别通过种子数据生成的图片
        output_dfake = dmodel(gmodel(fake_data))
        # 对于判别器来说，假数据生成的图片的判别结果和0越近越好
        loss_dfake = loss_fun(output_dfake, d_fake)
        # 两者的loss都是越小越好
        loss_d = loss_dtrue + loss_dfake

        # 更新判别器参数
        doptim.zero_grad()
        loss_d.backward()
        doptim.step()
        li_dloss.append(loss_d)

        # 训练生成器
        for i in range(3):
            # 因为生成器更难训练，所以每训练一次判别器就训练3次生成器
            # fake_data = torch.randn(batch_size, input_size).to(device)
            # 判别通过假数据生成的图片
            output_gtrue = dmodel(gmodel(fake_data))
            # 对于生成器来说，生成的图片的判别结果越接近1越好，也就是越接近原图越好
            loss_g = loss_fun(output_gtrue, d_true)

            # 更新生成器参数
            doptim.zero_grad()
            goptim.zero_grad()
            loss_g.backward()
            goptim.step()
            li_gloss.append(loss_g)

        print("epoch:{}, batch:{}, loss_d:{}, loss_g:{}".format(epoch, batch, loss_d, loss_g))
    # 每epoch保存一次
    torch.save(dmodel.state_dict(), "gan_dmodel.mdl")
    torch.save(gmodel.state_dict(), "gan_gmodel.mdl")

