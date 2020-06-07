import matplotlib.pyplot as plt
from models import *

# -----------------------------
# 设置gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# -----------------------------
# 载入模型
input_size = 100
gmodel = GNet(input_size).to(device)
gmodel.load_state_dict(torch.load("gan_gmodel.mdl"))
# dmodel.load_state_dict(torch.load("gan_dmodel.mdl"))

# -----------------------------
# 测试数据
gmodel.eval()
# 随机生成20个种子数据
data_test = torch.randn(20, input_size)
result = gmodel(data_test.to(device))
plt.figure(figsize=(10, 50))
for i in range(len(result)):
    ax = plt.subplot(len(result) / 5, 5, i + 1)
    plt.imshow((result[i].cpu().data.reshape(28, 28) + 1) * 255 / 2)
    plt.axis('off')
    plt.gray()
plt.show()
