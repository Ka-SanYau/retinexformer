import torch

# 创建两个 3x3 的张量，值为 0-9 的随机整数
tensor1 = torch.randint(0, 10, (3, 3))  # 第一个张量
tensor2 = torch.randint(0, 10, (3, 3))  # 第二个张量

# 打印张量
print("张量1：")
print(tensor1)
print("\n张量2：")
print(tensor2)
print("tensor * operatore: \n")
print(tensor1*tensor2)