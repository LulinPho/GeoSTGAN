import torch

# 构造3*N的张量，顺序为[目标值, 上限值, 下限值]，N=6
constraint = torch.tensor([
    [7048911.40,      0,        0,   1200000,     2300000,        0],      # 目标值
    [      0,         0,        0,   1250000,     2375932,        0],      # 上限值
    [6298243.60,      0,        0,   1180000,     2250000,        0],       # 下限值
], dtype=torch.float32)

torch.save(constraint, "constraint/2035.pth")
print("已保存constraint.pth，内容如下：")
print(constraint)


# 将矩阵转换为torch张量并保存到文件
import torch

# 定义矩阵数据
matrix = [
    [0.082,   -0.346,  0.051,  -1.290,  6.891,  0.159],   # 耕地
    [-0.437,  -0.516, -0.341,  -0.125, -1.626,  0.000],   # 林地
    [0.032,   -0.306,  0.007,  -0.110,  8.854,  0.209],   # 草地
    [-2.126,  -0.117, -0.018,  -0.956,  0.918,  0.016],   # 水域
    [1.320,    2.067,  4.097,  -1.351, 12.679,  2.618],   # 建设用地
    [0.053,    0.005,  0.368,   0.116,  3.080,  0.194],   # 未利用地
]

# 转换为torch张量
matrix_tensor = torch.tensor(matrix, dtype=torch.float32)

# 保存到文件
torch.save(matrix_tensor, "constraint/transfer_matrix.pth")

print("已保存transfer_matrix.pth，内容如下：")
print(matrix_tensor)
