import torch.nn as nn


class SimlpleClassifier(nn.Module):
    def __init__(self, num_in, num_out):
        """
        最简单的分类器
        测试编码器性能
        :param num_in: 输入维度
        :param num_out: 输出维度
        """
        super().__init__()
        self.linears = []
        self.linears.append(nn.Linear(num_in, num_out))
        # self.linears.append(nn.Sigmoid())
        self.linears = nn.ModuleList(self.linears)
        self.linears = nn.Sequential(*self.linears)

    def forward(self, x):
        return self.linears(x)


class Classifier(nn.Module):
    def __init__(self, num_in, num_out, rate=1, dropout=0.1):
        """
        全连接分类器
        测试编码器性能
        :param num_in: 输入维度
        :param num_out: 输出维度
        :param rate: 扩展维度（即线性层的扩展比）
        :param dropout: dropout的百分比
        """
        super().__init__()
        temp = num_in
        self.act = nn.ReLU(inplace=True)
        self.linears = []
        while num_out < temp >> rate:
            temp_temp = temp
            temp >>= rate
            self.linears.append(nn.Linear(temp_temp, temp))
            self.linears.append(nn.Dropout(p=dropout))
            self.linears.append(nn.BatchNorm1d(temp))
            self.linears.append(self.act)
        self.linears.append(nn.Linear(temp, num_out))
        self.linears = nn.ModuleList(self.linears)
        self.linears = nn.Sequential(*self.linears)

    def forward(self, x):
        return self.linears(x)
