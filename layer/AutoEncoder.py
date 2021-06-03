import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, num_in, num_embeddings, compress_rate=1, dropout=0.1):
        """
        编码器
        :param num_in: 输入维度
        :param num_embeddings: 输入维度
        :param compress_rate: 压缩率（即线性层的压缩比）
        :param dropout: dropout的百分比
        """
        super().__init__()
        temp = num_embeddings
        self.act = nn.ReLU(inplace=True)
        self.linears = []
        while num_in > temp << compress_rate:
            temp_temp = temp
            temp <<= compress_rate
            if temp_temp != num_embeddings:
                self.linears.append(self.act)
                self.linears.append(nn.BatchNorm1d(temp_temp))
                self.linears.append(nn.Linear(temp, temp_temp))
            else:
                self.linears.append(nn.Linear(temp, temp_temp))
            # self.linears.append(self.act)
            # self.linears.append(nn.BatchNorm1d(temp))
            # self.linears.append(nn.Linear(temp, temp))
        self.linears.append(nn.Dropout(p=dropout))
        self.linears.append(self.act)
        self.linears.append(nn.Linear(num_in, temp))
        self.linears.reverse()
        self.linears = nn.ModuleList(self.linears)
        self.linears = nn.Sequential(*self.linears)

    def forward(self, x):
        return self.linears(x)


class Decoder(nn.Module):

    def __init__(self, num_out, num_embeddings, extend_rate=1, dropout=0.1):
        """
        解码器
        :param num_out: 输出维度
        :param num_embeddings: 嵌入向量维度
        :param extend_rate: 扩展维度（即线性层的扩展比）
        :param dropout: dropout的百分比
        """
        super().__init__()
        temp = num_embeddings
        self.linears = []
        self.act = nn.ReLU(inplace=True)
        while num_out > temp << extend_rate:
            temp_temp = temp
            temp <<= extend_rate
            self.linears.append(nn.Linear(temp_temp, temp))
            self.linears.append(nn.BatchNorm1d(temp))
            self.linears.append(self.act)
            # self.linears.append(nn.Linear(temp, temp))
            # self.linears.append(nn.BatchNorm1d(temp))
            # self.linears.append(self.act)
        # self.linears.append(nn.Dropout(p=dropout))
        self.linears.append(nn.Linear(temp, num_out))
        self.linears.append(nn.Sigmoid())
        self.linears = nn.ModuleList(self.linears)

    def forward(self, x):
        for l in self.linears:
            x = l(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, num_in, num_out, num_embeddings, rate=1, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(num_in, num_embeddings, rate, dropout)
        self.decoder = Decoder(num_out, num_embeddings, rate, dropout)

    def forward(self, x):
        embed = self.encoder(x)
        return self.decoder(embed), embed
