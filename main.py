import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from dataset.mnist.load_data import MyDataSet
from layer.CNN import CNN
from layer.AutoEncoder import AutoEncoder
from layer.Classifier import Classifier
from utils.utils import show_image


# 平均值维护器
class Average:
    def __init__(self):
        super().__init__()
        self.total_val = 0
        self.count = 0

    def update(self, cur_val, count=1):
        self.total_val += cur_val * count
        self.count += count

    def get(self):
        """
        :return: 平均值
        """
        return self.total_val / (self.count + 1e-10)


# def accuracy(out, label):
#     # count = out.shape[0]
#     # true_count = 0
#     # for out_val, label_val in zip(out, label):
#     #
#     #     if torch.argmax(out_val) == torch.argmax(label_val):
#     #         true_count += 1
#     # return true_count * 100 / count
#     return sum((label.argmax(axis=1) == out.argmax(axis=1)) + 0.0) * 100.0 / out.shape[0]

def accuracy(out, label):
    # 计算多分类准确率
    return sum(label == out.argmax(axis=1)) * 100.0 / out.shape[0]


def train_classifier(epoch, train_loader, model, classifier, criterion, optimizer):
    loss_ = Average()
    acc_ = Average()
    for i, sample in enumerate(train_loader):
        image = sample['image']
        label = sample['label']
        if torch.cuda.is_available():
            image = image.float().cuda()
            label = label.long().cuda()
        # 前向传播
        # 截断编码器梯度
        with torch.no_grad():
            model_out, embed = model(image)

        # 对embedding进行分类
        out = classifier(embed)

        # 计算损失函数
        loss = criterion(out, label)

        # 前一次的梯度置零(Adma会自己保留历史梯度，这一步不会把历史梯度置零)
        optimizer.zero_grad()
        # 损失函数反向传播
        loss.backward()
        # 优化梯度
        optimizer.step()

        # 计算准确率
        acc = accuracy(out, label)
        # 注意加item() 不然会将 loss_的计算当成计算图的一部分
        loss_.update(loss.item(), image.shape[0])
        acc_.update(acc.item(), image.shape[0])

        if i % 100 == 0:
            print("[Train Classifier] Epoch: %d, Iterate:%d, Loss:%.6f(%.6f) Accuracy:%.6f(%.6f)"
                  % (epoch, i, loss.item(), loss_.get(), acc, acc_.get()))
    return acc_.get()


def val_classifier(epoch, val_loader, model, classifier, criterion, optimizer):
    loss_ = Average()
    acc_ = Average()
    for i, sample in enumerate(val_loader):
        image = sample['image']
        label = sample['label']
        if torch.cuda.is_available():
            image = image.float().cuda()
            label = label.long().cuda()
        # 前向传播
        # 截断编码器梯度
        with torch.no_grad():
            model_out, embed = model(image)

        # 对embedding进行分类
        out = classifier(embed)

        # visual = True
        # if visual:
        #     show_image(model_out.view(-1, 1, 28, 28), out, label)
        #     show_image(image.view(-1, 1, 28, 28), out, label)
        #     break

        # 计算损失函数
        loss = criterion(out, label)

        # 计算准确率
        acc = accuracy(out, label)
        # 注意加item() 不然会将 loss_的计算当成计算图的一部分
        loss_.update(loss.item(), image.shape[0])
        acc_.update(acc.item(), image.shape[0])

        if i % 100 == 0:
            print("[Val Classifier] Epoch: %d, Iterate:%d, Loss:%.6f(%.6f) Accuracy:%.6f(%.6f)"
                  % (epoch, i, loss.item(), loss_.get(), acc, acc_.get()))
    print("[Val Classifier] Epoch: %d, Loss:%.6f Accuracy:%.6f" % (epoch, loss_.get(), acc_.get()))
    return acc_.get()


def train(epoch, train_loader, model, criterion, optimizer):
    loss_ = Average()
    # acc_ = Average()
    for i, sample in enumerate(train_loader):
        image = sample['image']
        label = sample['label']
        if torch.cuda.is_available():
            image = image.float().cuda()
            label = label.long().cuda()
        # 前向传播
        out, embed = model(image)
        # 计算损失函数
        loss = criterion(out, image)
        # 前一次的梯度置零(Adma会自己保留历史梯度，这一步不会把历史梯度置零)
        optimizer.zero_grad()
        # 损失函数反向传播
        loss.backward()
        # 优化梯度
        optimizer.step()

        # 计算准确率
        # acc = accuracy(embed, label)
        # 注意加item() 不然会将 loss_的计算当成计算图的一部分
        loss_.update(loss.item(), image.shape[0])
        # acc_.update(acc.item(), image.shape[0])

        if i % 100 == 0:
            # print("[Train] Epoch: %d, Iterate:%d, Loss:%.6f(%.6f) Accuracy:%.6f(%.6f)"
            #      % (epoch, i, loss, loss_.get(), acc, acc_.get()))
            print("[Train] Epoch: %d, Iterate:%d, Loss:%.6f(%.6f) "
                  % (epoch, i, loss.item(), loss_.get()))
    return loss_.get()


def val(epoch, val_loader, model, criterion, optimizer):
    loss_ = Average()
    # acc_ = Average()
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['image']
            label = sample['label']
            if torch.cuda.is_available():
                image = image.float().cuda()
                label = label.long().cuda()
            out, embed = model(image)
            loss = criterion(out, image)
            # acc = accuracy(embed, label)

            # # 正常验证的时候是不进行梯度方向传播 自编码器自监督训练的时候可以将验证集也作为训练数据
            # # 前一次的梯度置零(Adma会自己保留历史梯度，这一步不会把历史梯度置零)
            # optimizer.zero_grad()
            # # 损失函数反向传播
            # loss.backward()
            # # 优化梯度
            # optimizer.step()

            # 注意加item() 不然会将 loss_的计算当成计算图的一部分
            loss_.update(loss.item(), image.shape[0])
            # acc_.update(acc.item(), image.shape[0])

            if i % 100 == 0:
                # print("[Val] Epoch: %d, Iterate:%d, Loss:%.6f(%.6f) Accuracy:%.6f(%.6f)"
                #       % (epoch, i, loss, loss_.get(), acc, acc_.get()))
                print("[Val] Epoch: %d, Iterate:%d, Loss:%.6f(%.6f) "
                      % (epoch, i, loss.item(), loss_.get()))
    # print("[Val] Epoch: %d, Loss:%.6f Accuracy:%.6f" % (epoch, loss_.get(), acc_.get()))
    print("[Val] Epoch: %d, Loss:%.6f " % (epoch, loss_.get()))
    return loss_.get()


def main():
    # 训练数据
    train_image_path = 'D:/DataSet/mnist/train-images.idx3-ubyte'
    train_label_path = 'D:/DataSet/mnist/train-labels.idx1-ubyte'
    val_image_path = 'D:/DataSet/mnist/t10k-images.idx3-ubyte'
    val_label_path = 'D:/DataSet/mnist/t10k-labels.idx1-ubyte'

    # 保存checkpoint的步长
    save_step = 1
    checkpoint_save_path = 'output/model_checkpoint.pth'
    classifier_checkpoint_save_path = 'output/classifier_checkpoint.pth'

    # 保存最好模型的模型参数
    model_save_path = 'output/model_best.pth'
    classifier_model_save_path = 'output/classifier_best.pth'

    # 日志文件
    log_save_path = 'log/log.txt'
    batch_size = 256

    # 打开日志文件
    log = open(log_save_path, 'a')

    # 是否训练编码器 不训练的时候执行测试
    train_model = False
    # 是否训练分类器 不训练的时候执行测试
    train_classifier_model = False

    # 是否加载已训练模型
    resume_model = True
    # 是否加载已训练模型
    resume_classifier_model = True

    # 模型的加载
    # model = CNN(28, 28)

    num_embedding = 32
    model = AutoEncoder(num_in=784, num_out=784, num_embeddings=num_embedding, rate=1, dropout=0.1)
    classifier_model = Classifier(num_in=num_embedding, num_out=10, rate=1, dropout=0.1)

    print(model)
    print(classifier_model)
    # 数据集加载部分
    train_dataset = MyDataSet(train_image_path, train_label_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataset = MyDataSet(val_image_path, val_label_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    # 重构损失函数的选择
    criterion = nn.MSELoss()

    # 多分类损失函数的选择
    classifier_criterion = nn.CrossEntropyLoss()

    # 将模型转移到GPU中
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        classifier_model = classifier_model.cuda()
        classifier_criterion = classifier_criterion.cuda()

    # 优化器的选择
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 分类优化器的选择
    classifier_optimizer = optim.Adam(classifier_model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 加载模型和训练参数
    last_model_epoch = 0
    last_classifier_epoch = 0
    if train_model and resume_model and os.path.exists(checkpoint_save_path):
        state_dict = torch.load(checkpoint_save_path)
        last_model_epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
    elif not train_model and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    if train_classifier_model and resume_classifier_model and os.path.exists(classifier_checkpoint_save_path):
        state_dict = torch.load(classifier_checkpoint_save_path)
        last_classifier_epoch = state_dict['epoch']
        classifier_model.load_state_dict(state_dict['model'])
        classifier_optimizer.load_state_dict(state_dict['optimizer'])
    elif not train_classifier_model and os.path.exists(classifier_model_save_path):
        classifier_model.load_state_dict(torch.load(classifier_model_save_path))

    # 设置训练学习率的调整计划参数
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1,
                                                  last_epoch=last_model_epoch - 1)
    classifier_lr_scheduler = optim.lr_scheduler.MultiStepLR(classifier_optimizer, milestones=[10, 15], gamma=0.1,
                                                             last_epoch=last_classifier_epoch - 1)
    # 训练自编码器
    if train_model:
        epoch = 100
        best_loss = 10000.0
        for i in range(last_model_epoch, epoch):
            lr_scheduler.step()
            # 训练 验证
            train_loss = train(i, train_loader, model, criterion, optimizer)
            val_loss = val(i, val_loader, model, criterion, optimizer)

            # 写日志文件
            log.write("[Train] Epoch: %d, Loss:%.6f \n" % (i, train_loss))
            log.write("[Val] Epoch: %d, Loss:%.6f \n" % (i, val_loss))
            # 实时显示
            log.flush()

            if i % save_step == 0:
                torch.save({'epoch': i + 1,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), },
                           checkpoint_save_path)

            # 保存最优模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
    else:
        model.eval()
        val_loss = val(0, val_loader, model, criterion, optimizer)
        print("Test Loss: %.6f" % val_loss)

    # 加载训练好的最好的自编码器参数
    if os.path.exists(model_save_path):
        print("Load the best embedding model")
        model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # 训练分类器
    if train_classifier_model:
        best_acc = 0.0
        classifier_epoch = 20
        for i in range(last_classifier_epoch, classifier_epoch):
            classifier_lr_scheduler.step()
            # 训练 验证
            train_acc = train_classifier(i, train_loader, model, classifier_model, classifier_criterion,
                                         classifier_optimizer)
            val_acc = val_classifier(i, val_loader, model, classifier_model, classifier_criterion, classifier_optimizer)

            # 写日志文件
            log.write("[Train Classifier] Epoch: %d, Accuracy:%.6f \n" % (i, train_acc))
            log.write("[Val Classifier] Epoch: %d, Accuracy:%.6f \n" % (i, val_acc))
            # 实时显示
            log.flush()

            # 保存checkpoint便于继续训练
            if i % save_step == 0:
                torch.save({'epoch': i + 1,
                            'model': classifier_model.state_dict(),
                            'optimizer': classifier_optimizer.state_dict(), },
                           classifier_checkpoint_save_path)
            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(classifier_model.state_dict(), classifier_model_save_path)

    else:
        classifier_model.eval()
        val_acc = val_classifier(0, val_loader, model, classifier_model, classifier_criterion, classifier_optimizer)
        print("Test Accuracy: %.6f" % val_acc)

    log.close()


if __name__ == '__main__':
    main()
