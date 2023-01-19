"""
模型训练器
"""
import torch


def train(iterator, model, loss_func, optimizer):
    """
    模型训练
    :param iterator: 迭代器
    :param model: 模型
    :param loss_func: 损失函数
    :param optimizer: 优化器
    :return: avg_loss, avg_acc
    """
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    step = 0
    for batch_idx, data in enumerate(iterator):
        inputs, length = data.review
        labels = data.label.squeeze(0)

        # 模型预测
        predicts = model.forward(inputs, length)

        # 计算损失及准确率
        loss = loss_func(predicts, labels)
        acc = (predicts.argmax(dim=1) == labels).sum().item() / len(labels)

        total_loss += loss.item()
        total_acc += acc
        step += 1

        # 反向传播及更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / step
    avg_acc = total_acc / step

    return avg_loss, avg_acc


def val(iterator, model, loss_func):
    """
    模型验证
    :param iterator: 迭代器
    :param model: 模型
    :param loss_func: 损失函数
    :return: avg_loss, avg_acc
    """
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    step = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(iterator):
            inputs, length = data.review
            labels = data.label.squeeze(0)

            # 模型预测
            predicts = model.forward(inputs, length)

            # 计算损失及准确率
            loss = loss_func(predicts, labels)
            acc = (predicts.argmax(dim=1) == labels).sum().item() / len(labels)

            total_loss += loss.item()
            total_acc += acc
            step += 1

    avg_loss = total_loss / step
    avg_acc = total_acc / step

    return avg_loss, avg_acc
