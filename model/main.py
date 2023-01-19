from utils import data_load, draw
from config import basic_config, get_acc, save_config, save_param, save_acc, info
from model import BiLSTM, save_model
from trainer import train, val

from torchtext.legacy.data import BucketIterator
from torch import nn, optim

epoch_list = []  # 迭代列表
train_loss_list = []  # 训练损失列表
train_acc_list = []  # 训练准确率列表
val_loss_list = []  # 验证损失列表
val_acc_list = []  # 验证准确率列表

if __name__ == '__main__':
    # 加载程序配置
    config = basic_config
    # 打印训练设备
    print('Training device:', config.device)
    # 保存基础配置
    save_config('info/config_file')

    # 数据加载
    train_data, val_data, TEXT, LABEL = data_load('clean_train.csv', 'clean_val.csv')

    # 获取模型训练参数
    vocab_size = TEXT.vocab.vectors.shape[0]
    embedding_dim = TEXT.vocab.vectors.shape[1]
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    vocab_to_id = dict(TEXT.vocab.stoi)
    label_to_id = dict(LABEL.vocab.stoi)
    # 保存模型训练参数
    save_param(vocab_size, embedding_dim, pad_idx, unk_idx, vocab_to_id, label_to_id, 'info/param_file')

    # 数据封装
    train_iter = BucketIterator(train_data,
                                batch_size=config.batch_size,
                                sort_key=lambda x: len(x.review),
                                sort_within_batch=True,
                                shuffle=True,
                                device=config.device)
    val_iter = BucketIterator(val_data,
                              batch_size=config.batch_size,
                              sort_key=lambda x: len(x.review),
                              sort_within_batch=True,
                              shuffle=False,
                              device=config.device)

    # 初始化模型
    model = BiLSTM(vocab_size=vocab_size,
                   embedding_dim=embedding_dim,
                   num_class=2,
                   hidden_size=config.hidden_size,
                   num_layers=config.num_layers,
                   dropout_rate=config.dropout_rate,
                   pad_idx=pad_idx,
                   unk_idx=unk_idx,
                   pre_trained=TEXT.vocab.vectors)
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # # 将模型转移到相应设备上
    # device = device(config.device)
    # model.to(device)

    # 获取历史最高准确率
    max_acc = get_acc('info/acc_file')
    # 模型训练
    for epoch in range(0, config.N_EPOCH):
        epoch_list.append(epoch + 1)
        # 训练模型获得每轮迭代的平均损失及平均准确率
        avg_loss_train, avg_acc_train = train(train_iter, model, loss_func, optimizer)

        train_loss_list.append(avg_loss_train)
        train_acc_list.append(avg_acc_train)

        # 获得验证集上的平均损失及平均准确率
        avg_loss_val, avg_acc_val = val(val_iter, model, loss_func)
        # print("Epoch: [{}/{}], avg_loss = {:.2f}%, avg_acc = {:.2f}%"
        #       .format(epoch + 1, basic_config.N_EPOCH, avg_loss_val * 100, avg_acc_val * 100))

        val_loss_list.append(avg_loss_val)
        val_acc_list.append(avg_acc_val)

        # 保存最优模型
        if avg_acc_val > max_acc:
            max_acc = avg_acc_val
            # 更新历史最高准确率
            save_acc(max_acc, 'info/acc_file')
            # 保存当前模型
            save_model(model.state_dict(), 'info', 'best_model.pth')

    # 训练结束
    print('Training Completed')

    # 保存训练信息
    info(epoch_list, train_loss_list, train_acc_list, 'info/train_info_file')
    info(epoch_list, val_loss_list, val_acc_list, 'info/val_info_file')

    # # 绘制训练曲线
    # draw('info/train_info_file')
