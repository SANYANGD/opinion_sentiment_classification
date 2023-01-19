"""
程序配置表
"""
import json
import os

import torch
from collections import OrderedDict


class basic_config:
    # 运行设备名称
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 外部词向量
    embedding_loc = 'Embedding/cc.zh.300.vec'
    # 词向量缓存
    cache = '.vector_cache'
    # 最优模型
    best_model = 'info/best_model.pth'

    # batch大小
    batch_size = 64
    # 学习率
    lr = 0.01
    # dropout率
    dropout_rate = 0.5
    # 总迭代次数
    N_EPOCH = 5

    # 模型参数
    hidden_size = 100
    num_layers = 1


def save_param(vocab_size, embedding_dim, pad_idx, unk_idx, vocab_to_id, label_to_id, path):
    """
    保存模型训练参数
    :param vocab_size: 词典大小
    :param embedding_dim: embedding维度
    :param pad_idx: index of <pad>
    :param unk_idx: index of <unk>
    :param vocab_to_id: 词汇索引
    :param label_to_id: 标签索引
    :param path: 保存路径
    :return:
    """
    mapper = OrderedDict()

    mapper['vocab_size'] = vocab_size
    mapper['embedding_dim'] = embedding_dim
    mapper['pad_idx'] = pad_idx
    mapper['unk_idx'] = unk_idx
    mapper['label_to_id'] = label_to_id
    mapper['vocab_to_id'] = vocab_to_id

    with open(path, 'w', encoding='UTF-8-sig') as file:
        json.dump(mapper, file, ensure_ascii=False, indent=4)


def save_config(path):
    """
    保存基础配置
    :param path: 保存路径
    :return:
    """
    mapper = OrderedDict()
    config = basic_config()

    # 保存basic_config中的参数
    keys = list(filter(lambda x: not x.startswith('__'), dir(config)))
    key_value = dict([(key, config.__getattribute__(key)) for key in keys])
    for key, value in key_value.items():
        mapper[key] = value

    with open(path, 'w', encoding='UTF-8-sig') as file:
        json.dump(mapper, file, ensure_ascii=False, indent=4)


def save_acc(max_acc, path):
    """
    保存准确率
    :param max_acc: 当前最优准确率
    :param path: 保存路径
    :return:
    """
    mapper = OrderedDict()
    mapper['max_acc'] = max_acc

    with open(path, 'w', encoding='UTF-8-sig') as file:
        json.dump(mapper, file, ensure_ascii=False, indent=4)


def info(epoch_list, loss_list, acc_list, path):
    mapper = OrderedDict()

    mapper['epoch_list'] = epoch_list
    mapper['loss_list'] = loss_list
    mapper['acc_list'] = acc_list

    with open(path, 'w', encoding='UTF-8-sig') as file:
        json.dump(mapper, file, ensure_ascii=False, indent=4)


def get_acc(path):
    if not os.path.exists(path):
        return 0.0
    else:
        with open(path, 'r', encoding='UTF-8-sig') as file:
            info = json.load(file)
        return info['max_acc']
