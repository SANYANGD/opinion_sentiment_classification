import os
import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class,
                 hidden_size, num_layers, dropout_rate,
                 pad_idx, unk_idx, pre_trained=None):
        """
        模型dyi
        :param vocab_size: 词汇大小
        :param embedding_dim: 嵌入维度
        :param num_class: 输出类别
        :param pad_idx: index of <pad>
        :param unk_idx: index of <unk>
        :param pre_trained: 训练好的词嵌入模型（如果有的话）
        """
        super(BiLSTM, self).__init__()

        # 实例化embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # 如果存在训练好的词嵌入模型
        if pre_trained is not None:
            # 将训练好的词嵌入模型的权重复制给当前模型的embedding层
            self.embedding.weight.data.copy_(pre_trained)

            # 将<pad>和<unk>对应的权重设为0
            self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
            self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

        # 实例化encoder层
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=True)

        # 实例化decoder层
        self.decoder = nn.Linear(2 * hidden_size, num_class)

        # 定义dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, length):
        """
        前向传播
        :param inputs: 输入文本
        :param length: 输入文本长度
        :return:
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(inputs))

        # 将经过<pad>填充的序列压紧
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length)

        packed_outputs, (hidden, cell) = self.encoder(packed_embedded)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.decoder(hidden)


def save_model(state, folder, model_name):
    """
    模型保存
    :param state: 当前模型权重及偏执
    :param folder: 文件夹
    :param model_name: 文件名
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    path = os.path.join(folder, model_name)
    torch.save(state, path)
