import json
import torch
import jieba
import pandas as pd
import numpy as np
from model import BiLSTM
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools


def stopwords_load():
    """
    加载停用词表
    :return: 停用词表
    """
    result = [' ']
    for word in open('Dataset/四川大学机器智能实验室停用词库.txt', 'r', encoding='UTF-8-sig'):
        result.append(word.strip())

    return result


def info_load(config_file, param_file):
    """
    加载配置及参数
    :param config_file: 配置文件
    :param param_file: 参数文件
    :return:
    """
    # 加载配置
    with open(config_file, 'r', encoding='UTF-8-sig') as file1:
        config = json.load(file1)
    # 加载参数
    with open(param_file, 'r', encoding='UTF-8-sig') as file2:
        param = json.load(file2)

    return config, param


# def text_dealer(text, stopwords):
#     """
#     文本处理
#     :param text: 待处理的文本
#     :param stopwords: 停用词表
#     :return: 文本处理结果
#     """
#     result = []
#     tokens = jieba.lcut(text)
#     for token in tokens:
#         if token not in stopwords:
#             result.append(token)
#
#     return result


def text_dealer(text):
    """
    文本处理
    :param text: 待处理的文本
    :return: 文本处理结果
    """
    text = str(text).split(' ')
    del text[-1]

    return text


def transform(words, vocab_to_id, batch_size):
    """
    将文本转换为可用于模型计算的输入
    :param words: 文本数据
    :param vocab_to_id: 词汇-id的映射表
    :param batch_size: batch大小
    :return:
    """
    result = []
    for word in words:
        result.append(vocab_to_id.get(word, 0))
    pad_s = [1] * (batch_size - len(result))
    # 对序列进行<pad>操作
    result = result + pad_s

    # 将经过<pad>之后的序列转为tensor
    inputs = torch.tensor(result).unsqueeze(0)
    # 将经过<pad>之后的序列长度转为tensor
    length = torch.tensor([len(result)])

    return inputs, length


if __name__ == '__main__':
    # 加载停用词表
    stopwords = stopwords_load()

    # 加载配置
    configs, params = info_load('info/config_file', 'info/param_file')

    # 定义模型
    model = BiLSTM(params['vocab_size'], params['embedding_dim'], 2,
                   configs['hidden_size'], configs['num_layers'], configs['dropout_rate'],
                   params['pad_idx'], params['unk_idx'])
    # 设置模型权重
    model.load_state_dict(torch.load(configs['best_model'], map_location='cpu'))
    model.eval()

    # 效果评估
    test_data = pd.read_csv('Dataset/clean_test.csv')

    predicts = np.array([], dtype=int)
    labels = np.array([], dtype=int)

    for i in range(test_data.shape[0]):
        record = test_data.loc[i, :].to_dict()
        words = text_dealer(record['review'])

        inputs, length = transform(words, params['vocab_to_id'], configs['batch_size'])
        label = torch.tensor(record['label'])

        predict = model.forward(inputs.t(), length).argmax(dim=1).item()

        labels = np.append(labels, int(label.item()))
        predicts = np.append(predicts, int(predict))

    label_to_mean = {'0': '积极', '1': '消极'}

    report = metrics.classification_report(labels, predicts, target_names=['积极', '消极'], digits=4)
    print(report)

    # 生成混淆矩阵
    confusion_matrix = metrics.confusion_matrix(labels, predicts)
    print(confusion_matrix)
    # 混淆矩阵可视化
    plt.rc('figure', figsize=(10, 8))
    plt.rc('font', size=10)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.get_cmap('Reds'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['积极', '消极']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="black")

    plt.ylabel('True')
    plt.xlabel('Predict')

    plt.show()

    # # 模型测试
    # label_to_mean = {'0': '积极', '1': '消极'}
    #
    # text = jieba.lcut('绽放在高山砾石间的藏波罗花')
    # # text = text_dealer(text, stopwords)
    # with torch.no_grad():
    #     data, seq_len = transform(text, params['vocab_to_id'], configs['batch_size'])
    #     predict = model.forward(data.t(), seq_len)
    #
    #     label = predict.argmax(dim=1).item()
    #     score = torch.nn.Softmax(predict)
    #
    #     print(label, label_to_mean[str(label)])
    #     print(score)
