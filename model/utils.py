"""
工具类
"""
import json

from pyecharts.charts import WordCloud
import seaborn as sns
import pandas as pd
import jieba
import torch

from torchtext.legacy.data import Field, TabularDataset
from torchtext.vocab import Vectors
import matplotlib.pyplot as plt

from config import basic_config


def stopwords_load():
    """
    加载停用词表
    :return: 停用词表
    """
    result = [' ']
    for word in open('Dataset/中文停用词库.txt', 'r', encoding='UTF-8-sig'):
        result.append(word.strip())

    return result


def clean(path, stopwords, target):
    """
    在训练之前对原始数据进行清洗
    :param path: 源数据的存储路径
    :param stopwords: 停用词表
    :param target: 清洗后数据的存储位置
    :return: Null
    """
    raw_data = open(path, 'r', encoding='UTF-8-sig')
    clean_data = open(target, 'w', encoding='UTF-8-sig')

    raw_lines = raw_data.readlines()
    clean_data.write(raw_lines[0])
    for i in range(1, len(raw_lines)):
        result = ''
        clean_line = ''
        raw_line = raw_lines[i].split(',', 1)[1]
        words = jieba.lcut(raw_line)
        for word in words:
            if word not in stopwords:
                if word == '\n' or word == words[-1]:
                    if word == ',':
                        word = '，'
                    clean_line += word
                else:
                    if word == ',':
                        word = '，'
                    clean_line += word + ' '
        result += raw_lines[i].split(',', 1)[0] + ',' + clean_line
        clean_data.write(result)


def tokenize(text):
    """
    对文本数据进行切割
    :param text: 待处理的文本数据
    :return: 切割结果
    """
    text = str(text).split(' ')
    del text[-1]

    return text


def get_freq(dataframe):
    """
    统计词频
    :return: 词频表
    """
    result = {}
    for vocabs in dataframe:
        for vocab in vocabs:
            freq = result.get(vocab, 0)
            if freq == 0:
                result[vocab] = 1
            else:
                result[vocab] += 1

    return result


def visualize(path):
    """
    数据探测
    :param path: 数据的存储路径
    :return: Null
    """
    # 读取数据
    data = pd.read_csv(path)

    data['vocab'] = data['review'].apply(lambda x: jieba.lcut(x))
    data['vocab_size'] = data['review'].apply(lambda x: len(jieba.lcut(x)))
    data['text_length'] = data['review'].apply(lambda x: len(x))

    # 绘制直方图
    sns.distplot(data['vocab_size'])
    sns.distplot(data['text_length'])

    word_freq_0 = get_freq(data[data['label'] == 0]['vocab'])  # 对情感积极的文本统计词频
    word_freq_0 = sorted(word_freq_0.items(), key=lambda x: x[1], reverse=True)  # 按照词频从大到小重新排列
    word_freq_1 = get_freq(data[data['label'] == 1]['vocab'])  # 对情感消极的文本统计词频
    word_freq_1 = sorted(word_freq_1.items(), key=lambda x: x[1], reverse=True)  # 按照词频从大到小重新排列

    # 构建词云
    word_cloud = WordCloud()
    word_cloud.add('', data_pair=word_freq_0[0:100])
    word_cloud.render(path='WordCloud_0.html')
    word_cloud.add('', data_pair=word_freq_1[0:100])
    word_cloud.render(path='WordCloud_1.html')


def data_load(train_file, val_file):
    """
    数据加载
    :param train_file: 训练集文件名
    :param val_file: 验证集文件名
    :return: train_data, val_data, TEXT, LABEL
    """
    # 定义字段
    LABEL = Field(pad_token=None, unk_token=None)
    TEXT = Field(tokenize=tokenize, include_lengths=True)
    fields = [('label', LABEL), ('review', TEXT)]

    # 加载数据
    train_data, val_data = TabularDataset.splits(path='Dataset',
                                                 train=train_file,
                                                 validation=val_file,
                                                 format='csv',
                                                 fields=fields,
                                                 skip_header=True)

    # 加载词向量
    vectors = Vectors(name=basic_config.embedding_loc, cache=basic_config.cache)

    # 构建词汇
    TEXT.build_vocab(train_data, max_size=250000, vectors=vectors, unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data, vectors=vectors)

    return train_data, val_data, TEXT, LABEL


def draw(path):
    """
    绘制训练曲线
    :param path: 模型训练信息保存途径
    :return:
    """
    with open(path, 'r', encoding='UTF-8-sig') as file:
        info = json.load(file)
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(info['epoch_list'], info['loss_list'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(2, 1, 2)
    plt.plot(info['epoch_list'], info['acc_list'])
    plt.ylabel('acc')
    plt.xlabel('epoch')

    plt.show()


if __name__ == '__main__':
    # 加载停用词表
    # stopwords = stopwords_load()
    stopwords = [' ']

    # 对数据进行可视化探测
    visualize('Dataset/train.csv')  # 训练集
    visualize('Dataset/val.csv')  # 验证集
    visualize('Dataset/test.csv')  # 测试集

    # 数据清洗
    clean('Dataset/train.csv', stopwords, 'Dataset/clean_train.csv')  # 训练集
    clean('Dataset/val.csv', stopwords, 'Dataset/clean_val.csv')  # 验证集
    clean('Dataset/test.csv', stopwords, 'Dataset/clean_test.csv')  # 测试集
