# 导入爬虫库
import requests

import json
import jieba
import torch

import pymysql

from test import info_load, transform
from model import BiLSTM


# 查询
def query(sql, args, connect):
    cursor = connect.cursor()

    cursor.execute(sql, args)
    results = cursor.fetchall()

    connect.commit()
    cursor.close()

    return results


# 插入
def insert(sql, args, connect):
    cursor = connect.cursor()
    cursor.execute(sql, args)
    connect.commit()
    cursor.close()


if __name__ == '__main__':
    db = pymysql.connect(host='localhost', port=3306, user='root', passwd='root', db='weibo_db')

    # 加载配置
    configs, params = info_load('info/config_file', 'info/param_file')
    # 定义模型
    model = BiLSTM(params['vocab_size'], params['embedding_dim'], 2,
                   configs['hidden_size'], configs['num_layers'], configs['dropout_rate'],
                   params['pad_idx'], params['unk_idx'])
    # 设置模型权重
    model.load_state_dict(torch.load(configs['best_model'], map_location='cpu'))
    model.eval()

    # 数据爬取
    url = 'https://m.weibo.cn/api/container/getIndex?containerid=1076032656274875_-_WEIBO_SECOND_PROFILE_WEIBO' \
          '&page_type=03&page={}'
    for i in range(1):
        response = requests.get(url.format(i))

        page = json.loads(response.text)
        blogs = page['data']['cards']

        for blog in blogs:
            count = 0  # 评论总数
            count_positive = 0  # 积极总数
            count_negative = 0  # 消极总数

            if blog['card_type'] == 9:
                blog_id = blog['mblog']['id']
                blog_text = blog['mblog']['text']
                blog_user_avatar = blog['mblog']['user']['avatar_hd']
                blog_user_name = blog['mblog']['user']['screen_name']

                detail = requests.get('https://m.weibo.cn/comments/hotflow?id={number}&mid={number}'
                                      '&max_id_type=0'.format(number=blog_id))
                data = json.loads(detail.text)
                comments = data['data']['data']
                for comment in comments:
                    comment_id = comment['id']
                    comment_text = comment['text']
                    comment_user_avatar = comment['user']['avatar_hd']
                    comment_user_name = comment['user']['screen_name']

                    # 清洗评论
                    comment_text = str(comment_text).split('<span class="url-icon">')[0]
                    comment_text = str(comment_text).split('<a')[0]

                    if comment_text == '':
                        continue
                    else:
                        # 数据库插入
                        insert('INSERT INTO comment VALUES(%s,%s,%s,%s,%s)',
                               (comment_id, comment_text, comment_user_avatar, comment_user_name, blog_id), db)
                        count += 1
                        words = jieba.lcut(comment_text)
                        with torch.no_grad():
                            inputs, length = transform(words, params['vocab_to_id'], configs['batch_size'])
                            predict = model.forward(inputs.t(), length).argmax(dim=1).item()
                            if predict:
                                count_negative += 1
                            else:
                                count_positive += 1

                result = count_positive / count * 100
                # 保留两位小数
                result = round(result, 2)

                # 数据库插入
                insert('INSERT INTO blog VALUES(%s,%s,%s,%s,%s)',
                       (blog_id, blog_text, blog_user_avatar, blog_user_name, result), db)
            else:
                continue

    db.close()

    # blog = blogs[1]
    # blog_id = blog['mblog']['id']
    # max_id = 0
    # for j in range(5):
    #     if j == 0:
    #         address = 'https://m.weibo.cn/comments/hotflow?id={number}&mid={number}&max_id_type=0'\
    #                   .format(number=blog_id)
    #     else:
    #         address = 'https://m.weibo.cn/comments/hotflow?id={number}&mid={number}&max_id={max_id}&max_id_type=0'\
    #                   .format(number=blog_id, max_id=max_id)
    #
    #     detail = requests.get(address)
    #     data = json.loads(detail.text)
    #     comments = data['data']['data']
    #     max_id = data['data']['max_id']
    #     for comment in comments:
    #         print(comment['text'])
