# 导入必要的库
import json
import datetime
from argparse import ArgumentParser
import torch
import pickle
from tqdm import tqdm

# 获取某个日期前k天的新闻索引
def fetch_past_k_days_unlabel_news(k, date2idx, post_date):
    assert k >= 1
    all_idx = []
    post_date = datetime.datetime.strptime(post_date, "%Y-%m-%d")
    for delta_date in range(1, k+1):
        target_date = (post_date + datetime.timedelta(
            days=-delta_date)).strftime("%Y-%m-%d")
        if (target_date in date2idx):
            idx = date2idx[target_date]
        else:
            idx = []
            if k == 1:
                print(
                    "Post at {} don't have news at yesterday.".format(post_date))
        all_idx.extend(idx)
    return all_idx

# 计算帖子和新闻之间的余弦相似度并排序
def get_cosine(post, date2recent_news_idx, all_news_vec, post_vec, dataset, split_name, days):
    ranked_sims_in_n_days = []
    for i in tqdm(range(len(post))):
        date = post[i]['time'].split(" ")[0]
        news_idxs = date2recent_news_idx[date]
        news_vec = all_news_vec[news_idxs, :]
        # 计算余弦相似度
        cos_sims = torch.matmul(post_vec[i], news_vec.transpose(0, 1)).tolist()
        tup = zip(news_idxs, cos_sims)
        tup = sorted(tup, key=lambda x: x[1], reverse=True)
        ranked_sims_in_n_days.append(tup)
    # 将相似度结果保存到文件中
    pickle.dump(ranked_sims_in_n_days, open(
        './data/{}/{}_{}d.pkl'.format(dataset, split_name, days), 'wb'))

# 主函数
if __name__ == "__main__":

    parser = ArgumentParser(description='get_env')
    parser.add_argument('--dataset', type=str, default="Chinese")
    parser.add_argument('--macro_env_days', type=int, default=3)
    args = parser.parse_args()

    dataset = args.dataset
    days = args.macro_env_days

    # 加载数据集
    train = json.load(
        open('News-Environment-Perception-main\\dataset\\{}\\post\\train.json'.format(dataset), 'r'))
    val = json.load(
        open('News-Environment-Perception-main\\dataset\\{}\\post\\val.json'.format(dataset), 'r'))
    test = json.load(
        open('News-Environment-Perception-main\\dataset\\{}\\post\\test.json'.format(dataset), 'r'))
    news = json.load(
        open('News-Environment-Perception-main\\dataset\\{}\\news\\news.json'.format(dataset), 'r'))

    # 加载预训练的嵌入向量
    train_vec = pickle.load(open(
        'News-Environment-Perception-main\\preprocess\\SimCSE\\data\\{}\\post\\train_vecs_origin.pkl'.format(dataset), 'rb'))
    val_vec = pickle.load(open(
        'News-Environment-Perception-main\\preprocess\\SimCSE\\data\\{}\\post\\val_vecs_origin.pkl'.format(dataset), 'rb'))
    test_vec = pickle.load(open(
        'News-Environment-Perception-main\\preprocess\\SimCSE\\data\\{}\\post\\test_vecs_origin.pkl'.format(dataset), 'rb'))
    news_vec = pickle.load(open(
        'News-Environment-Perception-main\\preprocess\\SimCSE\\data\\{}\\news\\all_vecs_origin.pkl'.format(dataset), 'rb'))

    # L2正则化嵌入向量
    train_vec = train_vec / train_vec.norm(dim=1, keepdim=True)
    val_vec = val_vec / val_vec.norm(dim=1, keepdim=True)
    test_vec = test_vec / test_vec.norm(dim=1, keepdim=True)
    news_vec = news_vec / news_vec.norm(dim=1, keepdim=True)

    # 创建日期到新闻索引的映射
    date2idx = {}
    for i in range(len(news)):
        date = news[i]['date'] #英文数据集
        if date in date2idx:
            date2idx[date].append(i)
        else:
            date2idx[date] = []
            date2idx[date].append(i)

    # 获取所有帖子日期
    all_post_dates = set()
    for dtst in [train, val, test]:
        for d in dtst:
            all_post_dates.add(d['time'].split(' ')[0])
    all_post_dates = sorted(list(all_post_dates))

    # 获取每个帖子日期前几天的新闻索引
    date2recent_news_idx = {}
    for d in all_post_dates:
        date2recent_news_idx[d] = fetch_past_k_days_unlabel_news(
            days, date2idx, d)

    # 计算训练集、验证集和测试集的余弦相似度并保存结果
    get_cosine(train, date2recent_news_idx, news_vec,
               train_vec, dataset, 'train_data', days)
    get_cosine(val, date2recent_news_idx, news_vec,
               val_vec, dataset, 'val_data', days)
    get_cosine(test, date2recent_news_idx, news_vec,
               test_vec, dataset, 'test_data', days)
