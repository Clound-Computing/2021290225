import pickle
import json
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
import os

 # 这个函数使用BERT分词器将输入的句子转换为token序列
 # add_special_tokens=False参数表示不添加BERT特有的特殊token（如CLS和SEP）
def get_tokens(sentence):
    return tokenizer.encode(sentence, add_special_tokens=False)

# 主函数首先解析命令行参数，获取数据集名称和预训练模型的路径。然后创建保存数据的目录，并实例化BERT分词器。
if __name__ == '__main__':
    parser = ArgumentParser(description='Tokenize by Transformers')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pretrained_model', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    pretrained_model = args.pretrained_model
    save_dir = 'data/{}'.format(dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    print('Dataset: {}, Pretrained Model: {}\n'.format(
            dataset, pretrained_model))

     # 这部分代码处理训练集、验证集和测试集数据。
     # 对于每个数据集，它读取JSON文件，使用分词器处理每个文本片段，记录token的数量，并将结果打印出来。
     # 处理后的token序列保存为.pkl文件，token数量的统计信息保存为.csv文件。
    for t in ['train', 'val', 'test']:
        file = '../../dataset/{}/post/{}.json'.format(args.dataset, t)
        with open(file, 'r') as f:
            pieces = json.load(f)

        pieces_tokens = [get_tokens(p['content']) for p in tqdm(pieces)]
        df = pd.DataFrame(
            {'tokens_num': [len(tokens) for tokens in pieces_tokens]})

        print('File: {}'.format(file))
        print('Posts: {}\nTokens num: {}\n'.format(len(df), df.describe()))

        # Export
        with open(os.path.join(save_dir, '{}.pkl'.format(t)), 'wb') as f:
            pickle.dump(pieces_tokens, f)
        df.describe().to_csv(os.path.join(save_dir, '{}.csv'.format(t)))
