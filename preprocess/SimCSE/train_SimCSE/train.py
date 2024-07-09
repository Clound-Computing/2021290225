# -*- coding: utf-8 -*-

import os
import datetime
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from datasets import load_dataset

import os


logger = logging.getLogger('SimCSE')
logger.setLevel(level=logging.INFO)
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class SimCSE(nn.Module):
    def __init__(self, pretrained, pool_type="cls", dropout_prob=0.3):
        super().__init__()
        # 从预训练的BERT模型配置文件中加载配置
        conf = BertConfig.from_pretrained(pretrained)
        # 设置注意力概率的dropout概率
        conf.attention_probs_dropout_prob = dropout_prob
        # 设置隐藏层的dropout概率
        conf.hidden_dropout_prob = dropout_prob
        # 初始化BERT模型作为编码器
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        # 确认池化类型是否有效（只允许 "cls" 或 "pooler" 两种类型）
        assert pool_type in ["cls", "pooler"], "invalid pool_type: %s" % pool_type
        self.pool_type = pool_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用BERT模型进行前向传播
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        # 根据池化类型选择输出的嵌入向量
        if self.pool_type == "cls":
            # 选择[CLS]标记的嵌入向量
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == "pooler":
            # 选择pooler输出的嵌入向量
            output = output.pooler_output
        return output


class CSECollator(object):
    def __init__(self,
                 tokenizer,
                 features=("input_ids", "attention_mask", "token_type_ids")):
        self.tokenizer = tokenizer
        self.features = features

    def collate(self, batch):
        new_batch = []
        for example in batch:
            for i in range(2):
                # repeat every sentence twice
                new_batch.append({fea: example[fea] for fea in self.features})
        new_batch = self.tokenizer.pad(
            new_batch,
            padding=False,
            return_tensors="pt"
        )
        return new_batch


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--train_file", type=str, help="train text file"
    )
    parser.add_argument(
        "--PTM_root_path", type=str, default="./PretrainedLM", help="the root path of PTMs"
    )
    parser.add_argument(
        "--pretrained", type=str, help="huggingface pretrained model"
    )
    parser.add_argument(
        "--num_proc", type=int, default=5, help="dataset process thread num"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="sentence max length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="learning rate"
    )
    parser.add_argument(
        "--tau", type=float, default=0.05, help="temperature coefficient"
    )
    parser.add_argument(
        "--display_interval", type=int, default=50, help="display interval"
    )
    parser.add_argument(
        "--save_final", type=bool, default=False, help="save the final model"
    )
    parser.add_argument(
        "--pool_type", type=str, default="cls", help="pool_type"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.3, help="dropout_rate"
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="num of gpu used"
    )
    args = parser.parse_args()

    return args


# make folder
def make_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(args, tokenizer):
    data_files = {"train": args.train_file}
    ds = load_dataset("text", data_files=data_files)
    logger.info(f"The size of dataset: {len(ds['train'])}")
    ds_tokenized = ds.map(
        lambda example: tokenizer(
            example["text"],
            add_special_tokens=True,  # add special tokens (<CLS> and <SEP>)
            max_length=args.max_length,  # set max sentence length
            truncation=True,
            padding='max_length',  # pad to max length
        ),
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=args.num_proc)
    collator = CSECollator(tokenizer)
    dl = DataLoader(ds_tokenized["train"],
                    batch_size=args.batch_size,
                    collate_fn=collator.collate)
    return dl


def compute_loss(y_pred, tau=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(
        y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - \
        torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tau
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(args, model_out):
    # 从预训练的BERT模型中加载分词器
    tokenizer = BertTokenizer.from_pretrained(args.PTM_root_path)
    # 保存分词器的词汇表
    tokenizer.save_vocabulary(model_out)
    # 加载数据
    dl = load_data(args, tokenizer)
    # 创建模型，将其移到设备上（CPU或GPU）
    model = SimCSE(args.PTM_root_path, args.pool_type, args.dropout_rate).to(device)
    # 如果有多个GPU，使用数据并行处理
    if args.n_gpu > 1:
        model = DataParallel(model, device_ids=[0, 1])
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 将模型设置为训练模式
    model.train()
    batch_idx = 0
    best_loss = float('inf')
    # 开始训练，循环遍历每个epoch
    for _ in range(args.epochs):
        # 循环遍历数据集中的每个batch
        for data in tqdm(dl):
            batch_idx += 1
            # 前向传播：将数据传入模型，得到预测结果
            pred = model(input_ids=data["input_ids"].to(device),
                         attention_mask=data["attention_mask"].to(device),
                         token_type_ids=data["token_type_ids"].to(device))
            # 计算损失值
            loss = compute_loss(pred, args.tau)
            # 清零梯度
            optimizer.zero_grad()
            # 反向传播：计算损失值对模型参数的梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 将损失值从tensor转为python数字
            loss = loss.item()
            # 每隔一定的步骤数，打印损失值，并保存最佳模型
            if batch_idx % args.display_interval == 0:
                logger.info(f"batch_idx: {batch_idx}, loss: {loss:>10f}")
                if loss <= best_loss:
                    best_loss = loss
                    if args.n_gpu == 1:
                        model.encoder.save_pretrained(model_out)
                    else:
                        model.module.encoder.save_pretrained(model_out)
    logger.info(f"best loss: {best_loss:>10f}")
    # 如果需要保存最后的模型
    if args.save_final:
        model_out = model_out + '-final'
        make_dir(model_out)
        tokenizer.save_vocabulary(model_out)
        if args.n_gpu == 1:
            model.encoder.save_pretrained(model_out)
        else:
            model.module.encoder.save_pretrained(model_out)
        logger.info(f"Save the final model!\n\tto{model_out}")


def main():

    # start time point of this training
    today = datetime.date.today()
    _, month, day = str(today).split('-')
    now = str(datetime.datetime.now())
    hour, minute = now.split(' ')[1].split(
        ':')[0], now.split(' ')[1].split(':')[1]
    folder_name = month + day
    file_name = month + day + hour + minute

    args = parse_args()

    # setup logger
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    log_root_path = f'./logs/{folder_name}'
    make_dir(log_root_path)
    file_handler = logging.FileHandler(
        os.path.join(log_root_path, f'{file_name}.log'))

    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # write hyperparameters to log file
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')

    make_dir(args.PTM_root_path)
    data_lang = args.train_file.split('/')[-1].split('_')[0]
    model_out = './ckpts/{}'.format(data_lang)
    make_dir(model_out)

    train(args, model_out)


if __name__ == "__main__":
    main()
