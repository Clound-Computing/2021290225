import os
import time
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler

from config import parser
from evaluate import evaluate
from DatasetLoader import DatasetLoader
from ModelFramework import EnvEnhancedFramework

if __name__ == "__main__":
    args = parser.parse_args()  # 解析命令行参数

    if args.debug:
        args.save = './ckpts/debug'  # 设置调试模式下的保存路径
        args.epochs = 2  # 设置调试模式下的训练轮数

    if os.path.exists(args.save):
        os.system('rm -r {}'.format(args.save))  # 如果保存路径已存在，则删除
    if not os.path.exists(args.save):
        os.mkdir(args.save)  # 创建保存路径

    print('\n{} Experimental Dataset: {} {}\n'.format(
        '=' * 20, args.dataset, '=' * 20))  # 打印实验数据集信息
    print('save path: ', args.save)  # 打印保存路径
    print('Start time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))  # 打印开始时间

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备
    args.n_gpu = torch.cuda.device_count()  # 获取GPU数量

    random.seed(args.seed)  # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 设置所有GPU的随机种子

    print('Loading data...')
    start = time.time()

    train_dataset = DatasetLoader(args, 'train')  # 加载训练数据集
    val_dataset = DatasetLoader(args, 'val')  # 加载验证数据集
    test_dataset = DatasetLoader(args, 'test')  # 加载测试数据集

    if not args.evaluate:
        train_sampler = RandomSampler(train_dataset)  # 训练集随机采样器
    else:
        train_sampler = SequentialSampler(train_dataset)  # 训练集顺序采样器
    val_sampler = SequentialSampler(val_dataset)  # 验证集顺序采样器
    test_sampler = SequentialSampler(test_dataset)  # 测试集顺序采样器

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=train_sampler
    )  # 创建训练数据加载器

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=val_sampler
    )  # 创建验证数据加载器

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=test_sampler
    )  # 创建测试数据加载器

    print('Loading data time: {:.2f}s\n'.format(time.time() - start))  # 打印数据加载时间

    print('-----------------------------------------\nLoading model...\n')
    start = time.time()
    model = EnvEnhancedFramework(args)  # 创建模型
    print(model)  # 打印模型结构
    print('\nLoading model time: {:.2f}s\n-----------------------------------------\n'.format(
        time.time() - start))  # 打印模型加载时间

    criterion = nn.CrossEntropyLoss().cuda()  # 设置损失函数
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                             model.parameters()), lr=args.lr)  # 设置优化器

    if args.fp16:
        scaler = GradScaler()  # 如果使用混合精度训练，创建GradScaler

    model = model.cuda()  # 将模型移动到GPU

    if args.resume != '':
        resume_dict = torch.load(args.resume)  # 加载之前训练的模型权重和优化器状态
        model.load_state_dict(resume_dict['state_dict'])
        optimizer.load_state_dict(resume_dict['optimizer'])
        args.start_epoch = resume_dict['epoch'] + 1  # 设置起始epoch

    with open(os.path.join(args.save, 'args.txt'), 'w') as f:
        for arg in vars(args):
            v = getattr(args, arg)
            s = '{}\t{}'.format(arg, v)
            f.write('{}\n'.format(s))
            print(s)  # 将参数写入文件并打印
        f.write('\n{}\n'.format(model))  # 将模型结构写入文件

    if args.evaluate:
        if not args.resume:
            print('No trained .pt file loaded.\n')  # 如果没有加载预训练模型，打印提示信息

        print('Start Evaluating... local_rank=', args.local_rank)  # 打印评估开始信息
        args.current_epoch = args.start_epoch

        train_losses, _ = evaluate(
            args, train_loader, model, criterion, 'train', inference_analysis=args.inference_analysis)  # 评估训练集
        val_losses, _ = evaluate(args, val_loader, model, criterion,
                                 'val', inference_analysis=args.inference_analysis)  # 评估验证集
        test_losses, _ = evaluate(
            args, test_loader, model, criterion, 'test', inference_analysis=args.inference_analysis)  # 评估测试集

        exit()  # 退出程序

    last_epoch = args.start_epoch if args.start_epoch != 0 else -1

    if args.local_rank in [-1, 0]:
        print('Start training...')  # 打印训练开始信息

    best_val_result = 0  # 初始化最佳验证结果
    best_val_epoch = -1  # 初始化最佳验证epoch

    start = time.time()
    args.global_step = 0  # 初始化全局步数
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch
        print('\n------------------------------------------------\n')
        print('Start Training Epoch', epoch, ':', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))  # 打印当前训练epoch信息
        model.train()  # 设置模型为训练模式

        train_loss = 0.  # 初始化训练损失

        lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        print_step = int(len(train_loader) / 20)  # 设置打印步数
        print_step = 10
        for step, (idxs, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()  # 梯度清零

            with autocast():  # 自动混合精度
                out, h_mac, h_mic = model(idxs, train_dataset)  # 前向传播

                if args.model == 'EANN':
                    out, event_out = out
                    event_labels = train_dataset.event_labels[idxs]

                labels = labels.long().to(args.device)
                CEloss = criterion(out, labels)  # 计算交叉熵损失

                if args.model == 'EANN':
                    event_loss = criterion(event_out, event_labels)
                    event_loss = args.eann_weight_of_event_loss * event_loss
                    CEloss += event_loss

                loss = CEloss

                if torch.any(torch.isnan(loss)):
                    print('out: ', out)
                    print('loss = {:.4f}\n'.format(loss.item()))
                    exit()  # 如果损失为NaN，退出程序

                if step % print_step == 0:
                    print('\n\nEpoch: {}, Step: {}, CELoss = {:.4f}'.format(
                        epoch, step, CEloss.item()))  # 打印损失信息

            if args.fp16:
                scaler.scale(loss).backward()  # 反向传播
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

            train_loss += loss.item()  # 累加训练损失
            args.global_step += 1  # 更新全局步数

        train_loss /= len(train_loader)  # 计算平均训练损失
        val_loss, val_result = evaluate(
            args, val_loader, model, criterion, 'val')  # 评估验证集
        test_loss, test_result = evaluate(
            args, test_loader, model, criterion, 'test')  # 评估测试集

        print('='*10, 'Epoch: {}/{}'.format(epoch, args.epochs),
              'lr: {}'.format(lr), '='*10)  # 打印epoch信息
        print('\n[Loss]\nTrain: {:.6f}\tVal: {:.6f}\tTest: {:.6f}'.format(
            train_loss, val_loss, test_loss))  # 打印损失信息
        print('-'*10)
        print('\n[Macro F1]\nVal: {:.6f}\tTest: {:.6f}\n'.format(
            val_result, test_result))  # 打印评估结果
        print('-'*10)

        if val_result >= best_val_result:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                os.path.join(args.save, '{}.pt'.format(epoch))
            )  # 保存最佳模型

            if best_val_epoch != -1:
                os.system('rm {}'.format(os.path.join(
                    args.save, '{}.pt'.format(best_val_epoch))))  # 删除之前的最佳模型

            best_val_result = val_result  # 更新最佳验证结果
            best_val_epoch = epoch  # 更新最佳验证epoch

    print('Training Time: {:.2f}s'.format(time.time() - start))  # 打印训练时间
    print('End time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))  # 打印结束时间
