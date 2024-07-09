import enum
from pickle import load
from tqdm import tqdm
from sklearn.metrics import classification_report
import time
import numpy as np
import json
import os
import torch
from config import INDEX2LABEL, INDEX_OF_LABEL, MAX_TOKENS_OF_A_POST


def compute_classification_metrics(ans, pred, category_num=2):
    return classification_report(ans, pred, target_names=INDEX2LABEL, digits=4, output_dict=True)


def eval_when_training_on_single_gpu(outputs_file, dataset):
    # gt = np.array([x[-1].item() for x in tqdm(dataset)])
    gt = dataset.labels

    wrong_cls_cases = []
    classifying_ans = np.array([], dtype=int)
    classifying_pred = np.array([], dtype=int)

    out = json.load(open(outputs_file, 'r'))
    # outputs = []
    # for o in out:
    #     outputs += o
    outputs = out

    for o in outputs:
        # o: [idx, 0_class_score, 1_class_score]
        idx, scores = o[0], np.array(o[1:])
        pred = int(scores.argmax())
        ans = int(gt[idx])

        if ans != pred:
            case = {'idx': idx, 'label': INDEX2LABEL[ans],
                    'prediction': INDEX2LABEL[pred], 'prediction_scores': list(scores)}
            wrong_cls_cases.append(case)

        classifying_ans = np.append(classifying_ans, ans)
        classifying_pred = np.append(classifying_pred, pred)

    class_report = compute_classification_metrics(classifying_ans, classifying_pred,
                                                  category_num=len(outputs[0]) - 1)

    res_file = outputs_file.replace('_outputs_', '_res_')
    wrong_cls_file = res_file.replace('_res_', '_wrong_cls_')
    json.dump({'classification': class_report}, open(res_file, 'w'), indent=4)
    json.dump(wrong_cls_cases, open(wrong_cls_file, 'w'), indent=4)

    return class_report['macro avg']['f1-score']


def evaluate(args, loader, model, criterion, dataset_type, inference_analysis=False):
    # 打印评估开始的时间。
    if args.local_rank in [-1, 0]:
        print('Eval time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    model.eval()  # 设置模型为评估模式

    eval_loss = 0.
    outputs = []
    env_weights = []
    raw_weights = dict()

    with torch.no_grad():  # 禁用梯度计算
        for idxs, labels in tqdm(loader):  # 遍历数据加载器
            out, h_mac, h_mic = model(idxs, loader.dataset)  # 前向传播
            # 如果启用了推理分析，计算环境权重并存储。
            if inference_analysis:
                weight_g = model.inference_analysis(idxs, loader.dataset)
                for i, idx in enumerate(idxs):
                    raw_weights[idx.item()] = weight_g[i].tolist()
                macro_weights = torch.sum(weight_g, dim=-1) / weight_g.size(-1)
                micro_weights = 1 - macro_weights
                macro_weights = macro_weights.tolist()
                micro_weights = micro_weights.tolist()

            # 如果使用的是EANN模型，处理事件输出和事件标签。
            if args.model == 'EANN':
                out, event_out = out
                event_labels = loader.dataset.event_labels[idxs]

            # 将标签移到设备上，计算交叉熵损失，累加评估损失
            labels = labels.long().to(args.device)
            CEloss = criterion(out, labels)
            if args.model == 'EANN':
                event_loss = criterion(event_out, event_labels)
                event_loss = args.eann_weight_of_event_loss * event_loss
                CEloss += event_loss
            loss = CEloss
            eval_loss += loss.item()

            # 将模型输出转换为列表并存储，如果启用了推理分析，存储环境权重。
            score = [[idxs[i].item()] + x for i, x in enumerate(out.cpu().numpy().tolist())]
            if inference_analysis:
                for i, idx in enumerate(idxs):
                    env_weights.append({'idx': idx.item(), 'macro_weight': macro_weights[i], 'micro_weight': micro_weights[i]})
            outputs += score

        eval_loss /= len(loader) # 计算平均评估损失

    file = os.path.join(args.save, dataset_type + '_outputs_' + str(args.current_epoch) + '.json')
    with open(file, 'w') as f:
        json.dump(outputs, f)
        
    # 如果启用了推理分析，保存环境权重
    if inference_analysis:
        with open(file.replace('_outputs_', '_env_weights_'), 'w') as f:
            json.dump(env_weights, f, indent=4, ensure_ascii=False)
        with open(file.replace('_outputs_', '_env_weights_raw_'), 'w') as f:
            json.dump(raw_weights, f, indent=4, ensure_ascii=False)
    
    # 计算F1
    classification_metrics = eval_when_training_on_single_gpu(file, loader.dataset)
    return eval_loss, classification_metrics
