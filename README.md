# 论文复现：假新闻检测与新闻环境感知
Zoom Out and Observe: News Environment Perception for Fake News Detection

这是该论文的官方资料:
> **Zoom Out and Observe: News Environment Perception for Fake News Detection**
>
> Qiang Sheng, Juan Cao, Xueyao Zhang, Rundong Li, Danding Wang, and Yongchun Zhu
>
> *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022)*
>
> [PDF](https://aclanthology.org/2022.acl-long.311.pdf) / [Poster](https://sheng-qiang.github.io/data/NEP-Poster.pdf) / [Code](https://github.com/ICTMCG/News-Environment-Perception) / [Chinese Video](https://www.bilibili.com/video/BV1MS4y1e7PY) / [Chinese Blog](https://mp.weixin.qq.com/s/aTFeuCYIpSoazeRi52jqew) / [English Blog](https://montrealethics.ai/zoom-out-and-observe-news-environment-perception-for-fake-news-detection/)


# 数据集

课堂上的数据集与原文数据集都已经处理后被放到该链接中：链接：https://pan.baidu.com/s/1R4C-2KvsSZO0xYrJvusy7Q 

# 代码

## 运行环境

```
python==3.6.10
torch==1.6.0
transformers==4.0.0
```

## 准备工作

### Step 1: 获取帖子和新闻环境的表示形式

#### Step1.1: 准备SimCSE模型

由于GitHub的空间限制，作者通过[Google Drive](https://drive.google.com/drive/folders/1J8p6ORqOhlpjl2lWAWq43pgUdG1O0L9T?usp=sharing)上传SimCSE的训练数据。您需要下载数据集文件(即' [dataset]_train.txt ')，并将其移动到此repo的' preprocess/SimCSE/train_SimCSE/data '中。然后:
```
cd preprocess/SimCSE/train_SimCSE

# Configure the dataset.
sh train.sh
```


#### Step1.2: 获取文本向量的具体表示

```
cd preprocess/SimCSE

# Configure the dataset.
sh run.sh
```

### Step 2:构造宏观/微观环境

获取宏观环境并按相似度对其内部表项进行排序：
```
cd preprocess/NewsEnv

# Configure the specific T days of the macro environment.
sh run.sh
```

### Step 3: 准备基础检测器

本文共有6个基本模型，它们的准备依赖关系如下:

<table>
   <tr>
       <td colspan="2"><b>Model</b></td>
       <td><b>Input (Tokenization)</b></td>
       <td><b>Special Preparation</b></td>
   </tr>
   <tr>
       <td rowspan="4"><b>Post-Only</b></td>
       <td>Bi-LSTM</td>
      <td>Word Embeddings</td>
      <td>-</td>
   </tr>
   <tr>
      <td>EANN</td>
      <td>Word Embeddings</td>
      <td>Event Adversarial Training</td>
   </tr>
   <tr>
      <td>BERT</td>
      <td>BERT's Tokens</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BERT-Emo</td>
      <td>BERT's Tokens</td>
      <td>Emotion Features</td>
   </tr>
   <tr>
       <td rowspan="2"><b>"Zoom-In"</b></td>
      <td>DeClarE</td>
      <td>Word Embeddings</td>
      <td rowspan="2">Fact-checking Articles</td>
   </tr>
   <tr>
      <td>MAC</td>
      <td>Word Embeddings</td>
   </tr>
</table>

在上述表中, 共有5个准备工作: (1) Tokenization by Word Embeddings, (2) Tokenization by BERT, (3) Event Adversarial Training, (4) Emotion Features, and (5) Fact-checking Articles.

#### Tokenization by Word Embeddings

这种标记化依赖于外部预训练的词嵌入。在本文中，我们使用[sgns.weibo. biggram -char](<https://github.com/Embedding/Chinese-Word-Vectors>)([下载网址](https://pan.baidu.com/s/1FHl_bQkYucvVk-j2KG4dxA))和[glove.840B.300d](https://github.com/stanfordnlp/GloVe)([下载网址](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip))作为中文和英文。

```
cd preprocess/WordEmbeddings

# Configure the dataset and your local word-embeddings filepath. 
sh run.sh
```

#### Tokenization by BERT

```
cd preprocess/BERT

# Configure the dataset and the pretrained model
sh run.sh
```

#### Event Adversarial Training

```
cd preprocess/EANN

# Configure the dataset and the event number
sh run.sh
```

#### Emotion Features

```
cd preprocess/Emotion/code/preprocess

# Configure the dataset
sh run.sh
```

#### Fact-checking Articles


```
cd preprocess/WordEmbeddings

# Configure the dataset and your local word-embeddings filepath. Set the data_type as 'article'.
sh run.sh
```

## 训练模型与推理

```
cd model

# Configure the dataset and the parameters of the model
sh run.sh
```

