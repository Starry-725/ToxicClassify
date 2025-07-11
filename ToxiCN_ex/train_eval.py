import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import time
import json
from src.datasets import to_tensor, convert_onehot,get_all_toxic_id
from src.Models import *
# from src.losses import *

# from tensorboardX import SummaryWriter
from ray import tune
import os


def train(config, train_iter, dev_iter, test_iter, task=1):
    embed_model = Bert_Layer(config).to(config.device)
    # embed_model = BiLSTM(config, embedding_weight).to(config.device)
    model = TwoLayerFFNNLayer(config).to(config.device)
    model_name = '{}_D-{}_B-{}_E-{}_Lr-{}_aplha-{}'.format(config.model_name, config.dropout, 
                                            config.batch_size, config.num_epochs, config.learning_rate, config.alpha1)
    embed_optimizer = optim.AdamW(embed_model.parameters(), lr=config.learning_rate)
    model_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    # fgm = FGM(embed_model, epsilon=1, emb_name='word_embeddings.')
    
    # 如果某些标签非常罕见（正样本很少），模型可能会倾向于全部预测为负样本。
    # 你可以通过 pos_weight 参数给正样本更高的权重，从而加大对稀有标签的惩罚。
    weights = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) 
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = get_loss_func("FL", [0.4, 0.6], config.num_classes, config.alpha1)
    max_score = 0

    for epoch in range(config.num_epochs):
        embed_model.train()
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        preds = []
        labels = []

        for batch in tqdm(train_iter, desc='Training', colour = 'MAGENTA'):
            embed_model.zero_grad()
            model.zero_grad()
            # print(batch)
            args = to_tensor(batch)
            
            att_input, pooled_emb = embed_model(**args)

            logit = model(att_input, pooled_emb).cpu()

            # label = args['toxic']
            label = args['toxic_type']
            # label = args['expression']
            # label = args['target']
            loss = loss_fn(logit, label.float())
            # pred = get_preds(config, logit)  
            # pred = get_preds_task2_4(config, logit)  
            pred = get_preds_task3(config, logit)  
            preds.extend(pred)
            labels.extend(label.detach().numpy())

            loss_all += loss.item()
            embed_optimizer.zero_grad()
            model_optimizer.zero_grad()
            loss.backward()

            embed_optimizer.step()
            model_optimizer.step()

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time)/60.))
        print("TRAINED for {} epochs".format(epoch))

        # 验证
        if epoch >= config.num_warm:
            # print("training loss: loss={}".format(loss_all/len(data)))
            trn_scores = get_scores(preds, labels, loss_all, len(train_iter), data_name="TRAIN")
            dev_scores, _ = eval(config, embed_model, model, loss_fn, dev_iter, data_name='DEV')

            # Ensure the directory exists
            os.makedirs(config.result_path, exist_ok=True)
            # Define the full path to the file, now using the relative path
            file_path = os.path.join(config.result_path, 'all_scores.txt')

            f = open(file_path, 'a')
            f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(epoch))
            f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(json.dumps(trn_scores), json.dumps(dev_scores))) 
            max_score = save_best(config, epoch, model_name, embed_model, model, dev_scores, max_score)
        print("ALLTRAINED for {} epochs".format(epoch))
    
    path = '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST')
    checkpoint = torch.load(path)
    embed_model.load_state_dict(checkpoint['embed_model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    test_scores, _ = eval(config, embed_model, model, loss_fn, test_iter, data_name='DEV')

    file_path = '{}/{}.all_scores.txt'.format(config.result_path, model_name)
    if not os.path.isfile(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    f = open(file_path, 'a')
    f.write('Test: \n{}\n'.format(json.dumps(test_scores)))
    # f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
    # f.write('Test: \n{}\n'.format(json.dumps(test_scores)))


def eval(config, embed_model, model, loss_fn, dev_iter, data_name='DEV'):
    loss_all = 0.
    preds = []
    labels = []
    for batch in tqdm(dev_iter, desc='Evaling', colour = 'CYAN'):
        with torch.no_grad():
            args = to_tensor(batch)
            att_input, pooled_emb = embed_model(**args)
            logit = model(att_input, pooled_emb)

            logit = logit.cpu()
            # label = args['toxic']
            label = args['toxic_type']
            # label = args['expression']
            # label = args['target']
            loss = loss_fn(logit, label.float())
            # pred = get_preds(config, logit)  
            # pred = get_preds_task2_4(config, logit)  
            pred = get_preds_task3(config, logit)  
            preds.extend(pred)
            labels.extend(label.detach().numpy())
            loss_all += loss.item()
            
    dev_scores = get_scores(preds, labels, loss_all, len(dev_iter), data_name=data_name)
    # if data_name != "TEST": # 2022.9.20 命令行输入为test模式时，不调用tune
    # tune.report(metric=dev_scores)

    return dev_scores, preds


# 预测函数
def predict(config, ori_text, tokenizer, embed_model, model, all_dirty_words):
    """
    预测单个句子的毒性
    
    Args:
        text: 需要预测的文本
        model_path: 模型检查点路径
        model_name: 预训练模型名称
        lexicon_path: 词典文件所在目录
        
    Returns:
        dict: 预测结果
    """
    # 1. 加载配置
    # all_dirty_words = get_all_dirty_words(lexicon_base_path)
    # tokenizer = AutoTokenizer.from_pretrained(model)
    encoded = tokenizer(ori_text, add_special_tokens=True,
                            max_length=80, padding='max_length', truncation=True)
    toxic_ids = get_all_toxic_id(80, encoded['input_ids'], all_dirty_words)
    test_data = [[{'text_idx':encoded['input_ids'],'text_ids':encoded['token_type_ids'],'text_mask':encoded['attention_mask'],'toxic_ids':toxic_ids}]]
    # test_iter = Dataloader(test_iter,  batch_size=int(config.batch_size), shuffle=False)
    embed_model.eval()
    model.eval()
    preds = []
    for batch in test_data:
        with torch.no_grad():
            args = to_tensor(batch)
            att_input, pooled_emb = embed_model(**args)
            # print("att_input, pooled_emb",att_input, pooled_emb)
            logit = model(att_input, pooled_emb)
            # print("logit:",logit)
            # pred = get_preds(config, logit)
            # 应用 sigmoid 得到概率
            probs = torch.sigmoid(logit)
            # pred = get_preds_task2_4(config, logit)
            pred = get_preds_task3(config,logit) 
            preds.extend(pred)

    return toxic_ids, probs, preds




# For Multi Classfication
def get_preds(config, logit):
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    new_results = []
    for result in results:
        result = convert_onehot(config, result)
        new_results.append(result)
    return new_results

# Task 2 and 4: 多分类 Toxic Type Discrimination and d Expression Type Detection
def get_preds_task2_4(config, logit):
    all_results = []
    logit_ = torch.sigmoid(logit)
    results_pred = torch.max(logit_.data, 1)[0].cpu().numpy() # 获得的是sigmoid的值，这些是置信度
    results = torch.max(logit_.data, 1)[1].cpu().numpy() # index for maximum probability，[1]获得的是最高值的下标
    for i in range(len(results)):
        if results_pred[i] < config.confi_thres:
            result = [0 for i in range(config.num_classes)]
        else:
            result = convert_onehot(config, results[i])
        all_results.append(result)
    return all_results

# Task 3: 多标签分类 Targeted Group Detection
def get_preds_task3(config, logit):
    all_results = []
    logit_ = torch.sigmoid(logit)
    results_pred = torch.max(logit_.data, 1)[0].cpu().numpy()
    results = torch.max(logit_.data, 1)[1].cpu().numpy()
    logit_ = logit_.detach().cpu().numpy()
    for i in range(len(results)):
        if results_pred[i] < config.confi_thres:
            result = [0 for i in range(config.num_classes)]
        else:
            result = get_pred_task3(config,logit_[i])
        all_results.append(result)
    return all_results

def get_pred_task3(config,logit):
    result = [0 for i in range(len(logit))]
    for i in range(len(logit)):
        if logit[i] >= config.confi_thres:
            result[i] = 1
    return result

def get_scores(all_preds, all_lebels, loss_all, len, data_name):
    score_dict = dict()
    f1 = f1_score(all_preds, all_lebels, average='weighted')
    # acc = accuracy_score(all_preds, all_lebels)
    all_f1 = f1_score(all_preds, all_lebels, average=None)
    pre = precision_score(all_preds, all_lebels, average='weighted')
    recall = recall_score(all_preds, all_lebels, average='weighted')

    score_dict['F1'] = f1
    # score_dict['accuracy'] = acc
    score_dict['all_f1'] = all_f1.tolist()
    score_dict['precision'] = pre
    score_dict['recall'] = recall

    score_dict['all_loss'] = loss_all/len
    print("Evaling on \"{}\" data".format(data_name))
    for s_name, s_val in score_dict.items(): 
        print("{}: {}".format(s_name, s_val)) 
    return score_dict

def save_best(config, epoch, model_name, embed_model, model, score, max_score):
    score_key = config.score_key
    curr_score = score[score_key]
    print('The epoch_{} {}: {}\nCurrent max {}: {}'.format(epoch, score_key, curr_score, score_key, max_score))

    if curr_score > max_score or epoch == 0:
        # Ensure the directory exists
        os.makedirs(os.path.join(config.checkpoint_path, "ckp-hfl"), exist_ok=True)

        torch.save({
        'epoch': config.num_epochs,
        'embed_model_state_dict': embed_model.state_dict(),
        'model_state_dict': model.state_dict(),
        }, '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))
        return curr_score
    else:
        return max_score
