import json
from transformers import BertTokenizer, AutoTokenizer
import re
import pandas as pd
import os
import random

def shuffle_lines_in_file(file_path):
    # 打开文件并读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 随机打乱行的顺序
    random.shuffle(lines)

    # 将打乱后的内容写回源文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


# 将txt文本打上毒性标签,并写入csv文件
def append_label_trans_csv(txt_path, csv_path, label_num):
    # 打开文件并读取内容
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 对lines进行去重
    unique_lines = list(set(lines))

    # 打开输出文件以写入模式
    with open(csv_path, 'w', encoding='utf-8') as out_file:
        for line in unique_lines:
            stripped_line = line.strip()
            if stripped_line:  # 忽略处理后的空行
                # 直接构造字符串并写入，末尾加上换行符
                out_file.write(f"{stripped_line}\t{label_num}\n")


def duty_words_trans_json(dutywords_txt,dutywords_json):
    # 加载tokenizer，必须与模型匹配
    # 选择一个选项加载
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/ToxicClassify/chinese-roberta-wwm-ext")


    # 读取txt文件中的词汇列表
    duty_words = []
    with open(dutywords_txt, "r", encoding="utf-8") as f:
        for line in f:
            # 去除每行末尾的换行符并添加到列表中
            if '\t' in line:
                word = line.strip().split('\t')[0]
            else:
                word = line.strip()
            if word:  # 确保不是空行
                duty_words.append(word)


    # 创建词汇到token_ids的映射
    word_to_token_ids = {}

    for word in duty_words:
        # 对每个词进行tokenize
        tokens = tokenizer.tokenize(word)
        # 转换为token ids
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 保存映射
        word_to_token_ids[word] = token_ids

    # 保存为JSON文件
    with open(dutywords_json, "w", encoding="utf-8") as f:
        json.dump(word_to_token_ids, f, ensure_ascii=False)


def write_json_keys_to_txt(json_file_path, txt_file_path):
    # 打开JSON文件并读取内容
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # 打开TXT文件以写入模式
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        # 遍历每一项并写入键
        for item in data:
            #print(item)
            txt_file.write(item + '\n')


if __name__ == "__main__":
    # 敏感词典编码并转化为json
    duty_words_trans_json(dutywords_txt="A01.txt",dutywords_json="A01.json")
    duty_words_trans_json(dutywords_txt="A02.txt",dutywords_json="A02.json")
    duty_words_trans_json(dutywords_txt="A03.txt",dutywords_json="A03.json")
    duty_words_trans_json(dutywords_txt="A04.txt",dutywords_json="A04.json")
    duty_words_trans_json(dutywords_txt="A05.txt",dutywords_json="A05.json")
    duty_words_trans_json(dutywords_txt="B01.txt",dutywords_json="B01.json")
    duty_words_trans_json(dutywords_txt="B02.txt",dutywords_json="B02.json")
    duty_words_trans_json(dutywords_txt="B03.txt",dutywords_json="B03.json")
    duty_words_trans_json(dutywords_txt="C01.txt",dutywords_json="C01.json")
    duty_words_trans_json(dutywords_txt="C02.txt",dutywords_json="C02.json")
    duty_words_trans_json(dutywords_txt="D01.txt",dutywords_json="D01.json")
    
    
    # 将敏感词json文件转为txt
    # write_json_keys_to_txt("region.json","dutywords.txt")
