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
    tokenizer = AutoTokenizer.from_pretrained("/media/dahan/data/wl/toxicCN/chinese-roberta-wwm-ext")


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

 
def json_trans_mydata(intentionclass_json,mytrain_json):

    # 打开json文件
    with open(intentionclass_json, "r", encoding="utf-8") as f:
        intention_class_train = json.load(f)
    
    train_list = []
    # 定义正则表达式模式
    pattern = r'# 用户问题\n(.*?)\n'
    for per in intention_class_train:
        instruction = re.search(pattern, per['instruction'])
        label = per['output']
        if instruction:
            extracted_text = instruction.group(1).strip()
            per_dict = {}
            per_dict["content"] = extracted_text
            # 现在构建的数据集只有二分类
            # 有害问题
            if label == "3":
                per_dict["toxic_one_hot"] = [0,1]
                per_dict["toxic_type_one_hot"] = [0,1]
                per_dict["expression_one_hot"] = [0, 1, 0]
                per_dict["target"] = [0,1,0,0,0]
            # 无害问题
            else:
                per_dict["toxic_one_hot"] = [1,0]
                per_dict["toxic_type_one_hot"] = [0,0]
                per_dict["expression_one_hot"] = [1, 0, 0]
                per_dict["target"] = [0,0,0,0,0]
            train_list.append(per_dict)
    
    # 将训练数据保存为JSON文件
    with open(mytrain_json, "w", encoding="utf-8") as f:
        json.dump(train_list, f, ensure_ascii=False,indent=4)
         
    return

def excel_trans_mydata(excel_data1,excel_data2,json_data):
    # 读取Excel文件
    df1 = pd.read_excel(excel_data1)
    df2 = pd.read_excel(excel_data2)
    # 获取指定列的数据并转换为list
    data_list1 = df1['question'].tolist()
    data_list2 = df2['question'].tolist()
    result_list = []
    
    # 无害问题
    for per_data in data_list1:
        per_dict = {}
        per_dict["content"] = per_data
        per_dict["toxic_one_hot"] = [1,0]
        per_dict["toxic_type_one_hot"] = [0,0]
        per_dict["expression_one_hot"] = [1, 0, 0]
        per_dict["target"] = [0,0,0,0,0]
        result_list.append(per_dict)
    
    # 有害问题
    for per_data in data_list2:
        per_dict = {}
        per_dict["content"] = per_data
        per_dict["toxic_one_hot"] = [0,1]
        per_dict["toxic_type_one_hot"] = [0,1]
        per_dict["expression_one_hot"] = [0, 1, 0]
        per_dict["target"] = [0,1,0,0,0]
        result_list.append(per_dict)    

    
    # 将数据保存为JSON文件
    with open(json_data, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)
    
    return
    
def json_to_txt(json_file,txt_file):

    # 打开指定的json文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将每一项的content和toxic_one_hot读取出来，按行写入txt文件中
    with open(txt_file, 'w', encoding='utf-8') as f:
        for item in data:
            if item['toxic_one_hot'] == [0,1]:
                toxic = 1
            elif item['toxic_one_hot'] == [1,0]:
                toxic = 0
            f.write(item['content'] +"\t"+str(toxic)+"\n")

import pandas as pd
import json
import csv

def csv_to_json(csv_file, json_file):
    ## 对于 pandas 1.3.0 及更高版本
    try:
        df = pd.read_csv(csv_file, encoding="utf-8", delimiter='\t', quoting=csv.QUOTE_NONE, header=None, names=['text', 'label'])
        # 或者 on_bad_lines='raise' 来直接报错
    except Exception as e:
        print(f"读取CSV时出错: {e}")
    result_list = []
    print("总长度为：",len(df))
    for _, row in df.iterrows():
        per_dict = {}
        per_dict["content"] = row["text"]
        if row["label"] == 0:
            per_dict["toxic_one_hot"] = [1, 0]
            per_dict["toxic_type_one_hot"] = [0,0]
            per_dict["expression_one_hot"] = [1, 0, 0]
            per_dict["target"] = [0,0,0,0,0]
        elif row["label"] == 1:
            per_dict["toxic_one_hot"] = [0,1]
            per_dict["toxic_type_one_hot"] = [0,1]
            per_dict["expression_one_hot"] = [0, 1, 0]
            per_dict["target"] = [0,1,0,0,0]
        else:
            continue
        result_list.append(per_dict)
    # 将数据保存为JSON文件
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)
        
# 合并两个json内容为一个文件
def merge_json_files(json_file1_path, json_file2_path, output_file_path):
    # 读取第一个JSON文件
    with open(json_file1_path, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    # 读取第二个JSON文件
    with open(json_file2_path, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)

    # 合并内容
    if isinstance(data1, dict) and isinstance(data2, dict):
        # 如果都是字典，使用update合并
        data1.update(data2)
    elif isinstance(data1, list) and isinstance(data2, list):
        # 如果都是列表，直接连接
        data1.extend(data2)
    else:
        raise ValueError("两个JSON文件的内容类型不一致，无法合并。")

    # 将合并后的内容写入新的JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(data1, output_file, ensure_ascii=False, indent=4)


def split_json_file_randomly(input_file_path: str,
                             output_file_path1: str,
                             output_file_path2: str,
                             ratio_for_file1: float = 0.95,
                             ensure_ascii: bool = False,
                             indent: int = 4) -> None:
    """
    将一个JSON文件（包含一个列表）按比例随机分割成两个JSON文件。

    参数:
    input_file_path (str): 输入的JSON文件路径。
    output_file_path1 (str): 第一个输出JSON文件的路径。
    output_file_path2 (str): 第二个输出JSON文件的路径。
    ratio_for_file1 (float): 第一个输出文件所占的比例 (0.0 到 1.0之间)。
                             第二个文件将获得 (1.0 - ratio_for_file1) 的比例。
                             默认为 0.7 (即70%的数据到第一个文件)。
    ensure_ascii (bool): 传递给 json.dump 的 ensure_ascii 参数。默认为 False (支持中文等非ASCII字符)。
    indent (int): 传递给 json.dump 的 indent 参数，用于美化输出。默认为 4。

    返回:
    None
    """
    if not (0 <= ratio_for_file1 <= 1):
        print("错误: ratio_for_file1 必须在 0.0 到 1.0 之间。")
        return

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file_path}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解码 '{input_file_path}' 中的JSON数据。请确保它是有效的JSON。")
        return
    except Exception as e:
        print(f"读取输入文件时发生未知错误: {e}")
        return

    if not isinstance(data, list):
        print("错误: 输入的JSON文件内容必须是一个列表 (list) 才能进行分割。")
        return

    total_items = len(data)
    if total_items == 0:
        print("警告: 输入的JSON列表为空。将创建两个空的JSON文件。")
        data1, data2 = [], []
    else:
        # 随机打乱数据副本
        shuffled_data = data[:]  # 创建浅拷贝以避免修改原始数据（如果原始数据在其他地方仍被使用）
        random.shuffle(shuffled_data)

        # 计算分割点
        split_point = int(total_items * ratio_for_file1)

        # 分割数据
        print("总数为：",len(shuffled_data))
        data1 = shuffled_data[:split_point]
        data2 = shuffled_data[split_point:]

    # 确保输出目录存在
    for out_path in [output_file_path1, output_file_path2]:
        output_dir = os.path.dirname(out_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"错误: 无法创建目录 '{output_dir}': {e}")
                return
    
    # 写入第一个输出文件
    try:
        with open(output_file_path1, 'w', encoding='utf-8') as f_out1:
            json.dump(data1, f_out1, ensure_ascii=ensure_ascii, indent=indent)
        print(f"成功将 {len(data1)} 条记录写入 '{output_file_path1}'")
    except IOError as e:
        print(f"错误: 无法写入文件 '{output_file_path1}': {e}")
        return
    except Exception as e:
        print(f"写入第一个输出文件时发生未知错误: {e}")
        return

    # 写入第二个输出文件
    try:
        with open(output_file_path2, 'w', encoding='utf-8') as f_out2:
            json.dump(data2, f_out2, ensure_ascii=ensure_ascii, indent=indent)
        print(f"成功将 {len(data2)} 条记录写入 '{output_file_path2}'")
    except IOError as e:
        print(f"错误: 无法写入文件 '{output_file_path2}': {e}")
        return
    except Exception as e:
        print(f"写入第二个输出文件时发生未知错误: {e}")
        return

    print(f"\n分割完成。总共 {total_items} 条记录。")
    print(f"文件1 ('{output_file_path1}'): {len(data1)} 条记录 ({len(data1)/total_items*100 if total_items else 0:.2f}%)")
    print(f"文件2 ('{output_file_path2}'): {len(data2)} 条记录 ({len(data2)/total_items*100 if total_items else 0:.2f}%)")


if __name__ == "__main__":
    # 敏感词典编码并转化为json
    #duty_words_trans_json(dutywords_txt="dutywords.txt",dutywords_json="dutywords.json")
    
    # 将敏感词json文件转为txt
    # write_json_keys_to_txt("region.json","dutywords.txt")
    
    # 将之前意图识别的训练数据转化为敏感句子识别数据
    # json_trans_mydata(intentionclass_json="intention_class_train.json",mytrain_json="mytrain.json")
    # excel_trans_mydata(excel_data1="non_toxic.xlsx",excel_data2="toxic.xlsx",json_data="mytest.json")
    # json_to_txt('mytrain.json','output.txt')
    
    # 将敏感句子打上标签并转化为csv
    # append_label_trans_csv("generate_sentences_legal2.txt","generate_sentences_legal2.csv",0)
    append_label_trans_csv("./用户问题十类别判定/10超出服务边界.txt","./用户问题十类别判定/10超出服务边界.csv","超出服务边界")
    # 手动将有毒和无毒句子放在一起，将csv按行打乱
    # shuffle_lines_in_file('my_toxicity_data.csv')
    
    # 将敏感句子csv转化为训练json数据
    # csv_to_json('my_toxicity_data.csv', 'my_toxicity_data.json')
    
    # 将敏感句子json按照比例分割为训练集和测试集
    # split_json_file_randomly(input_file_path="my_toxicity_data.json",output_file_path1="my_train_v3.json",output_file_path2="my_test_v3.json")
    
    
    
    # 将敏感句子json和意图识别训练数据json合并
    # merge_json_files("mytrain.json", "my_toxicity_data.json", "my_data_v3.json")
    
    
    