import json

# --- 配置 ---
# 请将 'input.txt' 替换为你的源文件名
input_file_path = 'illegal_questions_classify.txt'
# 请将 'output.txt' 替换为你想要保存结果的文件名
output_file_path = 'illegal_sentence_label.txt'
# --- 配置结束 ---

def transform_file(source_path, destination_path):
    """
    读取源文件，将其从 '句子\tJSON' 格式转换为 '句子\t类别' 格式，
    并写入目标文件。
    """
    print(f"开始处理文件 '{source_path}'...")
    
    category_id_map = {
        "A01": "1",
        "A02": "2",
        "A03": "3",
        "A04": "4",
        "A05": "5",
        "B01": "6",
        "B02": "7",
        "B03": "8",
        "C01": "9",
        "C02": "10",
        "C03": "11",
        "Z00": "0"
        
    }
    
    lines_processed = 0
    lines_written = 0
    
    try:
        # 使用 'with' 语句可以确保文件被安全地打开和关闭
        # 指定 encoding='utf-8' 来正确处理中文字符
        with open(source_path, 'r', encoding='utf-8') as infile, \
             open(destination_path, 'w', encoding='utf-8') as outfile:
            
            # 逐行读取输入文件
            for i, line in enumerate(infile, 1):
                # 去除行首和行尾的空白字符（如换行符）
                line = line.strip()
                if not line:
                    # 如果是空行，则跳过
                    continue
                
                lines_processed += 1
                
                try:
                    # 使用制表符(\t)分割句子和JSON字符串，只分割一次
                    sentence, json_str = line.split('\t', 1)
                    
                    # 解析JSON字符串为Python字典
                    data = json.loads(json_str)
                    
                    # 从字典中提取 'cate_id' 的值
                    category_id = data.get('cate_id')
                    
                    if category_id:
                        # 构建新的行格式：句子\t类别\n（\n是换行符）
                        new_line = f"{sentence}\t{category_id_map[category_id]}\n"
                        # 将新行写入输出文件
                        outfile.write(new_line)
                        lines_written += 1
                    else:
                        print(f"警告: 第 {i} 行的JSON中未找到 'cate_id'，已跳过。内容: {line}")

                except (ValueError, json.JSONDecodeError) as e:
                    # 如果分割失败或JSON格式错误，打印警告信息并跳过该行
                    print(f"警告: 第 {i} 行格式错误，无法解析，已跳过。错误: {e}。内容: {line}")

        print("\n处理完成！")
        print(f"总共处理了 {lines_processed} 行。")
        print(f"成功转换并写入了 {lines_written} 行到 '{destination_path}'。")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{source_path}' 未找到。请检查文件名和路径是否正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    transform_file(input_file_path, output_file_path)