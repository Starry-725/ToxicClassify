import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from importlib import import_module
from src.datasets import get_all_dirty_words, get_all_toxic_id, to_tensor
from src.Models import Bert_Layer, TwoLayerFFNNLayer

app = Flask(__name__)

# 全局变量
tokenizer = None
embed_model = None
model = None
config = None
all_dirty_words = None


def load_model(model_path, model_name="chinese-roberta-wwm-ext"):
    """
    加载模型和相关资源
    
    Args:
        model_path: 模型检查点路径
        model_name: 预训练模型名称
        lexicon_base_path: 词典文件所在目录
    """
    global tokenizer, embed_model, model, config, all_dirty_words
    
    # 加载配置
    x = import_module('model.Config_base')
    config = x.Config_base(model_name, "ToxiCN")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载词典
    all_dirty_words = get_all_dirty_words(config.lexicon_path)
    
    # 加载模型
    embed_model = Bert_Layer(config).to(config.device)
    model = TwoLayerFFNNLayer(config).to(config.device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=config.device)
    embed_model.load_state_dict(checkpoint['embed_model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    embed_model.eval()
    model.eval()
    
    print("模型加载完成！")


def get_preds(config, logit):
    """获取预测结果"""
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    new_results = []
    for result in results:
        result = [0] * config.num_classes
        result[int(result)] = 1
        new_results.append(result)
    return new_results


def predict_toxicity(text):
    """
    预测文本的毒性
    
    Args:
        text: 输入文本
        
    Returns:
        dict: 预测结果
    """
    # 对文本进行编码
    encoded = tokenizer(text, 
                       add_special_tokens=True,
                       max_length=config.pad_size, 
                       padding='max_length', 
                       truncation=True,
                       return_tensors='pt')
    
    # 计算toxic_ids
    text_idx = encoded['input_ids'][0].tolist()
    toxic_ids = get_all_toxic_id(config.pad_size, text_idx, all_dirty_words)
    
    # 准备批次数据
    batch = [{
        'text_idx': encoded['input_ids'][0].tolist(),
        'text_ids': encoded['token_type_ids'][0].tolist(),
        'text_mask': encoded['attention_mask'][0].tolist(),
        'toxic_ids': toxic_ids,
        'toxic': [0, 0],  # 占位符，预测时不需要真实标签
        'toxic_type': [0, 0],
        'expression': [0, 0],
        'target': [0]
    }]
    
    # 转换为模型输入格式
    args = to_tensor(batch)
    
    # 模型预测
    with torch.no_grad():
        args = {k: v.to(config.device) for k, v in args.items()}
        att_input, pooled_emb = embed_model(**args)
        outputs = model(att_input, pooled_emb)
        
        # 获取各项预测结果
        toxic_logits = outputs['toxic_pred'].cpu()
        toxic_probs = torch.sigmoid(toxic_logits).numpy()[0]
        
        if 'toxic_type_pred' in outputs:
            toxic_type_logits = outputs['toxic_type_pred'].cpu()
            toxic_type_probs = torch.sigmoid(toxic_type_logits).numpy()[0]
        else:
            toxic_type_probs = None
            
        if 'expression_pred' in outputs:
            expression_logits = outputs['expression_pred'].cpu()
            expression_probs = torch.sigmoid(expression_logits).numpy()[0]
        else:
            expression_probs = None
            
        if 'target_pred' in outputs:
            target_logits = outputs['target_pred'].cpu()
            target_probs = torch.sigmoid(target_logits).numpy()[0]
        else:
            target_probs = None
    
    # 确定主要标签
    is_toxic = toxic_probs[1] > 0.5  # 阈值为0.5
    label = "offensive" if is_toxic else "non-offensive"
    
    # 如果有毒，确定毒性类型
    toxic_types = ["一般侮辱", "LGBT歧视", "地域歧视", "性别歧视", "种族歧视"]
    detected_types = []
    
    if toxic_type_probs is not None and is_toxic:
        for i, prob in enumerate(toxic_type_probs):
            if prob > 0.5:  # 阈值可调整
                detected_types.append({
                    "type": toxic_types[i] if i < len(toxic_types) else f"Type-{i}",
                    "probability": float(prob)
                })
    
    # 返回预测结果
    result = {
        "text": text,
        "is_toxic": bool(is_toxic),
        "label": label,
        "confidence": float(toxic_probs[1] if is_toxic else toxic_probs[0]),
        "toxic_types": detected_types,
        "details": {
            "toxic_probabilities": toxic_probs.tolist(),
            "toxic_type_probabilities": toxic_type_probs.tolist() if toxic_type_probs is not None else None,
            "expression_probabilities": expression_probs.tolist() if expression_probs is not None else None,
            "target_probabilities": target_probs.tolist() if target_probs is not None else None
        }
    }
    
    return result


@app.route('/predict', methods=['POST'])
def predict_api():
    """API端点：接收文本并返回毒性预测结果"""
    # 获取请求数据
    text = request.form.get("text")
    
    if not text or not isinstance(text, str):
        return jsonify({"error": "文本必须是非空字符串"}), 400
    
    # 执行预测
    try:
        result = predict_toxicity(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"预测失败: {str(e)}"}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict_api():
    """API端点：批量预测多个文本的毒性"""
    # 获取请求数据
    data = request.get_json()
    
    # 验证输入
    if not data or 'texts' not in data:
        return jsonify({"error": "请提供texts字段"}), 400
    
    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({"error": "texts必须是字符串列表"}), 400
    
    # 执行批量预测
    try:
        results = [predict_toxicity(text) for text in texts]
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": f"批量预测失败: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "model_loaded": embed_model is not None and model is not None})


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ToxiCN毒性预测API服务')
    parser.add_argument('--model_path', type=str, 
                        default="/media/dahan/data/wl/toxicCN/ToxiCN_ex/ToxiCN/saved_dict/ckp-chinese-roberta-wwm-ext-NN_ML-80_D-0.5_B-32_E-10_Lr-1e-05_aplha-0.5-BEST.tar", 
                        help='模型路径')
    parser.add_argument('--model_name', type=str, default="chinese-roberta-wwm-ext", help='预训练模型名称')
    parser.add_argument('--lexicon_path', type=str, default="ToxiCN/lexicon/", help='词典路径')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='主机地址')
    parser.add_argument('--port', type=int, default=8858, help='端口号')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    
    args = parser.parse_args()
    
    # 加载模型
    load_model(args.model_path, args.model_name)
    
    # 启动Flask服务
    app.run(host=args.host, port=args.port, debug=args.debug)