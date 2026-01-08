from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets, Features, Value

import os
import scipy
import torch
from datasets import load_dataset, concatenate_datasets, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    XGLMForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification,
)


def load_model(model_name):
    if model_name == "xglm-7.5B-pretrained":
        model = XGLMForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "Qwen" in  model_name:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,  # 优化内存
                device_map=None  # 明确禁用 device_map
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError("Unknown model_name")
    return model, tokenizer



def dict_to_torch_tensors(data_dict, dtype=torch.long):
    """
    Convert a dictionary with list values to a dictionary with PyTorch tensors.
    
    Args:
        data_dict (dict): Dictionary where each value is a list.
        dtype (torch.dtype): Data type for the tensors (default: torch.long).
    
    Returns:
        dict: New dictionary with the same keys but values as torch tensors.
    """
    tensor_dict = {}
    for key, value in data_dict.items():
        if not isinstance(value, list):
            raise ValueError(f"Value for key '{key}' is not a list, got {type(value)}")
        tensor_dict[key] = torch.tensor(value, dtype=dtype)
    return tensor_dict

def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    inputs = examples["title"]
    targets = examples["content"]
    
    # 验证输入和目标
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        if not isinstance(inp, str) or not isinstance(tgt, str) or not inp.strip() or not tgt.strip():
            print(f"Invalid data at index {i}: input={inp}, target={tgt}")

    # 构造消息模板
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for title, content in zip(inputs, targets):
        # 构造消息
        messages = [
            {
                "role": "system",
                "content": "You are Qwen. You are a helpful assistant and expected to write a news report with the given title."
            },
            {"role": "user", "content": title}
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        tokenized_input = tokenizer(
            text,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",  # 批处理中动态填充
            return_tensors=None
        )
        
        # 编码目标（content）
        tokenized_target = tokenizer(
            content,
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        
        model_inputs["input_ids"].append(tokenized_input["input_ids"])
        model_inputs["attention_mask"].append(tokenized_input["attention_mask"])
        model_inputs["labels"].append(tokenized_target["input_ids"])
    
    return model_inputs


def fine_tune(config):
    """
    使用 Accelerate 进行模型微调，支持多 GPU 和分布式训练。
    """
    # 初始化 Accelerate
    accelerator = Accelerator()
    
    dataset_folder = config["dataset_folder"]
    output_model_dir = config["output_model_dir"]
    pretrained_model_path = config["pretrained_model_path"]
    num_train_epochs = config["num_train_epochs"]
    per_device_train_batch_size = config["per_device_train_batch_size"]
    per_device_eval_batch_size = config["per_device_eval_batch_size"]
    max_source_length = config["max_source_length"]
    max_target_length = config["max_target_length"]

    # 加载 tokenizer 和模型
    try:
        model, tokenizer = load_model(pretrained_model_path)
    except Exception as e:
        raise ValueError(f"加载模型或 tokenizer 失败: {e}")

    # 获取 dataset_folder 下的所有 .jsonl 文件
    dataset_files = [
        os.path.join(dataset_folder, f)
        for f in os.listdir(dataset_folder)
        if f.endswith(".jsonl")
    ]
    if not dataset_files:
        raise ValueError(f"在 {dataset_folder} 中没有找到 .jsonl 文件")

    # 合并不同语言的数据
    datasets_list = []
    for file in dataset_files:
        try:
            lang_code = os.path.basename(file).split("_")[0]
            ds = load_dataset(
                "json",
                data_files=file,
                split="train",
                features=Features({
                    "title": Value("string"),
                    "content": Value("string"),
                    "publish_time": Value("string"),
                    "url": Value("string"),
                })
            )
            ds = ds.map(lambda example: {**example, "lang": lang_code})
            
            # Clean dataset: Remove invalid rows
            ds = ds.filter(
                lambda x: (
                    x["title"] is not None
                    and isinstance(x["title"], str)
                    and len(x["title"].strip()) > 0
                    and x["content"] is not None
                    and isinstance(x["content"], str)
                    and len(x["content"].strip()) > 0
                    and not any(c in x["title"] + x["content"] for c in ['\ufffd', '\uFFFD'])
                )
            )
            datasets_list.append(ds)
        except Exception as e:
            accelerator.print(f"警告: 加载或处理文件 {file} 时出错: {e}")
            continue

    if not datasets_list:
        raise ValueError("没有成功加载任何数据集")

    full_dataset = concatenate_datasets(datasets_list).shuffle(seed=42)
    accelerator.print(f"数据集长度：{len(full_dataset)}")
    accelerator.print(f"数据集列名：{full_dataset.column_names}")

    # Tokenize dataset
    try:
        tokenized_dataset = full_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, max_source_length, max_target_length),
            batched=True,
            num_proc=16,
            remove_columns=full_dataset.column_names,
        )
    except Exception as e:
        raise ValueError(f"数据集 tokenization 失败: {e}")

    # 划分训练/验证集
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # accelerator.print(train_dataset[0])

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 使用 Accelerate 准备数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    # 使用 Accelerate 准备模型和优化器
    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader
    )

    training_args = TrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_model_dir, "logs"),
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        bf16=True,  # 启用 BF16
        fp16=False,  # 禁用 FP16
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["none"],
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 使用 Accelerate 准备 Trainer
    trainer = accelerator.prepare(trainer)

    accelerator.print("开始微调，请耐心等待...")
    try:
        trainer.train()
        # 保存模型
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_model_dir)
            tokenizer.save_pretrained(output_model_dir)
        accelerator.print(f"微调结束，模型保存在：{output_model_dir}")
    except Exception as e:
        accelerator.print(f"微调过程中出错: {e}")
        raise

    return model, tokenizer

def generate(model, tokenizer, promt, max_target_length = 512):
    messages = [
        {
            "role": "system",
            "content": "You are Qwen. You are a helpful assistant and expected to write a 300 words news report with the given topic."
        },
        {"role": "user", "content": promt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_target_length
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

import os
import torch

# 导入 Hugging Face Transformers 和相关模块
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    pipeline,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, concatenate_datasets, Features, Value
from transformers import AutoTokenizer, XGLMForCausalLM
# 导入 Argo Translate 模块
import argostranslate.translate
API_KEY = "your-key" # 500k
# -------------------------------
# 使用 Argo Translate 加载翻译流水线（支持 4 种目标语言）
# -------------------------------
from google.cloud import translate_v2 as translate

import requests

def translate_text(api_key: str, text: str, target_lang: str, source_lang: str = None) -> str:
    """
    使用 Google 翻译 API 翻译文本。

    参数:
        api_key (str): Google Cloud 的 API 密钥。
        text (str): 要翻译的文本。
        target_lang (str): 目标语言的代码（例如 'zh' 表示中文，'en' 表示英文）。
        source_lang (str, 可选): 源语言代码（不填则自动检测）。
    
    返回:
        str: 翻译后的文本。
    """
    url = "https://translation.googleapis.com/language/translate/v2"
    
    params = {
        'q': text,
        'target': target_lang,
        'key': api_key
    }

    if source_lang:
        params['source'] = source_lang

    try:
        response = requests.post(url, data=params)
        response.raise_for_status()
        result = response.json()
        return result['data']['translations'][0]['translatedText']
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    
    return ""


# -------------------------------
# Pipeline 函数
# -------------------------------
def pipeline_fn(config, query, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """
    完整的流水线：
      1. 将英文 query 翻译成 4 个目标语言，同时保留原英文；
      2. 利用 fine-tuned 模型生成各语言的回复；
      3. 将回复回译成英文；
      4. 对回译结果进行英文情感分析。
    """
    # 0. 初始化模型与设备
    finetuned_model_dir = config["infer_model"]

    model,tokenizer = load_model(finetuned_model_dir)
    
    model = model.to(device)
    language_list = ["zh", "ja", "fr", "es"]
    queries = {}
    print(">> start translation ... ")
    # 3. 翻译英文 query 到其他语言
    for lang in language_list:
        queries[lang] = translate_text(API_KEY, query, lang)

    # 保留原始英文
    queries["en"] = query
    print(">> start generation ... ")
    # 4. 使用 fine-tuned 模型生成回复
    responses = {}
    for lang, q in queries.items():
        input_text = q
        responses[lang] = generate(model, tokenizer, input_text, max_target_length=config["max_target_length"])
    print(">> start translation back ... ")
    # 5. 回译所有回复为英文（仅对非英文部分）
    back_translations = {}
    for lang in language_list:
        back_translations[lang] = translate_text(API_KEY, responses[lang], "en")
    
    
    # 英文回复无需回译
    back_translations["en"] = responses["en"]
    print(">> start analysing ... ")
    # 6. 情感分析
    # sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # evaluations = {}
    # for lang, resp in back_translations.items():
    #     evaluations[lang] = sentiment_analyzer(resp)[0]
    sentiment_analyzer_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(sentiment_analyzer_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(sentiment_analyzer_model_name)

    # Define label mapping
    labels = ['negative', 'neutral', 'positive']

    def get_sentiment_scores(text, max_input_length):
        # Tokenize and get model output
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length = max_input_length)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits[0].numpy()
        scores = scipy.special.softmax(scores)  # Convert to probabilities
        return dict(zip(labels, scores))

    evaluations = {}
    for lang, resp in back_translations.items():
        evaluations[lang] = get_sentiment_scores(resp, config["max_target_length"])

    return {
        "translated_queries": queries,
        "responses": responses,
        "back_translations": back_translations,
        "evaluations": evaluations
    }
    

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import defaultdict


import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parallel_inference(GPU_NUM, TOPIC_IDX, EXAMPLE_IDX):
    generate_times = 25
    config = {
        "dataset_folder": "./dataset",
        "output_model_dir": "Qwen2.5-3B-Instruct-Fine-Tune",
        "pretrained_model_path": "Qwen2.5-3B-Instruct",
        "infer_model": "Qwen2.5-3B-Instruct-Fine-Tune",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 2,
        "max_source_length": 32,
        "max_target_length": 512
    }
    
    # Load topics and examples from JSON
    with open('news_topics.json', 'r') as f:
        topics_data = json.load(f)
    topic = list(topics_data.keys())[TOPIC_IDX]
    example = topics_data[topic][EXAMPLE_IDX]

    res_list = []
    for infer_idx in range(generate_times):
        result = pipeline_fn(config, example, device = torch.device(f"cuda:{GPU_NUM}"))
        for lang in ["en", "zh", "ja", "fr", "es"]:
            res_list.append({
                "topic": topic,
                "example": example,
                "infer_idx": infer_idx,
                "language": lang,
                "prompt_in_target_lang": result["translated_queries"][lang],
                "response": result["responses"][lang],
                "back_translations": result["back_translations"][lang],
                "negative": result["evaluations"][lang]["negative"],
                "neutral": result["evaluations"][lang]["neutral"],
                "positive": result["evaluations"][lang]["positive"]
            })
    df = pd.DataFrame(data=res_list)
    df.to_pickle(os.path.join('result',f'{TOPIC_IDX}-{EXAMPLE_IDX}.pkl'))

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Inference Script")
    parser.add_argument("--gpu_num", type=int, default=0, help="GPU number to use")
    parser.add_argument("--topic_idx", type=int, default=0, help="Topic index to use")
    parser.add_argument("--example_idx", type=int, default=0, help="Example index to use")
    args = parser.parse_args()
    parallel_inference(args.gpu_num, args.topic_idx, args.example_idx)

