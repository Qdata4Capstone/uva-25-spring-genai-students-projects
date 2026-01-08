from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets, Features, Value

import os
import torch
from datasets import load_dataset, concatenate_datasets, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    XGLMForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def load_model(model_name):
    if model_name == "xglm-7.5B-pretrained":
        model = XGLMForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_name in ["Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"]:
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


# -------------------------------
# 示例调用
# -------------------------------
import torch
if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = {
        "dataset_folder": "./dataset",  
        "output_model_dir": "Qwen2.5-3B-Instruct-Fine-Tune",
        "pretrained_model_path": "Qwen2.5-3B-Instruct", 
        "infer_model": "Qwen2.5-3B-Instruct", 
        "num_train_epochs": 8,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 2,
        "max_source_length": 128,
        "max_target_length": 512
    }
    
    # 1. 微调模型 is done
    model, tokenizer = fine_tune(config)
    
    # 2. 测试流水线
    query = "Give me a short introduction About Japan's nuclear wasterwater discharge"

    # 运行流水线
    
    result = pipeline_fn(config, query)

    print("==> 1. The original input is <==")
    for lang in ["en", "zh", "ja", "fr", "es"]:
        print(f'{lang}: {result["translated_queries"][lang]}')
    print("==> 2. The generated text is <==")
    for lang in ["en", "zh", "ja", "fr", "es"]:
        print(f'{lang}: {result["responses"][lang]}')
    print("==> 3. The generated text in English is <==")
    for lang in ["en", "zh", "ja", "fr", "es"]:
        print(f'{lang}: {result["back_translations"][lang]}')
    print("==> 4. The sentiment analyse result is <==")
    for lang in ["en", "zh", "ja", "fr", "es"]:
        print(f'{lang}: {result["evaluations"][lang]}')

        