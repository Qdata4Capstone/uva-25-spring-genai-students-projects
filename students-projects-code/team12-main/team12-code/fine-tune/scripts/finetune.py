#!/usr/bin/env python
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from src.prompt_builder import build_training_prompt, AgentType, load_safety_categories
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def evaluate_and_print(model, tokenizer, eval_dataset, max_samples=5):
    model.eval()
    for i in range(min(max_samples, len(eval_dataset))):
        full_ids = eval_dataset[i]["input_ids"]
        full_ids = full_ids.tolist() if hasattr(full_ids, "tolist") else full_ids

        agent_token_ids = tokenizer("[AGENT]", add_special_tokens=False)["input_ids"]
        for j in range(len(full_ids) - len(agent_token_ids)):
            if full_ids[j:j+len(agent_token_ids)] == agent_token_ids:
                prompt_ids = full_ids[:j+len(agent_token_ids)]
                break
        else:
            prompt_ids = full_ids  # fallback

        input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=30,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        #print(f"\n=== Example {i} ===")
        #print("Input:\n", tokenizer.decode(input_ids[0], skip_special_tokens=True))
        #print("Output:\n", generated_text)



def load_and_preprocess_data(train_file: str, test_file: str, tokenizer):

    dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})
    
    def preprocess_example(example):
        try:
            user_text = example["conversation"][0]["content"][0]["text"]
            agent_text = example["conversation"][1]["content"][0]["text"]
        except (KeyError, IndexError):
            logger.error("Malformed conversation: %s", example)
            return example
        
        safety_categories = load_safety_categories()
        full_prompt = build_training_prompt(
            user_text=user_text,
            agent_text=agent_text,
            categories=safety_categories,
            category_short_name_prefix="S",
            with_policy=True
        )
        print("FULL PROMPT",full_prompt)
        tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=1024)
        input_ids = tokenized["input_ids"]
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

        print("[DEBUG] Prompt token length:", len(input_ids))
        print("[DEBUG] Last tokens:", tokenizer.convert_ids_to_tokens(input_ids[-10:]))
        print("[DEBUG] Decoded tail (last 100 chars):", decoded[-100:])

        agent_delim = "[AGENT]" #
        delim_ids = tokenizer(agent_delim, add_special_tokens=False)["input_ids"]

        start_index = -1
        for i in range(len(input_ids) - len(delim_ids) + 1):
            if input_ids[i:i+len(delim_ids)] == delim_ids:
                start_index = i + len(delim_ids)
                break

        if start_index == -1:
            logger.error("Agent delimiter not found in prompt: %s", full_prompt)
            labels = input_ids.copy()
        else:
            labels = [-100] * start_index + input_ids[start_index:]
            labels = labels[:len(input_ids)]

        eos_token_id = tokenizer.eos_token_id
        if labels[-1] != eos_token_id:
            labels.append(eos_token_id)
        # Adjust labels length to match input_ids length
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))

        example["input_ids"] = input_ids
        example["labels"] = labels
        return example

    dataset = dataset.map(preprocess_example)
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["input_ids", "labels"]])
    dataset.set_format("torch")
    return dataset

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")


def build_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if "[AGENT]" not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[AGENT]"]})
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        # device_map="auto" # should be removed when using accelerate
    )

    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model vocab size: {model.config.vocab_size}, Tokenizer length: {len(tokenizer)}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model, tokenizer

def get_training_args(config):
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        fp16=config.use_fp16,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
        ddp_find_unused_parameters=False,
    )
    return training_args

def main():
    from configs.finetune_config import FinetuneConfig
    config = FinetuneConfig()
    model, tokenizer = build_model(config)
    dataset = load_and_preprocess_data(config.train_file, config.test_file, tokenizer)
    training_args = get_training_args(config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
       
    )
    logger.info("Starting training...")
    trainer.train()
    evaluate_and_print(model, tokenizer, dataset["test"], max_samples=5)
    trainer.save_model()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
