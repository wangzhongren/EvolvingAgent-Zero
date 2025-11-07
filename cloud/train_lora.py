import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 路径配置（现在从上传的文件读取）
input_file = "dialogues_uploaded.jsonl"
output_dir = "../models"
os.makedirs(output_dir, exist_ok=True)

def load_dialogues():
    conversations = []
    if not os.path.exists(input_file):
        print(f"[错误] 未找到上传的对话文件: {input_file}")
        return None
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    conversations.append(f"User: {item['user']}\nAgent: {item['agent']}")
                except:
                    continue
    return conversations

def main():
    conversations = load_dialogues()
    if not conversations:
        print("无有效对话数据，训练终止。")
        return
    
    print(f"[训练] 加载 {len(conversations)} 条对话，准备微调...")
    
    # 加载 tokenizer 和模型（使用 phi3:mini 的 Hugging Face 对应模型）
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 构建数据集
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    dataset = Dataset.from_dict({"text": conversations})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./lora_tmp",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        optim="paged_adamw_8bit",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("[训练] 开始 LoRA 微调 (100 steps)...")
    trainer.train()
    
    # 保存适配器
    adapter_path = os.path.join(output_dir, "latest_lora.safetensors")
    model.save_pretrained(output_dir, safe_serialization=True)
    
    # 确保生成 safetensors 文件（PEFT 默认保存为 adapter_model.bin，需转换）
    from safetensors.torch import save_file
    if os.path.exists(os.path.join(output_dir, "adapter_model.bin")):
        state_dict = torch.load(os.path.join(output_dir, "adapter_model.bin"), map_location="cpu")
        save_file(state_dict, adapter_path)
        os.remove(os.path.join(output_dir, "adapter_model.bin"))
        print(f"[保存] LoRA 权重已保存至 {adapter_path}")
    else:
        print("[警告] 未找到 adapter_model.bin，可能保存格式不同")
    
    print("[完成] 训练结束！本地模型将自动更新。")

if __name__ == "__main__":
    main()