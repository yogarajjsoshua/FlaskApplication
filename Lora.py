import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 1. Load dataset (FIXED NAME)
dataset = load_dataset("databricks/databricks-dolly-15k")

# 2. Load model + tokenizer FIRST (you had this too late)
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# 3. Attach LoRA BEFORE training
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 4. Format dataset
def format_dolly(example):
    if example["context"]:
        text = f"""### Instruction:
{example['instruction']}

### Context:
{example['context']}

### Response:
{example['response']}"""
    else:
        text = f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""

    return {"text": text}

dataset = dataset["train"].map(format_dolly)

# 5. Tokenize (YOU MISSED APPLYING THIS)
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = dataset.map(tokenize, batched=True)

# 6. Data collator (IMPORTANT for causal LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. Training config
training_args = TrainingArguments(
    output_dir="./mistral-dolly-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# 9. Train
trainer.train()

# 10. Save LoRA adapters
model.save_pretrained("./lora-dolly")
tokenizer.save_pretrained("./lora-dolly")