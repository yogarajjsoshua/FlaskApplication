
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import LoraConfig, get_peft_model 

model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], # attention layers 
lora_dropout=0.05, bias="none", task_type="CAUSAL_LM" ) 

model = get_peft_model(model, lora_config)