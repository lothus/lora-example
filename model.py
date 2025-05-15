import torch
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    )
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
import re

def build_model(model_id:str="mistralai/Mistral-7B-Instruct-v0.1"):
    model_id = model_id
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map='cuda:0',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )   
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    model.gradient_checkpointing_enable()

    model = get_peft_model(model,lora_configs(get_layers(model)))
    return model, tokenizer


def get_layers(model):
    pattern = r'\((\w+)\): Linear'
    linear_layers = re.findall(pattern, str(model.modules))
    target_modules = list(set(linear_layers))
    return target_modules

def lora_configs(target_modules:list,r:int=64,lora_dropout:float=0.15) ->LoraConfig:
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=lora_dropout,
        r=r,
        bias="none",
        target_modules=target_modules,
        task_type='CAUSAL_LM'
    )


def get_trained_model():
    '''
    Pulls the model with the PEFT adapter
    '''
    pass