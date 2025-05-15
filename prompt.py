from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    )
import torch
from peft import PeftModel

class MyModel():
    
    def __init__(self,model_id,lora=False) -> HuggingFacePipeline:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='cuda:0',
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        if lora:
            model = PeftModel.from_pretrained(model,'adapter')
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
        self.hfmodel = HuggingFacePipeline(pipeline=pipe)

    def call(self,prompt:str):
        answer = self.hfmodel.invoke(prompt)
        # print(answer)
        return answer

