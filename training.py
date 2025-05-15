from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from dataset import DatasetDict

def formatting_func_mistral(example):
    prompt = str(example['prompt'])
    completion = str(example['completion'])
    text = f"<s>[INST] {prompt} [/INST] {completion}</s>"
    return text

def train(ds:DatasetDict,model,tokenizer,save_model_path:str="./adapter"):
    training_arguments = SFTConfig(
        output_dir=save_model_path,                   
        num_train_epochs=1,                       
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,            
        gradient_accumulation_steps=1,           
        gradient_checkpointing=True,             
        optim="paged_adamw_32bit",
        save_steps=1000,
        logging_steps=50,                        
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3, 
        max_steps=-1,
        warmup_ratio=0.03, 
        lr_scheduler_type="constant",
        packing=True,
        dataset_num_proc=16,
        # pad_token=tokenizer.eos_token,
        # max_seq_length=4096
        # eos_token=tokenizer.eos_token
    )

    response_template = "[/INST]"

    data_collator=DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=ds['train'],
            eval_dataset=ds['valid'],
            # peft_config=lora_config,
            data_collator=data_collator,
            formatting_func=formatting_func_mistral
    )

    trainer.train()
    trainer.save_model()


