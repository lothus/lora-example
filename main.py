from dataset import create_dataset
from model import build_model
from training import train
import os
from getpass import  getpass
from prompt import  MyModel
from evaluation import evaluate
import csv

os.environ['HF_TOKEN'] = (getpass('Hugging face token: '))
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
#Get model and dataset
ds = create_dataset()
# model,tokenizer = build_model(model_id)



#train model
# train(ds,model,tokenizer)

#Eval movel after training
#Eval model before training
og_model = MyModel(model_id)
lora_model = MyModel(model_id,lora=True)
answers = list()
for row in ds['valid']:
    og_answer = og_model.call(row['prompt'])
    lora_answer = lora_model.call(row['prompt'])
    answer = {
        'prompt' : row['prompt'],
        'og_answer': og_answer,
        'train_answer': lora_answer,
        'reference_answer': row['completion']
    }
    answers.append(answer)
headers = ['Prompt','answer_a','answer_b','reference_answer']

with open('answers_to_eval.csv', 'w') as f:
    
    # using csv.writer method from CSV package
    write = csv.writer(f)
    
    write.writerow(headers)
    write.writerows(answers)
#Run evlatuation
# eval_call = evaluate(og_answer,lora_answer,row['prompt'],row['completion'])
# print(eval_call)