from dataset import create_dataset
from model import build_model
from training import train
from getpass import  getpass
from prompt import  MyModel
from evaluation import evaluate

import os
import sys
import csv
import pandas as pd

def train():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    #Get model and dataset
    ds = create_dataset()
    model,tokenizer = build_model(model_id)
    #train model
    train(ds,model,tokenizer)

def inference():
    og_model = MyModel(model_id)
    lora_model = MyModel(model_id,lora=True)
    answers = []
    test = 50
    for row in ds['valid']:
        og_answer = og_model.call(row['prompt'])
        lora_answer = lora_model.call(row['prompt'])
        answer = {
            'prompt' : row['prompt'],
            'answer_a': str(og_answer),
            'answer_b': str(lora_answer),
            'reference_answer': row['completion']
        }
        
        answers.append(answer)
        if len(answers) == test:
            break
    headers = ['prompt','answer_a','answer_b','reference_answer']
    df = pd.DataFrame.from_dict(answers)
    df.to_csv('answers_to_eval.csv', index=False)

def eval():
    df = pd.read_csv('answers_to_eval.csv')
    llama_eval = MyModel("meta-llama/Llama-3.1-8B-Instruct")
    llama_eval.hfmodel
    eval_set = []
    
    for index, row in df.iterrows():
        eval_call = evaluate(llama_eval.hfmodel,row['answer_a'],row['answer_b'],row['prompt'],row['reference_answer'])
        eval_set.append(eval_call)

    df = pd.DataFrame.from_dict(eval_set)
    df.to_csv('eval.csv', index=False)

def display_results():
    df = pd.read_csv('eval.csv')
    print(df['value'].value_counts())

def display_menu(menu):
    for k, function in menu.items():
        print(k,function.__name__)

def quit():
    os.system('clear')
    print("Exiting...")
    sys.exit()

def main():
    function_names = [
        train,
        inference,
        eval,
        display_results,
        quit
    ]
    menu_items = dict(enumerate(function_names,start=1))
    while True:
        display_menu(menu_items)
        try:
            selection = int(
                input("Please enter your selection number: ")
            )
        except:
            print("please enter an existing number")
            main()
        selected_value = menu_items[selection]
        selected_value()



if __name__=="__main__":
    os.environ['HF_TOKEN'] = (getpass('Hugging face token: '))
    main()