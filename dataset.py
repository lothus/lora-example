from datasets import load_dataset, DatasetDict, Dataset

def create_dataset(name:str="lmarena-ai/arena-human-preference-55k") -> DatasetDict:
    ds = load_dataset(name,trust_remote_code=True)
    # Create proper answer set for training
    df = ds['train'].to_pandas()
    df['completion'] = df.apply(lambda row: row['response_b'] if row['winner_model_b'] == 1 else row['response_a'], axis=1)
    ds['train'] = Dataset.from_pandas(df)
    #Create Test, Validation & Training set
    ds_test = ds['train'].train_test_split(0.1)
    ds_eval = ds_test['test'].train_test_split(0.5)
    #Create to proper dictionnary
    ds = DatasetDict({
        'train': ds_test['train'],
        'test': ds_eval['test'],
        'valid': ds_eval['train']
    })

    #Verify for empty string or missing information
    ds['train'] = ds['train'].filter(lambda example: example['completion'] is not None and example['completion'] != "")
    ds['test'] = ds['test'].filter(lambda example: example['completion'] is not None and example['completion'] != "")
    ds['valid'] = ds['valid'].filter(lambda example: example['completion'] is not None and example['completion'] != "")

    ds['train'] = ds['train'].filter(lambda example: example['prompt'] is not None and example['prompt'] != "")
    ds['test'] = ds['test'].filter(lambda example: example['prompt'] is not None and example['prompt'] != "")
    ds['    '] = ds['valid'].filter(lambda example: example['prompt'] is not None and example['prompt'] != "")
    # After filtering, you would proceed with creating ds_formatted
    ds = format_for_training(ds)
    return ds

def format_for_training(ds:DatasetDict) ->DatasetDict:
    training_set = list()
    for row in ds['train']:
        training_set.append(to_instruct(row))
    testing_set = list()
    for row in ds['test']:
        testing_set.append(to_instruct(row))
    validation_set =  list()
    for row in ds['valid']:
        validation_set.append(to_instruct(row))
    #Return proper training dataset
    ds_formatted = DatasetDict({
        'train': Dataset.from_list(training_set),
        'test': Dataset.from_list(testing_set),
        'valid': Dataset.from_list(validation_set)
    })
    ds_formatted
    return ds_formatted


def to_instruct(row:dict):
    if len(row['prompt']) < 1 or row['prompt'] =="":
        print(row['prompt'])
    if len(row['completion']) < 1 or row['completion'] == "":
        print(row['completion'])

    return {"prompt": f"{row['prompt']}", "completion": f"{row['completion']}"}