from langchain.evaluation import load_evaluator
from dataset import DatasetDict
import getpass
import os


def evaluate(llm,result_a,result_b,question,answer):  
    evaluator = load_evaluator("labeled_pairwise_string",llm=llm)

    return evaluator.evaluate_string_pairs(
        prediction=result_a,
        prediction_b=result_b,
        input=question,
        reference=answer
    )
