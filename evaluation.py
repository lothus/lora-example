from langchain.evaluation import load_evaluator
# from langchain_openai import AzureChatOpenAI
from model import build_model
from dataset import DatasetDict
import getpass
import os


def evaluate(result_a,result_b,question,answer):  
    # if not os.environ.get("AZURE_OPENAI_API_KEY"):
    #     os.environ["AZURE_OPENAI_API_KEY"] =getpass.getpass("Enter your OpenAI API key: ")

    # llm = AzureChatOpenAI(
    #     azure_deployment='gpt-4-32k', 
    #     api_version='2024-12-01-preview',
    #     azure_endpoint=f"https://aisystem-openai.openai.azure.com/",
    #     temperature=1,
    #     timeout=None,
    #     max_retries=2,
    #     max_tokens=32000,
    #     top_p=0.95
    # )
    llm = build_model(model_id="meta-llama/Llama-3.1-8B-Instruct")
    evaluator = load_evaluator("labeled_pairwise_string",llm=llm)

    return evaluator.evaluate_string_pairs(
        prediction=result_a,
        prediction_b=result_b,
        input=question,
        reference=answer
    )
