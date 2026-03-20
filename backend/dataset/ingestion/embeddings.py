import os
from pathlib import Path
from openai import AzureOpenAI

def generate_embeddings(text):
    model_name = os.getenv("Text_embedding_model")
    azure_api_key = os.getenv("AZURE_API_KEY")
    if azure_api_key:
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        api_version = os.getenv("API_VERSION")
        if not azure_endpoint or not api_version:
            raise ValueError(
                "AZURE_ENDPOINT and API_VERSION must be set when using Azure OpenAI."
            )
        client=AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("API_VERSION"), 
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        ) 

        response = client.embeddings.create(model=model_name, input=text)
        return [item.embedding for item in response.data]
    else:
        raise ValueError("Set AZURE_API_KEY before running ingestion.")