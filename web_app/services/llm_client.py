
import boto3
from utils.config import LLM_MODELS

class LLMClient:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_config = LLM_MODELS[model_name]
        self.boto3_config = self.model_config["boto3_config"]
        self.client = boto3.client("sagemaker", **self.boto3_config)

    def invoke_model(self, input_data):
        response = self.client.invoke_endpoint(
            EndpointName=self.model_config["aws_bedrock_model_id"],
            Body=input_data
        )
        return response