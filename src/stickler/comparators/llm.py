"""LLM comparison comparator."""

import boto3
import json
from typing import Any, Dict
from stickler.comparators.base import BaseComparator

class LLMComparator(BaseComparator):
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",  # or your preferred model
        region_name: str = "us-east-1",
        prompt_template: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.prompt_template = prompt_template or self._default_prompt_template()
        
        # Initialize Bedrock client
        session = boto3.Session()
        self.bedrock_client = session.client('bedrock-runtime')
    
    def _default_prompt_template(self) -> str:
        return """Compare these two values and determine if they are equivalent:

Value 1: {value1}
Value 2: {value2}

Return only 'true' if they are equivalent, 'false' if they are not. Only return one word: 'true' or 'false'."""
    
    def _invoke_bedrock_model(self, prompt: str) -> str:
        # Format request based on model type
        if "anthropic" in self.model_id.lower():
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        
        # Extract text based on model response format
        if "anthropic" in self.model_id.lower():
            return response_body['content'][0]['text']
        
        return response_body.get('completion', '')
    
    def compare(self, value1: Any, value2: Any) -> bool:
        # Handle None values
        if value1 is None and value2 is None:
            return 1.0
        elif value1 is None or value2 is None:
            return 0.0

        # Format the prompt with your values
        formatted_prompt = self.prompt_template.format(
            value1=str(value1),
            value2=str(value2)
        )
        
        try:
            # Get LLM response
            response = self._invoke_bedrock_model(formatted_prompt)
            # Parse response to boolean
            response_lower = response.strip().lower()
            if 'true' in response_lower:
                return 1.0
            else:
                return 0.0
            
        except Exception as e:
            # Handle errors appropriately
            print(f"Error in Bedrock comparison: {e}")
            return 0.0
    
    def get_comparison_details(self, value1: Any, value2: Any) -> Dict[str, Any]:
        formatted_prompt = self.prompt_template.format(
            value1=str(value1),
            value2=str(value2)
        )
        
        try:
            response = self._invoke_bedrock_model(formatted_prompt)
            return {
                "prompt": formatted_prompt,
                "llm_response": response,
                "model_id": self.model_id,
                "comparison_result": self.compare(value1, value2)
            }
        except Exception as e:
            return {
                "error": str(e),
                "comparison_result": False
            }