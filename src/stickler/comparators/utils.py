import json

try:
    import boto3
    from botocore.config import Config

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


def generate_bedrock_embedding(
    text, model_id="amazon.titan-embed-text-v2:0", max_retries=3, region=None
):
    """
    Generate an embedding for the given text using the specified model.

    Args:
        text (str): The input text to embed.
        model_id (str): The ID of the model to use for embedding. Default is "amazon.titan-embed-text-v2:0".
        max_retries (int): The maximum number of retries for the embedding generation. Default is 3.
        region (str): The AWS region for the Bedrock client. If None, the default region is used.

    Returns:
        list: The embedding vector for the input text.

    Raises:
        Exception: If the embedding generation fails after the maximum number of retries.
    """
    if not _HAS_BOTO3:
        raise ImportError(
            "boto3 is required for Bedrock embeddings. Please install it with: pip install boto3"
        )

    if not text or not isinstance(text, str):
        # Return an empty vector for empty input
        return []

    config = Config(
        connect_timeout=10,
        read_timeout=120,
        retries={"max_attempts": max_retries, "mode": "adaptive"},
    )

    # Initialize the Bedrock client
    brt = boto3.client(
        service_name="bedrock-runtime", region_name=region, config=config
    )

    # Normalize whitespace and prepare the input text
    normalized_text = " ".join(text.split())
    if "amazon.titan-embed" in model_id:
        request_body = json.dumps({"inputText": normalized_text})
    else:
        # Default format for other models
        request_body = json.dumps({"text": normalized_text})

    try:
        response = brt.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=request_body,
        )
        response_body = json.loads(response["body"].read())

        embedding = response_body.get("embedding", [])

        return embedding
    except Exception as e:
        raise Exception(
            f"Failed to generate embedding after {max_retries} attempts: {str(e)}"
        )
