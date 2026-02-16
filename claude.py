import boto3
import json

prompt_data = """
Act as a Christopher Nolan and write a movie script on Genertaive AI
"""

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Llama 3 payload format (using same model as llama2.py as fallback)
payload = {
    "prompt": prompt_data,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)
model_id = 'meta.llama3-70b-instruct-v1:0'

try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    response_text = response_body.get("generation")
    print(response_text)

except Exception as e:
    print(f"Error invoking model: {e}")