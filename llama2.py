import boto3
import json

prompt_data ="""
Act as a Gulzar poet and write a poem on the topic of rain in hindi.

"""

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

payload = {
    "prompt": prompt_data,
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}

body = json.dumps(payload)
model_id='meta.llama3-70b-instruct-v1:0'

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    contentType='application/json',
    accept='application/json'
)

response_body=json.loads(response.get('body').read())
print("Response keys:", response_body.keys())
response_text=response_body.get('generation')
print("Generation:", response_text)