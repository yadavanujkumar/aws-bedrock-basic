import boto3
import json
import base64
import os
import random

prompt_data = """
A river flowing through mount everest
"""

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Nova Canvas payload
payload = {
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt_data
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "height": 512,
        "width": 512,
        "cfgScale": 8.0,
        "seed": random.randint(0, 2147483647)
    }
}

body = json.dumps(payload)
model_id = "amazon.nova-canvas-v1:0"

try:
    print(f"Invoking model {model_id}...")
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    
    # Nova Canvas response structure: "images": ["base64..."]
    if "images" in response_body:
        base64_image = response_body["images"][0]
        image_bytes = base64.b64decode(base64_image)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "image.png")
        
        with open(output_file, "wb") as f:
            f.write(image_bytes)
        
        print(f"Image saved to {output_file}")
    else:
        print("No image found in response")
        print(response_body)

except Exception as e:
    print(f"Error invoking model: {e}")
