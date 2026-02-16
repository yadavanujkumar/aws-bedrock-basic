# AWS Bedrock Basic Multi-Tool

A basic Streamlit-based application that leverages Amazon Bedrock for simple PDF-based RAG (Retrieval-Augmented Generation) and AI image generation.

## Features

- **Chat PDF**: Upload and query PDF documents using advanced LLMs and vector search.
  - **Vector Store**: Powered by FAISS and Amazon Titan Embeddings V2.
  - **LLMs**: Supports Anthropic Claude 3.7 Sonnet and Meta Llama 3 70B.
- **Image Generation**: Generate high-quality images from text prompts.
  - **Model**: Powered by Amazon Titan Image Generator V2.
  - **Region Support**: Automatically handles cross-region requests (Image models in `us-east-1`).

## Setup

### Prerequisites

1.  **AWS Account**: Ensure you have an AWS account with access to Amazon Bedrock.
2.  **Model Access**: Enable the following models in your AWS Bedrock console:
    - Amazon Titan Embeddings V2
    - Amazon Titan Image Generator V2
    - Anthropic Claude 3.7 Sonnet
    - Meta Llama 3 70B
3.  **AWS CLI**: Configured with credentials that have Bedrock permissions.

### Installation

1.  Clone the repository and navigate to the project directory.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

### PDF Chat Workflow
1.  Place your PDF files in a folder named `data`.
2.  Click **"Vectors Update"** in the sidebar to process the documents.
3.  Enter your question in the text input.
4.  Switch between **Claude** or **Llama** outputs to get answers.

### Image Generation Workflow
1.  Switch to the **"Image Generation"** tab.
2.  Enter a descriptive text prompt.
3.  Click **"Generate Image"** to see the AI-generated result.

## Dependencies

- `langchain` & `langchain-community`
- `boto3`
- `streamlit`
- `faiss-cpu`
- `pypdf`
- `numpy`

## Deployment

### Option 1: Streamlit Community Cloud (Easiest)
1. Push your code to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. **Manage Secrets**: In the Streamlit app settings, add your AWS credentials under "Secrets":
   ```toml
   AWS_ACCESS_KEY_ID = "your_access_key"
   AWS_SECRET_ACCESS_KEY = "your_secret_key"
   AWS_DEFAULT_REGION = "eu-west-2"
   ```
4. Streamlit will automatically pick these up for `boto3`.

### Option 2: AWS App Runner (Production-Ready)
1. **Containerize**: Create a `Dockerfile` for your app.
2. **Push to ECR**: Push your container image to Amazon Elastic Container Registry (ECR).
3. **App Runner Setup**:
   - Create a service in AWS App Runner.
   - Link it to your ECR image.
   - **IAM Role**: Assign an IAM instance role to the service with `AmazonBedrockFullAccess` (or least-privilege Bedrock permissions).
   - This avoids the need for manual secret management.

> [!IMPORTANT]
> Ensure that the IAM user or role used for deployment has **cross-region permission** to access `us-east-1` for image generation models if your default region is different.
