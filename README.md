# AWS Bedrock Basic Multi-Tool

A basic Streamlit-based application that leverages Amazon Bedrock for simple PDF-based RAG (Retrieval-Augmented Generation) and AI image generation.

## Features

- **Chat PDF**: Upload your own PDF documents and query them using advanced LLMs and vector search.
  - **Dynamic Upload**: Upload multiple PDFs via the sidebar.
  - **Vector Store**: Powered by FAISS and Amazon Titan Embeddings V1.
  - **LLMs**: Supports Mistral Large and Meta Llama 3 70B.
- **Image Generation**: Generate high-quality images from text prompts.
  - **Model**: Powered by Amazon Titan Image Generator V2.
  - **Region Support**: Automatically handles cross-region requests (Image models in `us-east-1`).

## Setup

### Prerequisites

1.  **AWS Account**: Ensure you have an AWS account with access to Amazon Bedrock.
2.  **Model Access**: Enable the following models in your AWS Bedrock console:
    - Amazon Titan Embeddings G1 - Text (V1)
    - Amazon Titan Image Generator V2
    - Mistral Large (24.02)
    - Meta Llama 3 70B Instruct
3.  **AWS CLI**: Configured locally with credentials that have Bedrock permissions.

### Installation

1.  Clone the repository and navigate to the project directory.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure AWS**: Before running the application, you must configure your AWS credentials. Run the following command and follow the prompts:

    ```bash
    aws configure
    ```

## Usage

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

### PDF Chat Workflow
1.  Navigate to the **"Chat PDF"** tab.
2.  In the sidebar, click **"Browse files"** to upload one or more PDF documents.
3.  Click **"Vectors Update"** to process and index the uploaded files.
4.  Enter your question in the text input on the main page.
5.  Click either **"Mistral Output"** or **"Llama2 Output"** to get your answer.

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
3. **Manage Secrets**: In the Streamlit app settings (**Settings > Secrets**), add your AWS credentials in exactly this format:
   ```toml
   [aws]
   access_key_id = "your_access_key"
   secret_access_key = "your_secret_key"
   region = "us-east-1"
   ```
4. The app includes a **Diagnostic Status** in the sidebar to confirm if keys are loaded correctly.

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
