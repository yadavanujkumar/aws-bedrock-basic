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
