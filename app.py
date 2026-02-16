import json
import os
import sys
import boto3
import streamlit as st
import base64
import random

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat

## Data Ingestion

import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain_community.vectorstores import FAISS

## LLm Models
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


## Data ingestion
def data_ingestion(uploaded_files=None):
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())
            finally:
                # Ensure the temporary file is deleted
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        # Fallback to local data directory if no files uploaded
        if os.path.exists("data"):
            loader = PyPDFDirectoryLoader("data")
            documents = loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    try:
        vectorstore_faiss = FAISS.from_documents(
            docs,
            bedrock_embeddings
        )
        vectorstore_faiss.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"❌ **Vector Store Error**: {str(e)}")
        if "AccessDeniedException" in str(e):
            st.info("💡 **Tip**: Ensure 'Titan Text Embeddings' access is granted in your AWS Bedrock console for the us-east-1 region.")
        return False

def get_mistral_llm():
    ##create the Mistral Model
    llm=Bedrock(model_id="mistral.mistral-large-2402-v1:0",client=bedrock,
                model_kwargs={'max_tokens':512})
    
    return llm

def get_llama2_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

def get_image_response(prompt_content):
    # Image generation might require us-east-1
    bedrock_image = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    
    payload = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt_content
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
    model_id = "amazon.titan-image-generator-v2:0"

    response = bedrock_image.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    base64_image = response_body["images"][0]
    image_bytes = base64.b64decode(base64_image)
    return image_bytes

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer=qa({"query":query})
        return answer['result']
    except Exception as e:
        error_msg = str(e)
        if "ResourceNotFoundException" in error_msg and "Anthropic" in error_msg:
            return "❌ **Model Access Error**: You need to submit the 'Anthropic use case details' form in your AWS Bedrock console (Model access section) before using Claude."
        elif "ValidationException" in error_msg and "content filters" in error_msg:
            return "⚠️ **Content Filter Blocked**: Your prompt or the retrieved context was flagged by AWS content filters. Please try rephrasing."
        else:
            return f"❌ **An error occurred**: {error_msg}"


def main():
    st.set_page_config("Bedrock Multi-Tool")
    
    st.header("Bedrock Multi-Tool: Chat & Image💁")

    tab1, tab2 = st.tabs(["Chat PDF", "Image Generation"])

    with tab1:
        user_question = st.text_input("Ask a Question from the PDF Files")

        with st.sidebar:
            st.title("Upload PDFs & Create Vector Store:")
            uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
            
            if st.button("Vectors Update"):
                if not uploaded_files:
                    st.warning("Please upload at least one PDF file first.")
                else:
                    with st.spinner("Processing..."):
                        docs = data_ingestion(uploaded_files)
                        if docs:
                            if get_vector_store(docs):
                                st.success("Vector Store Created Successfully!")
                        else:
                            st.error("No text could be extracted from the uploaded PDFs.")

        if st.button("Mistral Output"):
            if not user_question:
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Processing..."):
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm=get_mistral_llm()
                    
                    #faiss_index = get_vector_store(docs)
                    st.write(get_response_llm(llm,faiss_index,user_question))
                    st.success("Done")

        if st.button("Llama2 Output"):
            if not user_question:
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Processing..."):
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm=get_llama2_llm()
                    
                    #faiss_index = get_vector_store(docs)
                    st.write(get_response_llm(llm,faiss_index,user_question))
                    st.success("Done")

    with tab2:
        st.header("Image Generation using Nova Canvas")
        image_prompt = st.text_input("Enter your image prompt")
        
        if st.button("Generate Image"):
            if not image_prompt:
                st.warning("Please enter an image prompt first.")
            else:
                with st.spinner("Generating..."):
                    try:
                        image_bytes = get_image_response(image_prompt)
                        st.image(image_bytes)
                        st.success("Generated!")
                    except Exception as e:
                        error_msg = str(e)
                        if "ValidationException" in error_msg and "content filters" in error_msg:
                            st.error("⚠️ **Content Filter Blocked**: Your prompt was flagged by AWS content filters. Please try rephrasing.")
                        elif "ModelNotAllowedException" in error_msg or "AccessDeniedException" in error_msg:
                            st.error("❌ **Access Denied**: Ensure you have granted access to 'Titan Image Generator V2' in the AWS Bedrock console for the us-east-1 region.")
                        else:
                            st.error(f"❌ **Generation Failed**: {error_msg}")

if __name__ == "__main__":
    main()