import os
import bs4
from langchain_community.document_loaders import PyMuPDFLoader # <-- Better PDF parser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ['USER_AGENT'] = 'my-rag-app/1.0'
hf_token = os.environ.get("HF_TOKEN")

pdf_location = '../example_docs/CV_noL.pdf'

def load_pdf(pdf_location):
    # PyMuPDF is much better at reading CVs with multi-column layouts
    loader = PyMuPDFLoader(pdf_location)
    return loader.load()

# Use a model specifically trained for Question/Answer retrieval
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Loading PDF...")
documents = load_pdf(pdf_location)

# Increased chunk size to keep section headers attached to their content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)

split_docs = text_splitter.split_documents(documents)

print("Creating vector store...")
vectorstore = Chroma.from_documents(split_docs, embeddings)
Question = "What should this candidate expect in terms of salary and benefits in the UK, working for a mid-sized company as a ML researcher?"

# You don't need to manually embed the query. similarity_search handles it.
print("\nRetrieving Documents...")
retrieved_docs = vectorstore.similarity_search(Question, k=5)

print("\nRetrieved Documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Retrieved Document {i} ---")
    print(f"Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content.strip()}...")


# now, you can feed retrieved_docs into your LLM for question answering, e.g. using a prompt like:
prompt = f"""
You are an assistant that helps answer questions based on the following retrieved text segments from a candidate's CV:
1) {retrieved_docs[0].page_content}/n
2) {retrieved_docs[1].page_content}/n
3) {retrieved_docs[2].page_content}/n
Based on the above segments, answer the following question:
{Question}
"""

# Then, pass it to Qwen.
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

local_qwen_path = '/home/bill/Downloads/qwen'

access_token = os.getenv("HF_TOKEN")
# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    local_qwen_path,local_files_only=True,dtype="auto", device_map="auto",
    token=access_token, trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(local_qwen_path,local_files_only=True,dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    token=access_token
)

inputs = inputs.to(model.device)

# Inference: Generation of the output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=500)

generated_text = processor.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Answer:")
print(generated_text)


