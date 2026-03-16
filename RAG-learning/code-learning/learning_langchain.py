import os
import bs4
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

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
retriever = vectorstore.as_retriever()
Question = "What should this candidate expect in terms of salary and benefits in the UK, working for a mid-sized company as a ML researcher?"

local_qwen_path = '/home/bill/Downloads/qwen'

access_token = os.getenv("HF_TOKEN")
# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    local_qwen_path,local_files_only=True,dtype="auto", device_map="auto",
    token=access_token, trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(local_qwen_path,local_files_only=True,dtype="auto", device_map="auto")


# convert huggingface model interface to langchain's Runnable interface using a wrapper

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=processor.tokenizer,
    max_new_tokens=2000, # Adjust based on how long you want the answer to be
    return_full_text=False # Ensures the model only returns the answer, not the prompt
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt_lg = ChatPromptTemplate.from_template("""
You are an assistant that helps answer questions based on the following retrieved text segments from a candidate's CV:
{context}
Based on the above segments, answer the following question:
{question}
""")

# the first argument holds all things that should be inserted into the prompt, the second part of the chain is the prompt itself,
# the third part is the LLM pipeline (i.e., not just the model, but also the tokenizer and generation parameters), and the last part
# is something that converts it all back into a string (since the output of the LLM pipeline is tokens and a dictionary with more info, but we just want the generated text).

rag_chain = ( {'context':retriever, 'question':RunnablePassthrough()} | prompt_lg | llm | StrOutputParser() )

response = rag_chain.invoke('What should this candidate expect in terms of salary and benefits in the UK, working for a mid-sized company as a ML researcher?')
print(response)