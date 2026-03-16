import os
import bs4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, pipeline, AutoModelForCausalLM
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.load import dumps, loads
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter



def get_retriever(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=embeddings)
    retriever = vectorstore.as_retriever()

    return retriever

def get_qwen_pipeline(location,max_new_tokens=500):
    model = Qwen3VLForConditionalGeneration.from_pretrained(location, local_files_only=True, dtype="auto",
                                                            device_map="auto")
    processor = AutoProcessor.from_pretrained(location, local_files_only=True, dtype="auto", device_map="auto")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=processor.tokenizer,
        max_new_tokens=max_new_tokens,  # Adjust based on how long you want the answer to be
        return_full_text=False  # Ensures the model only returns the answer, not the prompt
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def get_top_unique_documents(documents: list[list], top_k=5):
    """ Unique union of retrieved docs """
    # This is a really unoptimised function - we are comparing whole document sections for exact matches with


    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]

    # Counter automatically tallies frequencies
    doc_counts = Counter(flattened_docs)

    # most_common(top_k) returns a list of tuples: [(doc_string, count), ...]
    top_k_docs = [doc for doc, count in doc_counts.most_common(top_k)]

    # Deserialize and return
    return [loads(doc) for doc in top_k_docs]

def get_example_docsplits():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(blog_docs)
    return splits
