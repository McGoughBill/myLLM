import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://eu.api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'myLLM'
from dotenv import load_dotenv
load_dotenv()

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utils.math import cosine_similarity
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import utils

splits = utils.get_example_docsplits() #loads a set of agentic AI papers from the web
retriever = utils.get_retriever(splits=splits)

# Two prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

#now, we embed these prompts
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True}
)
prompt_templates = [physics_template, math_template]
embedded_docs = embeddings.embed_documents(prompt_templates)

# Route question to prompt
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], embedded_docs)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)



question = """What is moment of a 2kN force around a 30cm lever?
"""
local_qwen_path = '/Users/bill/Documents/qwen_3.5_8B'
llm_generation = utils.get_qwen_pipeline(local_qwen_path, max_new_tokens=1000)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm_generation
    | StrOutputParser()
)

response = chain.invoke(question)
print(response)

#structured output requires pydantic, which requires python<3.14, so will leave this out of this script