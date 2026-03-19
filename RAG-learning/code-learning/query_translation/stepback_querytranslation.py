import os

from langchain_core.runnables import RunnablePassthrough

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
import utils
from operator import itemgetter

splits = utils.get_example_docsplits() #loads a set of agentic AI papers from the web
retriever = utils.get_retriever(splits=splits)

# Decomposition: reducing the user query into multiple sub-questions related to the input question.
template = """You are an expert in world knowledge. Your aim is to understand the essence of a user's query, and then to generate a single-question, concise query that would retrieve a superset of the information contained within the user's query. The goal here is to generate a query that is optimal for information retreival. See below for some examples. \n
 ---------------------\n
 Example Query 1: What are some important places to visit in Manchester, England?\n 
 Optimal AI-generated Query 1 : Describe the most notable features of the city of Manchester, England.\n
 ---------------------\n
 Example Query 2: How can I improve my relationship with my cat? She never sits near me and never purrs.\n
 Optimal AI-generated Query 2 : How does someone win over a cat's trust?\n
 ---------------------\n
 Example Query 3: My houseplants keep wilting and going brown, despite my best efforts to keep them well watered and in sunlight. What am I doing wrong?
 Optimal AI-generated Query 3 : What are the essential components of houseplant health?\n
 ---------------------\n
Given those examples, generate a single-question, concise query AI-generated query that would retrieve a superset of the information for the following question: {question}\n
Optimal AI-generated Query:"""
prompt_stepback = ChatPromptTemplate.from_template(template)
local_qwen_path = '/Users/bill/Documents/qwen_3.5_8B'

llm_d = utils.get_qwen_pipeline(local_qwen_path, max_new_tokens=200)

generate_queries_stepback = (
    prompt_stepback
    | llm_d
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.replace("Assistant: ", "").split("\n") if q.strip()][0])
)

# Run
original_question = "Why should I bother learning retrieval augmentated generation?"
# stepback_question = generate_queries_stepback.invoke({"question":original_question})
# print(stepback_question)

llm_answer = utils.get_qwen_pipeline(local_qwen_path, max_new_tokens=500)

answer_template = """You are a helpful AI assistant, who has been asked a question by a human user. Please use the context and related information below to answer the user's question.\n
 ---------------------\n
 Contextual information: {context}\n
 Related information: {stepback_context}\n
 ---------------------\n
 Using the information above, please answer the following question: {question}\n
"""

prompt_answer = ChatPromptTemplate.from_template(answer_template)

# LangChain Expression Language uses the '|' to band together runnable objects. When you start a chain with a dictionary, Langchain automatically converts
# it into a runnable map/runnable parallel. Hence, we need to turn the string above into a langchain runnable chat template
rag_chain = (
        {"context" : itemgetter("question") | retriever, #each line in here asks: how do we go from the invoked dictionary to the variable defined by the key. On this line, the variable "context".
        "stepback_context": generate_queries_stepback | retriever,
        "question": lambda x: x["question"]}# this line converts the invoked dictionary into a string, and then back into a dictionary.
             | prompt_answer
             | llm_answer
             | StrOutputParser())

answer = rag_chain.invoke({"question":original_question})
