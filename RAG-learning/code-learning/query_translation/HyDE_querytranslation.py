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
from .. import utils
from operator import itemgetter

splits = utils.get_example_docsplits() #loads a set of agentic AI papers from the web
retriever = utils.get_retriever(splits=splits)
local_qwen_path = '/Users/bill/Documents/qwen_3.5_8B'
llm_d = utils.get_qwen_pipeline(local_qwen_path, max_new_tokens=200)


# Decomposition: reducing the user query into multiple sub-questions related to the input question.
scientific_template = """Write a scientific paper on the following user query.
User Query: {question}
Scientific paper:"""
prompt_scientific_hyde = ChatPromptTemplate.from_template(scientific_template)

disagree_template = """Write an opinion piece disagreeing with the following user query.
User Query: {question}
Scientific paper:"""
prompt_disagree_hyde = ChatPromptTemplate.from_template(disagree_template)

agree_template = """Write an opinion piece agreeing with the following user query.
User Query: {question}
Scientific paper:"""
prompt_agree_hyde = ChatPromptTemplate.from_template(agree_template)

generate_scientific_hyde = (
    prompt_scientific_hyde
    | llm_d
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.replace("Assistant: ", "").split("\n") if q.strip()])
)

generate_disagree_hyde = (
        prompt_disagree_hyde
        | llm_d
        | StrOutputParser()
)

generate_agree_hyde = (
        prompt_agree_hyde
        | llm_d
        | StrOutputParser()
)

# Run
original_question = "Why should I bother learning retrieval augmentated generation?"
hyde_question = generate_scientific_hyde.invoke({"question":original_question})
agree_question = generate_agree_hyde.invoke({"question":original_question})
disagree_question = generate_disagree_hyde.invoke({"question":original_question})
print(hyde_question)
print(agree_question)
print(disagree_question)
#
# llm_answer = utils.get_qwen_pipeline(local_qwen_path, max_new_tokens=500)
#
# answer_template = """You are a helpful AI assistant, who has been asked a question by a human user. Please use the context and related information below to answer the user's question.\n
#  ---------------------\n
#  Contextual information: {context}\n
#  Related information: {stepback_context}\n
#  ---------------------\n
#  Using the information above, please answer the following question: {question}\n
# """
#
# prompt_answer = ChatPromptTemplate.from_template(answer_template)
#
# # LangChain Expression Language uses the '|' to band together runnable objects. When you start a chain with a dictionary, Langchain automatically converts
# # it into a runnable map/runnable parallel. Hence, we need to turn the string above into a langchain runnable chat template
# rag_chain = (
#         {"context" : itemgetter("question") | retriever, #each line in here asks: how do we go from the invoked dictionary to the variable defined by the key. On this line, the variable "context".
#         "stepback_context": generate_queries_stepback | retriever,
#         "question": lambda x: x["question"]}# this line converts the invoked dictionary into a string, and then back into a dictionary.
#              | prompt_answer
#              | llm_answer
#              | StrOutputParser())
#
# answer = rag_chain.invoke({"question":original_question})
