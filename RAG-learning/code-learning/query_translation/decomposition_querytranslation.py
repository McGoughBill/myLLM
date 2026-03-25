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
from code_learning import utils
from operator import itemgetter

splits = utils.get_example_docsplits() #loads a set of agentic AI papers from the web
retriever = utils.get_retriever(splits=splits)

# Decomposition: reducing the user query into multiple sub-questions related to the input question.
template = """Generates multiple sub-questions related to an input question. The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
local_qwen_path = '/Users/bill/Documents/qwen_3.5_8B'

llm_d = utils.get_qwen_pipeline(local_qwen_path, max_new_tokens=200)

generate_queries_decomposition = (
    prompt_decomposition
    | llm_d
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.replace("Assistant: ", "").split("\n") if q.strip()])
)

# Run
original_question = "How can I become an AI engineer for agentic AI?"
questions = generate_queries_decomposition.invoke({"question":original_question})
print(questions)

#we then go away, retrieve the data for these sub-queries AND ANSWER THEM as if they were stand-alone queries, and then combine all of the
#sub-queries and attendant answers into a single answer at the end. It goes something like this:

template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser


def format_qa_pair(question, answer):
     """Format Q and A pair"""

     formatted_string = ""
     formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
     return formatted_string.strip()


# llm
llm = utils.get_qwen_pipeline(local_qwen_path, max_new_tokens=500)

q_a_pairs = ""
for q in questions:
     rag_chain = (
             {"context": itemgetter("question") | retriever, # we retrieve and insert any documents that directly relate to the sub-query
              "question": itemgetter("question"), # this is the sub-query we are performing retrieval for
              "q_a_pairs": itemgetter("q_a_pairs")} #here, we insert all the q_a pairs that have been gathered so far (will be none for the first sub-query)
             | decomposition_prompt
             | llm
             | StrOutputParser())

     answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs}) # Here, we are growing our list of existing question answer pairs within our retrieval prompt
     q_a_pair = format_qa_pair(q, answer)
     q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

#now, we run the original query back with all of the retrieved answered questions
final_answer = rag_chain.invoke({"question": original_question, "q_a_pairs": q_a_pairs})
print(final_answer)