# the dictionary acts as the agent's state and memory. the functions act as nodes that process the agent. The meta-script acts as the edges that respond
# to each function's processing and passes it on to it's next state.

#all tools and pipelines are defined outside of the graph state.

#this was inspired by the following notebook: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://eu.api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'myLLM'
from dotenv import load_dotenv
load_dotenv()

from code_learning import utils
from pprint import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults

from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START


#local filepaths
local_qwen_fp = '/Users/bill/Documents/qwen_3.5_9B_text'

#instantiate the agent's state class
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

# instantiate the grader class
class GradeDocuments(BaseModel):
    """A score for relevance check on retrieved documents."""

    score: float = Field(
        description="Relevance of documents to the question, graded between 0 and 1."
    )

##### define all tools and pipelines here #####
#define core LLM used for all generation activities
llm,tokenizer = utils.get_qwen_text_model(local_qwen_fp)

# local document retriever
retriever = utils.get_retriever(
                RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                    add_start_index=True
                    ).split_documents(
                            PyMuPDFLoader(
                                os.path.join(os.path.dirname(__file__), '../../example_docs/CV_noL.pdf')
                                ).load()
                            ),
                doc_name='Bills_CV')

# final answering machine
generation_rag_chain = ( {"context": lambda x: x["context"], "question": lambda x: x["question"]}
    | ChatPromptTemplate.from_messages(
        [
        ("system", "You are an assistant for question-answering tasks. Use the retrieved context below to answer the question. If you don't know the answer, say so."),
        ("user", "Context:\n{context}\n\nQuestion: {question} /nothink"),
        ("assistant", "<think>\n\n</think>")
        ])
    | utils.apply_pipeline_qwen_model(llm,tokenizer, max_new_tokens=512)
    | StrOutputParser()
)

# grader parser
parser = PydanticOutputParser(pydantic_object=GradeDocuments)

# document relevance grader
retrieval_grader = (ChatPromptTemplate.from_messages(
        [
        ("system", """You are a grader assessing relevance of a retrieved document to a user question. Give a score between 0 and 1 to grade the relevance of the document to the question.
        If the document contains information allowing you to answer the question, give a score between 0.8-1.0.
        If the document gives you information that would allow you reasonably infer the answer, give a score between 0.3-0.8.
        Otherwise, give a low score between 0.0-0.3.
        Make sure your score output is a single number that could be formatted as a float, such as 0.4 or 0.12. \n{format_instructions} /nothink"""
        ),
        ("user", "Please grade the score of the following retrieved document to my subsequent question. Here is the document: \n\n {document} \n\n Here is the question: {question}. /nothink"),
        ("assistant","<think>/n/n</think>") #<- we uses this to enforce no thinking and ensure json output.
        ]).partial(format_instructions=parser.get_format_instructions())
    | utils.apply_pipeline_qwen_model(llm,tokenizer, max_new_tokens=20)
    | parser)

# web-searcher based on retrieved documents
question_rewriter = ( {"context": lambda x: x["context"], "question": lambda x: x["question"]}
    | ChatPromptTemplate.from_messages(
        [
        ("system", "You are an assistant for searching the web. We have retrieved the context below for answering the user's questions. However, this context has been deemed insufficient for answering the question posed by the user. Use this retrieved context rephrase the question into a concise 5-or-less word web search that would gather all remaining information for answering the user's query."),
        ("user", "Context:\n{context}\n\nQuestion: {question} /nothink"),
        ("assistant", "<think>\n\n</think>")
        ])
    | utils.apply_pipeline_qwen_model(llm,tokenizer, max_new_tokens=20)
    | StrOutputParser()
)

#websearch tool
web_search_tool = TavilySearchResults(k=3)

#### define the nodes. nodes are functions ####

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval — tag each doc with source metadata to match web_search schema
    retrieved_docs = retriever.vectorstore.similarity_search(question, k=5)
    documents = [Document(page_content="--- Local Documents Begin ---", metadata={"source": "separator"})]
    for d in retrieved_docs:
        d.metadata["source"] = "local"
        documents.append(d)

    return {"documents": documents, "question": question}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = generation_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search_result = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.score
        if float(grade) > 0.5:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search_result = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search_result}


def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question,"context": documents})
    return {"documents": documents, "question": better_question}


def web_search(state):

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Mark the boundary between local and web documents
    documents.append(Document(page_content="--- Local Documents End | Web Results Begin ---", metadata={"source": "separator"}))

    # Web search
    #declare the web search
    print(f"--SEARCH QUESTION: \'{question}\'")
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    documents.append(Document(page_content=web_results, metadata={"source": "web"}))

    documents.append(Document(page_content="--- Web Results End ---", metadata={"source": "separator"}))

    return {"documents": documents, "question": question}

#conditional node
def decide_to_generate(state):

    print("---ASSESS GRADED DOCUMENTS---")
    web_search_result = state["web_search"]

    if web_search_result == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search_node", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# run the agent
inputs = {"question": "What are the types of agent memory?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])