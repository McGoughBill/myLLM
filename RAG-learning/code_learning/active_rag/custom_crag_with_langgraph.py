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
    web_documents: List[str]
    relevant_local_documents: List[str]
    max_relevance: float

# instantiate the grader class
class GradeDocuments(BaseModel):
    """Binary relevance criteria for a retrieved document."""

    is_related: bool = Field(
        description="Does the document contain ANY information related to the question?"
    )
    fully_answers: bool = Field(
        description="Does the document FULLY answer the question on its own?"
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

# --- Grader prompt variants ---
GRADER_VARIANT = "binary_criteria"

GRADER_PROMPTS = {
    "binary_criteria": {
        "system": """You are a grader assessing relevance of a retrieved document to a user question. Answer two yes/no questions:
1. is_related: Does the document contain ANY information related to the question?
2. fully_answers: Does the document FULLY answer the question on its own?

Respond with JSON only: {{"is_related": <true or false>, "fully_answers": <true or false>}} /nothink""",
        "use_format_instructions": False,
    },
}

# document relevance grader
_v = GRADER_PROMPTS[GRADER_VARIANT]
_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", _v["system"]),
    ("user", "Please grade the score of the following retrieved document to my subsequent question. Here is the document: \n\n {document} \n\n Here is the question: {question}. /nothink"),
    ("assistant", "<think>\n\n</think>"),  # enforces no thinking and ensures json output
])
if _v["use_format_instructions"]:
    _grader_prompt = _grader_prompt.partial(format_instructions=parser.get_format_instructions())

retrieval_grader = (
    _grader_prompt
    | utils.apply_pipeline_qwen_model(llm, tokenizer, max_new_tokens=50)
    | parser
).with_config(run_name=f"retrieval_grader/{GRADER_VARIANT}")

# web-searcher based on slightly relevant retrieved documents
question_rewriter_informed = ( {"context": lambda x: x["context"], "question": lambda x: x["question"]}
    | ChatPromptTemplate.from_messages(
        [
        ("system", """You are an assistant generating web searches to supplement retrieved personal documents. \
The context below is extracted from someone's personal documents (e.g. a CV). \
It partially answers the user's question but lacks broader factual context. \
Your task: identify the specific domains, technologies, industries, or roles mentioned in the context, \
then write a concise web search (10 words or fewer) that finds publicly available information \
about THOSE TOPICS as they relate to the question. \
The search must be about the topic itself, not about the person. \
Do NOT rephrase the original question. Do NOT mention 'candidate' or 'person' - instead, describe the candidate by their career history and personal characteristics for example: \"deep learning engineer with phd\" or \"accountant with 3 years experience at Deloitte\" """),
        ("user", "Context:\n{context}\n\nQuestion: {question} /nothink"),
        ("assistant", "<think>\n\n</think>")
        ])
    | utils.apply_pipeline_qwen_model(llm,tokenizer, max_new_tokens=50)
    | StrOutputParser()
)

# web-searcher based on no background
question_rewriter_uninformed = ( {"question": lambda x: x["question"]}
    | ChatPromptTemplate.from_messages(
        [
        ("system", "You are an assistant for searching the web. Rephrase this question into a concise 10-or-less word web search that would gather all information for answering the user's query."),
        ("user", "Question: {question} /nothink"),
        ("assistant", "<think>\n\n</think>")
        ])
    | utils.apply_pipeline_qwen_model(llm,tokenizer, max_new_tokens=50)
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
    documents = []
    for d in retrieved_docs:
        d.metadata["source"] = "local"
        documents.append(d)

    return {"documents": documents, "question": question}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    max_relevance = state["max_relevance"]

    if max_relevance < 0.3:
        documents = state["web_documents"]
    elif max_relevance < 0.8:
        documents = ['-- Candidate information sourced from their CV --\n'] + state["relevant_local_documents"] + ['-- Contextual information from the web --\n'] + state["web_documents"]
    else:
        documents = state["relevant_local_documents"]

    # RAG generation
    generation = generation_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    max_relevance = 0
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )

        if not score.is_related:
            grade = 0.1
        elif not score.fully_answers:
            grade = 0.5
        else:
            grade = 0.9
        if max_relevance < grade:
            max_relevance = grade
        if grade > 0.3:
            print(f"---GRADE: DOCUMENT RELEVANT: {grade} (is_related={score.is_related}, fully_answers={score.fully_answers})---")
            # print(d.page_content)
            filtered_docs.append(d)
        else:
            print(f"---GRADE: DOCUMENT NOT RELEVANT: {grade}---")
            # print(d.page_content)

    return {"relevant_local_documents": filtered_docs, "question": question, "max_relevance": max_relevance}


def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    local_relevant_docs = state["relevant_local_documents"]
    max_relevance = state["max_relevance"]

    # Re-write question
    if max_relevance < 0.3:
        print("---TRANSFORM QUESTION WITH NO RELEVANT DOCS---")
        better_question = question_rewriter_uninformed.invoke({"question": question})
    else:
        print("---TRANSFORM QUESTION WITH RELEVANT DOCS---")
        better_question = question_rewriter_informed.invoke({"question": question,"context": local_relevant_docs})

    return {"question": better_question}


def web_search(state):

    print("---WEB SEARCH---")
    question = state["question"]
    local_documents = state["relevant_local_documents"]

    # Web search
    #declare the web search
    print(f"--SEARCH QUESTION: \'{question}\'")
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_documents = [Document(page_content=web_results, metadata={"source": "web"})]

    return {"relevant_local_documents": local_documents, "question": question, "web_documents": web_documents}

#conditional node
def decide_to_generate(state):

    print("---ASSESS GRADED DOCUMENTS---")
    max_relevance = state["max_relevance"]

    if max_relevance<0.8:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NO DOCUMENTS PERFECTLY ANSWER THE QUESTION, TRANSFORM QUERY---"
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
questions = [
    "Has this candidate led a team, or worked as an individual contributor?",# max relevancy = 0.9
    "Does this candidate have experience finetuning deep learning models in PyTorch?",# max relevancy = 0.9
    "What was the business impact of the candidate's work at their most recent company?", # max relevancy = 0.5
    "Given their professional skills, what salary range would be appropriate for this candidate?", # max relevancy = 0.5
    "Is this candidate likely to be comfortable in a fast-paced startup environment?", # max relevancy = 0.5
    "What is this candidate's favourite flavour of ice cream?", # max relevancy = 0.1
    "What day is it today?" # max relevancy = 0.1
]

for question in questions:
    print(f"\n{'='*60}\nQUESTION: {question}\n{'='*60}")
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
        pprint("\n---\n")
    pprint(value["generation"])