import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://eu.api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'myLLM'
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from code_learning import utils
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException

#we will use my CV as a base document.
pdf_location = os.path.join(os.path.dirname(__file__), '../../example_docs/CV_noL.pdf')
local_qwen_fp = '/Users/bill/Documents/qwen_3.5_9B_text' # downloaded from https://huggingface.co/Qwen/Qwen3.5-9B

### Get the local private knowledge (my CV!)
def load_pdf(pdf_location):
    # PyMuPDF is much better at reading CVs with multi-column layouts
    loader = PyMuPDFLoader(pdf_location)
    return loader.load()

documents = load_pdf(pdf_location)

### Convert this document into chunks and then parse into a vector database as a retriever
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)
split_docs = text_splitter.split_documents(documents)
retriever = utils.get_retriever(split_docs,doc_name='Bills_CV')

#### Create relevance assessment LLM
class GradeDocuments(BaseModel):
    """A score for relevance check on retrieved documents."""

    score: float = Field(
        description="Relevance of documents to the question, graded between 0 and 1."
    )

# LLM with function call
llm = utils.get_qwen_text_pipeline(local_qwen_fp,max_new_tokens=50)

# structured_llm_grader = llm.with_structured_output(GradeDocuments) ## this is only really possible with OpenAI models
parser = PydanticOutputParser(pydantic_object=GradeDocuments) ## thank you to our lord pydantic

# this pydantic parser instantiates a system prompt that is optimised to enforce an output from instruction-tuned LLMs
# then, this pydantic parser reads the output of our LLM to actually enforce this structure upon the output.
# the partial functionality later on makes sure the output-instruction is automatically passed as variable, that
#we do not need to control, so that we don't need to worry about stating this at invocation time.

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. Give a score between 0 and 1 to grade the relevance of the document to the question.
    If the document contains information allowing you to answer the question, give a score between 0.8-1.0.
    If the document gives you information that would allow you reasonably infer the answer, give a score between 0.3-0.8.
    Otherwise, give a low score between 0.0-0.3.
    
    Make sure your score output is a single number that could be formatted as a float, such as 0.4 or 0.12.
    """

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system + "\n{format_instructions} /nothink"),
        ("user", "Please grade the score of the following retrieved document to my subsequent question. Here is the document: \n\n {document} \n\n Here is the question: {question}. /nothink"),
        ("assistant","<think>/n/n</think>") #<- we uses this to enforce no thinking and ensure json output.
    ]
).partial(format_instructions=parser.get_format_instructions())

def log_llm_output(text: str) -> str:
    """Passthrough that prints the raw LLM output before parsing."""
    print(f"\n{'~'*40}\nRaw LLM output:\n{text}\n{'~'*40}\n")
    return text

retrieval_grader = grade_prompt | llm | RunnableLambda(log_llm_output) | parser

question = "Does this candidate have experience developing deep learning solutions?"

docs = retriever.invoke(question)
for i,doc in enumerate(docs):
    print(f"\n{'─' * 40}\n Document {i + 1}\n{'─' * 40}")
    print(doc.page_content)
    try:
        relevance_grade = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    except OutputParserException as err:
        relevance_grade = None
    print(f"\n{'#' * 10}\n Relevance grade: {relevance_grade}\n{'#' * 10}\n\n")
