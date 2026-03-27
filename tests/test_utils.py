import pytest
from langchain_core.documents import Document
from code_learning import utils


# ---------------------------------------------------------------------------
# get_unique_union
# ---------------------------------------------------------------------------

def test_unique_union_deduplicates_identical_docs():
    doc = Document(page_content="hello", metadata={})
    result = utils.get_unique_union([[doc], [doc]])
    assert len(result) == 1


def test_unique_union_preserves_distinct_docs():
    doc_a = Document(page_content="hello", metadata={})
    doc_b = Document(page_content="world", metadata={})
    result = utils.get_unique_union([[doc_a], [doc_b]])
    assert len(result) == 2


def test_unique_union_empty_input():
    result = utils.get_unique_union([])
    assert result == []


def test_unique_union_returns_document_objects():
    doc = Document(page_content="hello", metadata={})
    result = utils.get_unique_union([[doc]])
    assert all(isinstance(d, Document) for d in result)


def test_unique_union_single_sublist_passthrough():
    docs = [Document(page_content=str(i), metadata={}) for i in range(3)]
    result = utils.get_unique_union([docs])
    assert len(result) == 3


# ---------------------------------------------------------------------------
# get_top_unique_documents
# ---------------------------------------------------------------------------

def test_top_unique_returns_most_frequent_first():
    common = Document(page_content="common", metadata={})
    rare = Document(page_content="rare", metadata={})
    # common appears twice, rare once
    result = utils.get_top_unique_documents([[common, rare], [common]], top_k=1)
    assert len(result) == 1
    assert result[0].page_content == "common"


def test_top_unique_respects_top_k():
    docs = [Document(page_content=str(i), metadata={}) for i in range(5)]
    result = utils.get_top_unique_documents([docs], top_k=3)
    assert len(result) == 3


def test_top_unique_top_k_larger_than_pool():
    docs = [Document(page_content=str(i), metadata={}) for i in range(2)]
    result = utils.get_top_unique_documents([docs], top_k=10)
    assert len(result) == 2


def test_top_unique_deduplicates():
    doc = Document(page_content="repeat", metadata={})
    result = utils.get_top_unique_documents([[doc], [doc], [doc]])
    assert len(result) == 1


def test_top_unique_returns_document_objects():
    doc = Document(page_content="hello", metadata={})
    result = utils.get_top_unique_documents([[doc]])
    assert all(isinstance(d, Document) for d in result)


# ---------------------------------------------------------------------------
# Functions requiring hardware / network — skipped in unit test runs
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires MPS device and downloads embedding model (~90MB)")
def test_get_retriever_returns_retriever():
    doc = Document(page_content="test content", metadata={})
    retriever = utils.get_retriever([doc], doc_name="test_collection")
    assert retriever is not None


@pytest.mark.skip(reason="requires local Qwen model files and GPU")
def test_get_qwen_text_model_loads():
    model, tokenizer = utils.get_qwen_text_model('/Users/bill/Documents/qwen_3.5_9B_text')
    assert model is not None
    assert tokenizer is not None


@pytest.mark.skip(reason="requires local Qwen model files and GPU")
def test_get_qwen_pipeline_loads():
    llm = utils.get_qwen_pipeline('/Users/bill/Documents/qwen_3.5_9B_text')
    assert llm is not None


@pytest.mark.skip(reason="requires local Qwen model files and GPU")
def test_get_qwen_text_pipeline_loads():
    llm = utils.get_qwen_text_pipeline('/Users/bill/Documents/qwen_3.5_9B_text')
    assert llm is not None


@pytest.mark.skip(reason="requires network access to lilianweng.github.io")
def test_get_example_docsplits_returns_nonempty_list():
    splits = utils.get_example_docsplits()
    assert len(splits) > 0


@pytest.mark.skip(reason="requires network access to lilianweng.github.io")
def test_get_example_docsplits_returns_document_objects():
    splits = utils.get_example_docsplits()
    assert all(isinstance(d, Document) for d in splits)