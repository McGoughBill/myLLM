*Learning RAG from Scratch: Active-RAG*

This is where we start to move from RAG to agentic AI, or the border between the two. Here, we are using
RAG to generate responses, and another AI system to assess the quality of the responses, with a view to 
potentially letting further generation occur. The key 'agentic' aspect of this workflow is that tool-use is
gatekept by an LLM system. The initial RAG system deploys and generates an answers as a standalone tool, but 
the active "review" gatekeeping system is what makes the system agentic, as it can indefinitely (if the system
is poorly designed!) retrigger new retrieval until the answer meets criteria set by the user-agent dialogue.

Active RAG can be use to precisely determine which information is missing from an answer in order to service the user's query.
Hence, successive improvement-RAGs can be seen as a narrowing of the user's query. 
Agentic systems may even generate multiple novel queries needed to answer a user's question, and then set off parallel
RAG calls to retrieve those points of information in parallel. Then, a final answering tool can be used to sumamrise the initial
answer with all subsequently retrieved additional information.

The basic idea behind retrieving missing information, which forms the basis of agentic AI, is
corrective-RAG (or CRAG).

** CRAG **

1) Embed all documents (using summaries, or however else)
2) Then, upon receiving a query, assess the relevance of all documents (likely via cosine similarity)
3) Depending upon the relevance levels, take one of the two following actions:
    1) If any of the documents achieve a certain relevance threshold, we perform normal RAG with the system's documents
   2) Otherwise, we seek outside information via a web search or an alternative source.

The key here is the idea of 'absolute relevance' and a 'relevance threshold'.

This relevance-ranking system is the engine behind Cohere's RAG system, which is used by many software companies, including
Notion. Not only is this system fantastic at assessing quality and implementing a non-arbitrary relevance cut-off, it allows
the observability and monitoring of document utility over time.

Cohere's Rerank can be applied on top of retrieved documents. Hence, usually, the workflow is that a large number of documents are 
initially retrieved (say, 10-20), before being reranked and selected by Cohere's system. See an example at this link: https://docs.langchain.com/oss/python/integrations/retrievers/cohere-reranker#doing-reranking-with-coherererank

We are going to play around with crag in langgraph! Checkout crag_basic.py for this, and see here for the notebook this learning came from: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb.
We use tavily as an API-based search engine.


**CRAG in langgraph**

The essential promise of langgraph is that the state of our agent - that is, everything referring to what LLM has done, information it has stored,
and what it will do next - is stored in a simple dictionary (or dictionary-like object). Functions interact with this state, altering its values
and adding new key:value pairs, and a meta-script passes it along.

the dictionary acts as the agent's memory. the functions act as nodes that process the agent. The meta-script acts as the edges that respond
to each function's processing and passes it on to it's next state.

In the example script from crag_with_langgraph, the meta-script is hard-coded (i.e., the steps are somewhat deterministic; there are conditional statements,
but the code will deterministically end after n steps). We could imagine a future where the LLM's decide what step to take next, and whether previous steps need to be
revisited in light of new information and conditions.

We coded up our own custom agent in langgraph! It was great!

**Langgraph**

The key aspect of technical advancement present in our agent is the notion of a graph-like agent state, supported by the python package LangGraph. Langgraph
explicitly requires the user to define the workflow as a graph, where each node is associated with a python-defined function, and for this graph to compile.
We chose a simple graph to start out journey, obviously. All but one graph edge was defined as a normal feed-forward edge. We added a single conditional edge,
according to the presence of relevance in our retrieved documents.

Conditional edge use string-based outputs of python-defined functions to inform direction of travel - in custom_crag_with_langgraph, note how 'decide_to_generate's output
is a string, unlike the other functions.

**Adaptive RAG**

The idea of constantly checking outputs, retrieved documents, and LLM statements for relevance. This is using an LLM to identify holes in answers and actually seek to close
them actively. Think of the RAG system as having some PID controller where the answer is the output of the system, which the PID is looking to optimise.

We can unit-test sections of the graph workflow! this is quite a large personal revelation, but of course we can! So, this is making sure our retrievers are reliably bringing 
relevant documents to the user.