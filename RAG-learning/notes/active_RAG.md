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