*Learning RAG from Scratch: MR-indexing*

The idea here is to go one step beyond indexing into representing information in different ways.

Really obvious way - representing documents as summaries within vector-databases (or embeddings of summaries)

Then there is RAPTOR. This can be thought of as a hierarchical indexing/summaries for efficient information retrieval.
Raptor stands for recursive abstractive processing for tree-organised retrieval.


**Dedicated RAG deepdive**
video link:https://www.youtube.com/watch?v=jbGchdTL7d0

There was a discussion, with long-context LLMs, about whether RAG was dead. This is because, with enough tokens,
all contextual knowledge could just be inserted into the prompt. However, this is cost- and token-inefficient, and
would not be an optimal engineering solutions
