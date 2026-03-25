*Learning RAG from Scratch: MR-indexing*

The idea here is to go one step beyond indexing into representing information in different ways.

Really obvious way - representing documents as summaries within vector-databases (or embeddings of summaries)

Then there is RAPTOR. This can be thought of as a hierarchical indexing/summaries for efficient information retrieval.
Raptor stands for recursive abstractive processing for tree-organised retrieval.


**Dedicated RAPTOR deepdive**
video link:https://www.youtube.com/watch?v=jbGchdTL7d0

There was a discussion, with long-context LLMs, about whether RAG was dead. This is because, with enough tokens,
all contextual knowledge could just be inserted into the prompt. However, this is cost- and token-inefficient, and
would not be an optimal engineering solutions

Idea of building a document tree - which is the basis of raptor - is to allow efficient hierarchical navigation across enormous
amounts of offline knowledge. For example, imagine a thesis, split into chapters and then methods/techniques. IF we received a question like:
"What types of neural networks has Bill worked with in his PhD?", we would like a hierarchical method of dipping into chapters method sections to inspect
this. This requires a hierarchical structure like a document tree.

Raptor works as follows:
1) Get your raw documents
2) Embed your raw documents to get a set of embeddings
3) cluster them to find similar text chunks (Number of clusters n, or have a self-determining number of clusters)
4) Then, we abstract/summarise all information within each cluster's text chunks, summarising everything that cluster relates to
5) Then we repeat stage 2 on the abstracts generated in 4
6) then we repeat stage 3 on the embeddings from 5, with less clusters
7) we repeat 5 and 6 until we end up with one high-level document that describes all documents.

The purpose of raptor is to cover all layers of abstraction, so that raw data can be recovered but also summaries across multiple documents,
ensuring queries across all abstraction layers can be answered.

**ColBERT - a different indexing method**

An approach for tackling the information loss in embedding. during indexing and retrieval, embedding information
loss means we may miss the ability to connect correct documents to our answers. ColBERT attempts to tackle this problem.

Approach for ColBERT:

1) produce embedding for every TOKEN in document (could be thousands, or millions)
2) Then, do the same for the question (per token embedding)
3) Then, find the similarity of all query tokens to all document tokens (omg this sounds like a lot)
4) find the document that corresponds to the max similarity to each query token
5) Then, tally the score of each document (how many times was one of its tokens the max)
6) Insert the top-k documents.

Performance is extremely high for ColBERT, but latency (and raw computation!) is a real issue.


