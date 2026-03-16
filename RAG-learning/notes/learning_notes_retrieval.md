*Learning RAG from Scratch: Retrieval*

In the dummy example 'indexing_and_retrieval' script, we used a simple cosine similarity metric via a QA-trained sentence embedding model to retrieve relevant sections of a document.
Thia is the simplest functional form of retrieval.

In that script, we had previously tried using BERT or another small language model for sentence embedding, but we ran into the problem identified in the previous notes
(learning_notes_indexing.md) that questions have completely seperate semantic meaning to statements, so their embeddings would not naturally be drawn to the same
vector space. To tackle this, we changed from our BERT embedding model to a QA-model trained for this specific task. Literally, it was trained to bring questions and correct answers
into the same dense vector space over 215M questions.

Documents in similar location in semantic space should contain similar information. This is the core of vector-based retrieval.
It is a similarity-ranked search based on the latent space vectors.

Langchain has some fantastic integrations for document loading, embedding, and retrieval.

Chain of thought relates to the encouragement of LLMs to first reason about how to solve a task, before attempting to solve it.
This can be done with a simple preliminary prompt, for example "Tell me how you would solve this problem". Then, the plan gets fed
into the system prompt at the next iteration, and the LLM can use this to guide its retrieval and reasoning. 
This is a powerful technique, and can be used in conjunction with retrieval to get better results.

***My unresolved questions:***

What is not so clear to me is how to host the vector database on a server, and how to query it from an LLM. Can we use a similarity search on
the SQL server? Are there specialised vector databases that we can use?
