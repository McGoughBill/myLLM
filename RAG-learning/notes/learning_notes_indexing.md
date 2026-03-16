*Learning RAG from Scratch: Indexing*

Indexing is the process of breaking down a document/set of documents into a database-like data structure that can be
accessed and queried by an LLM.

If we are using vector-represented databases, indexing is done using text embeddings, and search/selection is guided by a similarity
metric such as cosine similarity. If we are using a text-represented database, we are summarising documents/sections of documents
automatically using text (likely, using an LLM). This is more interpretable, but less efficient. 

Vector indexing does not have to just be embeddings from LLMs/tokenizers. Bag-of-word indexing (each index in a vector represents a word,
each location tallies the frequency of the word) is also possible, and is more interpretable too. These vectors tend to be massive and sparse,
whereas LLM-based embeddings are dense and relatively low-dimensional, whilst being completely uninterpretable.

Idea: take a document, split it into overlapping chunks, represent each chunk with a vector that summarises semantic meaning.
Then, questions can be embedded in a similar way. Then, similarity can be compared. This seems imperfect - so much semantic meaning
is tied to the fact we are actually asking a question to begin with. Could we refine this by finding the average 'question vector'
and subtracting this from the query vector, to get a more 'pure' representation of the question? This is just an idea, but it seems like it could be useful.

We can also fire off multiple queries, to get multiple answers, and then use an LLM to summarise these answers. This is a bit like the 'chain of thought' prompting technique, but applied to retrieval instead of reasoning.

See python script 'indexing_and_retrieval.py' for simple example using my CV!
