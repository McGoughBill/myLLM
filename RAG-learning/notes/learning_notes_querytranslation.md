*Learning RAG from Scratch: Query Translation*

This process converts a user's query into something that can be used for retrieval. Implicit in this step is that
user's queries are often in non-optimal form for retrieval, questions may be open-ended or context-dependent, or 
may be wordy/contain unnecessary infomration. Queries should be concise, containing all (and only) the information
needed to answer the request.

We can either go more abstract or less abstract. More abstract: can we use an LLM to summarise the essence of a question in a more
concise way?

Less abstract: what specific pieces of information are needed to answer the question? First, let's find those answers, and then
use those answers to construct a query. This is more of a "chain of thought" approach, where we break down the question into sub-questions, answer those, and then use those answers to construct a final query.

In vector representation, the theory behind this is that a poorly worded question/overly verbose, imprecise question will have a poorly matching 
vector representation to the information needed to answer the question. By contract, by tackling the problem from different perspectives with more directed
questioning, we can uncover the vector representation needed to access the finromation that the user actually wants.



