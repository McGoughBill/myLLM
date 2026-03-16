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

This is where things can go wrong and get complicated. We are relying on an LLM's output to do systemic workflow-critical steps in the query-translation step. This, inherently, can go wrong.
For example in the multiquestion_querytranslation.py script, in order to map translated queries to the retriever, they have to be delimited by '/n'. There should  be a fixed number, or limited number, of translated questions.
If the LLM does not delimit the question correctly, and does not limit it's translation to a sensible low number of queriable questions, then the system can silently fail at the retrieval stage.

Therefore, it is really important to correctly engineer the query translation step. You want a reliable workflow in this step, that consistently outputs what you expect. This requires
high-power LLMs, that are instruct-finetuned, a good subsystem prompt, and careful evaluation once written.

Another method of query translation, besides multiple-question rewording, is RAG fusion. This is like multi-question query translation, but with a ranking element.
My first guess for this step is that the more often a document is brought back by one of the multiple queries, the more important it likely is, and then we can do
top-k from these documents. That's my guess.