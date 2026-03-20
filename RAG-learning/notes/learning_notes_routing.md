*Learning RAG from Scratch: Query Translation*

This process converts a user's query into something that can be used for improved retrieval. Implicit in this step is that
user's queries are often in non-optimal form for retrieval, questions may be open-ended or context-dependent, or 
may be wordy/contain unnecessary infomration. Queries should be concise, containing all (and only) the information
needed to answer the request.

At the high-level, we can actually let a large-language model decide which way to go perhaps? So we let the language model decide if we
need sub-question answering, chain of thought, or step-back.

We can either go more abstract or less abstract. More abstract: can we use an LLM to summarise the essence of a question in a more
concise way?

Less abstract: what specific pieces of information are needed to answer the question? First, let's find those answers, and then
use those answers to construct a query. This is more of a "chain of thought" approach, where we break down the question into sub-questions, answer those, and then use those answers to construct a final query.

In vector representation, the theory behind this is that a poorly worded question/overly verbose, imprecise question will have a poorly matching 
vector representation to the information needed to answer the question. By contract, by tackling the problem from different perspectives with more directed
questioning, we can uncover the vector representation needed to access the information that the user actually wants.

**Multi-query translation (simple)**

This is where things can go wrong and get complicated. We are relying on an LLM's output to do systemic workflow-critical steps in the query-translation step. This, inherently, can go wrong.
For example in the multiquestion_querytranslation.py script, in order to map translated queries to the retriever, they have to be delimited by '/n'. There should  be a fixed number, or limited number, of translated questions.
If the LLM does not delimit the question correctly, and does not limit it's translation to a sensible low number of queriable questions, then the system can silently fail at the retrieval stage.

Therefore, it is really important to correctly engineer the query translation step. You want a reliable workflow in this step, that consistently outputs what you expect. This requires
high-power LLMs, that are instruct-finetuned, a good subsystem prompt, and careful evaluation once written.

**Multi-query translation (more advanced)**

Another method of query translation, besides multiple-question rewording, is RAG fusion. This is like multi-question query translation, but with a ranking element.
My first guess for this step is that the more often a document is brought back by one of the multiple queries, the more important it likely is, and then we can do
top-k from these documents. That's my guess.

Yeah, this was pretty much correct. In real workflows, it would be likely very important to optimise the comparison/rank function and metrics,
but here we just did a simple count.

**Query decomposition (IR-CoT) (more advanced)**

And then finally, there is task decomposition. This is probably the most useful query translation mechanism there is, and it will incorporate aspects of both of the above.
Least-to-most chain-of-thought reasoning: decompose the user query into smaller problems, then solve them sequentially.

Then, we can map chain-of-though reasoning to information retrieval (IR-CoT). We create several sub-queries that would answer part of the user query, and we retrieve
sections of information that correspond to each of those sections individually.

the key aspect of IR-CoT is the iterative 

**Query decomposition (Step-back) (more advanced)**

Main idea here is to answer question in a similar way to how a therapist would talk: first, summarise what you think you have heard from the user, and generate a more abstract query.

The aim of the more abstract question is to retrieve a superset of the information necessary to answer the user's query.

Example: query: What country was person X born in; step-back: What is person's X's personal history
Example 2: query: what city is the person who broadened the doctrine of philosophy of language from?; step-back: Who broadened the doctrine of the philosophy of language?

Then, we can do retrieval on the (1) original query, and (2) the abstracted super-set query, and then finally answer the question.


**Query decomposition (HyDE) (Most advanced)**

This is the final method we will consider. The basic function behind RAG is matching the vector-space of queries and documents that hold
information that would allow queries to be answered. We can do this with dedicated QA matching vector-space LLMs like sentence-transformer, but
in reality this can fall short for highly complex questions. So, the essential function behind RAG can fail in the case of complex, out-of-distribution
queries, such as highly-technical queries or multi-disciplinary queries.

For these cases, HyDE was set up to allow complex document retrieval. HyDE stands for hypothetical document
