*Learning RAG from Scratch*

These are the notes of my learning from the freeCodeCamp video 'Learn RAG From Scratch'
- RAG stands for Retrieval Augmented Generation
- RAG is a method that combines retrieval-based and generation-based approaches in natural language processing (NLP) tasks.
- In RAG, a retrieval model is used to fetch relevant information from a large corpus of documents, and then a generation model is used to generate a response based on the retrieved information.

**Introduction and basic concepts**

Generally, RAG works by first indexing all documents in a database that can be accesesed via SQL or similar. Then,
a query is made using the LLM (not sure how exactly that works yet), where we seek to find all/the most relevant documents
and sections of those documents for responding to a question.

This database is called a vector database.

1) The first step in RAG is query translation; this is the process of translating the user's request into a form
better suited for retrieval and querying the vector database. This is: rewriting, and decomposing into sub-questions.

2) Second, there is query routing. It is important to figure out which document or database should be queried, based on the user's 
request. This is done by a 'router' that determines which document or database is most relevant to the user's request. The router
can be logical (i.e., chose between database A, B, or C) or it can be semantic (convert the query into a vector
and then find the most similar vector in the vector database). 

3) Then, there is query construction. This process takes the translated query and turns it into a 'DSL' - domain specific language - 
that converts the query into a SQL query (for example) that can be executed against the database. But also, text-to-vector
for vector databases is commpnly used. This is the process of converting text into a vector representation that can be used for similarity search in the vector database.


4) Fourth, there is indexing. This is essentially allowing documents to be easily retrieved. Think of it like a high-level text
description of each section of a document, or whole documents. Like an abstract if text/SQL based, or a vector representation if vector based. 

5) Then, there is the retrieval step. This process uses the query constructed in step 3 to select documents/data in the indexed database.
Deciding how many documents / which sections is an open problem, that is likely task and context dependent. Then, using these documents,
it retrieves the necessary information to satisfy the request.

6) Generation then generates the responses using the retrieved information. You can add an extra step here, such as 'refinement' or 'active retrieval'
which automatically assesses the response for relevance and accuracy, and if it is not good enough, the system can go back to retrieval again.


The basics of RAG are Indexing, Retrieval, and Generation. The more advanced methods are Query transformations/construction, routing using LLMs, and active retrieval and refinement.