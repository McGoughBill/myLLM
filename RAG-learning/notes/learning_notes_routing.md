*Learning RAG from Scratch: Routing*

The idea here is that we want to connect our retrieval query to the correct source, such as a database.

My initial guess for this is that we probably want to have various sources of knowledge. In the case of a medical doctor's assitant,
we might have textbooks, the latest scientific literature in oncology, the latest literature in radiology, the hospital's guidelines 
and patient pathways, insurance policies for the customer, etc. this data will likely come from completely seperate sources.

We want to make sure that our query retrieves from the correct source(s). We can do this via logical routing or semantic routing.
We can also use routing to 

Logical routing fine-tunes (or, otherwise, repurposes) an LLM for choosing between different discreet options itself. I.e., it should only
output 'A', 'B', or 'C' for example, where those letters correspond to data sources.

Semantic routing associates each knowledge source with a semantic text embedding vector. This is different to per-document or per-sentence
retrieval, because it is the knowledge-source itself that has the embedding. To get this embedding, we would likely have to describe the
source in natural language, and describe example documents, and then associate this vector with the database.

**Logical routing**

To do logical routing, there are two good methods:
1) We can force structured output using a pydantic structured output model (we can enforce arbitrary structure on output) - this performs identically to a classification task
2) We can use free text, and scan the outputs for the variable names that we asked it to produce - this is a soft-classification approach