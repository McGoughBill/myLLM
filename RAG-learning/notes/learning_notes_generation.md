*Learning RAG from Scratch: Generation*

In the dummy example 'indexing_and_retrieval' script, we simply inserted all retrieved document section into a system prompt.
The generation step is where the answer happens, and therefore is suited to high-quality LLMS. So, in our generation, we generated a prompt
like the following: 'Using the excerpts from documents [source], 1) [excerpt 1], 2) [excerpt 2] answer the following question: [question].'

This worked really well in my simple example. We added the relevant documents into the system prompt (context window),
which makes the question answering (usually) trivial. The more context the better. For example, where does each excerpt come from,
who was it written by, when was it written, title of document, chapter, subtitle, etc. This is all information that can be added to the system prompt, 
and can help the LLM to understand the context of the retrieved information, and therefore generate a better answer.

In langchain, there is an expression language (LCEL) that allows you to swiftly build prompts and prompt worklfows in something called
'chains'. See this at the end of the python script 'learning_langchain.py' for an example of an executable langchain.