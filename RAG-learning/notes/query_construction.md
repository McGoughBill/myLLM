*Learning RAG from Scratch: Query construction*

This is the process of converting natural language to a domain-specific source such as SQL, cypher, or
any other database language/api.

For example, constructing a query of using natural language prompts, where the user might not have perfect access
to the data columns/schema.

This will be very similar to logical routing using a structured output, where we constraint our LLM to output a structured
data format such as a JSON. Then the structured output can be used to query a dataset.

The pydantic structured output binding is very very useful here.
