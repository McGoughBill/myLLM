**Reading list**

This document will cover the basics of a few background topics that I should brush up on for agentic AI engineering.

Here are some topics, ordered by technical background required, that I am aiming to learn

***basic***
1) HTTP & Rest
2) JSON serialisation/deserialisation
3) Async programming

***multi-agent systems***
1) client-server architecture: client make requests, server handles them. An agent is a client of a large number of requests,
and itself becomes a server when dealing with someone else's query through an api.
2) Queues & message brokers (handling production for concurrent users)
3) tokens and auth - Oauth, JWT tokens

***infra***
1) Fast API / REST api
2) Containerising the agent
3) Databases

**actual reading list**
  1. https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview                                                                                                                                          
  Free, ~30 min read. Covers HTTP, request/response, status codes, and client-server architecture in one go. MDN is the definitive reference — bookmark the whole site.                                  
  2. https://realpython.com/async-io-python/                                                                                                                                                             
  Free, ~1 hour read. The clearest explanation of async/await for someone coming from a scientific Python background. Skip to "The Event Loop" section first.                                            
  3. https://fastapi.tiangolo.com/tutorial/                                                                                                                                                              
  Free, ~3 hours hands-on. This single resource covers: JSON serialisation, REST API design, authentication with tokens, and turning a Python function into a server. Build it top to bottom once.       
  4. https://docs.docker.com/get-started/                                                                                                                                                                
  Free, ~2 hours hands-on. Parts 1-3 only. Everything after that is beyond the minimum.
  5. https://www.oreilly.com/library/view/architecture-patterns-with/9781492052197/ — Harry Percival & Bob Gregory                                                                                       
  The one book. Chapters 1-3 cover the repository pattern and service layer (how to structure code around a database). The rest covers queues and event-driven systems. Free to read online at cosmicpython.com. 