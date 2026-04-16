*RL explained with LLMS*

Here, we are following UCLA's RL course https://www.youtube.com/watch?v=q9972BRoXzQ&list=PLir0BWtR5vRp5dqaouyMU-oTSzaU5LK9r.

**Course outline and prologue**

Quote from Richard Sutton: _The only thing that matters to AI researcher's in the long run is Moore's law._

The intended takeaway of  this quote is: when developing methods, we must exploit the fact that the newest hardware makes computation SO MUCH cheaper than previous generations.
It is the price, and availability, of compute that has made way for AI's current ability, and therefore our methods in the future can be much more highly powered than previous generations.
One should analyse and develop methods, and scale neural networks, as if computation will be much cheaper in the future.

This MASSIVELY simplifies research goals, in my opinion. Basic, legible, clear objectives that have weak signals become strong
in large datasets and enormous compute (next token prediction, for example).

Search and learning have dominated Go and Chess. It now contributes to finetuning LLMs. But we are still at a precipice; RL has not
found a home in all applications.

Finding RL's home relies on enormous compute, a complete search strategy, efficient software, and a clear reward signal. As researchers,
it is our job to find the clear reward and define the search strategy - (this can be brutish!). The rest, Moore's law.

The natural enemy of this philosophy is the knowledge-first approach; engineering our anthropomorphism and heuristics into models for incremental gain. 
Such machines are built on a preposition of understanding of human cognition, and optimum knowledge access, modelled on our own minds. 
This is the guided-search strategy for chess, rather than brute force. 

The point here is human-centric methods, methods that place human strategy as the ultimate goal or lean on them too heavily, often succumb to RL-trained systems eventually.
RL-systems are incompletely understood. Human-imitation methods actually bring models closer to human alignment. This makes human imitation attractive. 
But the incomplete RL radical approach scales better with Moore's law over time. Human alignment is an incremental gain; computation-centred approaches
allow step changes.

Two methods that scale well with computation: search, and learning.

**LLM training process**

1) Collective internet-scale text data. In GPT 4: 1.3 trillion tokens!!!!
2) Create the transformer architecture. In GPT 4: 1.8 trillion parameters!! (I imagine mixture of experts?)
3) Pretraining: next token prediction, a classification task over a whole vocabulary.
4) Finetuning
5) Reinforcement-learning

**Dual process theory (?)**

This is a theory of the mind: we have two systems for thinking. One is highly emotional, reflexive, and fast. It is our habitual normal networks of response.
The second system provides a more regulated, more expansive, slower and broadened perspective. 

It seems that LLMs allow fantastic system-1 thinking, but perhaps struggle with system 2? I would say that stimulation elicit's response, 
which is reflexive and therefore conceptually similar to a next-token prediction conditioned on an input and an individual's habits (policy), but
this can be supressed and reconsidered through an unstimulated lens, given time. The second system, the unstimulated lens, allows a broad inner search of feeling and sentiment.

Francois Chollet compares this to an interpolative database with a natural language interface. This makes complete sense.

From the paper: Language models solve math with a bag of heuristics. They provided evidence that models, trained on next-token-prediction, tackle new mathematical
problems with the heuristics learned during training. They are not simply recapitulating training data, they have learned the "causal model" of maths in training,
and apply that to new problems.

The current perspective at openai is that test-time reasoning (search!) and improving strategies for doing so is where the future research must lie.

The idea in this course is that RL can mimic system-2 thinking with search and self-play to evaluate positions. System 1 is actually active within system 2 as the reward estimator;
the search system and system memory is what allows the broad 'inner search' that isn't overly condition on system 2's reactions.

Chess/Go and similar use a neural-network monte carlo tree search. It would do me well to read about what this is. Idea is Neural-guided search:
proposes possible ways forward, a second system (essentially identical to system 1) estimates the value of each possible path, and then search continues from those
points.

As System 2's search happens during test-time, this is a form of 'test-time scaling' - where a large proportion of compute is dedicated to
the deployment of a model, rather than model training. Chain-of-thought is like a neural-guided search, and DeepSeek showed that it could
be trained via pure reinforcement learning!

SFT Memorises, RL generalises?