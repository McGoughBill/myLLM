*Chapter 1: Deep Reinforcement Learning*

**MDP: Markov Decision Process**

RL presumes sequential decision making within an MDP. Unlike supervised learning, RL has 'sequential' actions and states.
Because a model is acting in a temporal space, with a state, we call them 'agents'. This usefully differentiates them from
SFT models which take one action, even if it is of arbitrary length as in generative AI models trained via SFT.

RL agents and MDPs have a notion of time, or order. Pi, used through RL literature, refers to the policy of an agent -
this is the agent's literal wieghts. Policy is the function that receives state as input, and return actions as output.
Following an action, the environment's state has a transition probability (is this not just a changed state?) that can 
be evaluated for a reward.

The whole trajectory of the MDP is denoted by tau, which is an array defined by time-matched state, action, and reward,
where the reward relates to the action in that space, NOT the subsequent space caused by that action... interesting. always
terminate the trajectory with a state.

States that an agent could reside in belong in a state set big S. We have to define the notion of a terminal state (game end being an example).
S+ is that set of states available to an agent AND the terminal state (all action that could be taken by the agent AND terminal state).
Always have to assume the state-set is non-empty.

Similarly, the action set is all the actions available to an agent, big A. Action set
and reward set can never be empty. At time t, agent will collect a reward (a real finite number). The policy is trained to maximise the sum
of rewards.

Terminal time/stopping time - this is a hyperparameter that causes sequential episodes to end.
Episodes start at t=0, and end at t=T. Terminal state can enforce a terminal time (game end, death in a game), but
also, I am fairly sure we can manually set a time here.

In some environments (self-driving cars), terminal time can be infinite. This is equivalent to the probability of transition to
s=terminal state being zero. If T=infinite, the task is a continuous task, otherwise it is an infinite task.

***Probability bits***

The initial state s0 is assumed to be sample from a distribution of initial sates, p0. Often, s0 is practically fixed. But it can help to have a
variation of initial states.

The transition probability p(r,s'|s,a) given by the interaction between environment an action, (which is usually not exactly understood by policy) where an agent's action
is selected by the policy.

***MDP base definitions***

Sometimes, the rewards is a fully deterministic function. The reward function takes state and action as input.

Similarly, sometimes the next-state is a fully-deterministic function. It takes the previous state and action as input.

For now, we will assume dynamics are stationary. This means that the reward function/next-state are not themselves functions of
time, and thus transition probabilities are deterministic. For example, the reward  of a football game/video game could be defined by the score, and a reward defined by the score
would be stationary throughout the whole time playing. However, a more complicated dynamic would reward different behaviours in different phases, as the opposition grows tired,
and start losing physicality and focus, which may reward different behaviours in different moments. 

A reward function could, hypothetically, be made to be a function of state itself. A crowd cheers attacking play; this encourages
players to be more aggressive.

Policy (agent NN) can be deterministic, can be probabilistic (like a VAE or a GAN). If the policy is stochastic, then the
action given a state becomes a distribution rather than an answer.

Markovian - all decision can be made using only information in the current state, and the past is irrelevant. Classic counter example - a tiger hides behind a tree.
Because humans understand object permanence, we should be wary. A markov model would not be wary, having no longer access to data
on the tiger's presence.

The same is kind-of true amongst all RL problems - momentary states can occlude and mask important background dynamics. For us, we have access to the full state as
a conversational history. But this is a general problem to be aware of.

***Terminal time notation***

T, terminal time, is a random variable. We have no idea exactly how many actions will need to be taken to produce a terminal-state.

We will later run into issues with this, so formally we will treat a terminal state as an absorbing state. This means a state from which we cannot
change (despite any choice of action), and which always rewards 0. Under this definition, the policy at terminal state becomes irrelevant.

***Imitation learning***

This is process of trying to match the policy of our agent to the policy of sum expert. This essentially trivialises reinforcement learning
to a supervised learning problem, where the expert/our agent observe a state, and try to reproduce an action reward via cross-entropy loss.

Method:
1) sample trajectories, collecting states and actions from expert observation. This comes from the 'triplet' - the initial state distribution,
the expert, and the observed interaction between the two (transition probabilities). From this, we form a dataset of state-action pairs.
2) train policy to minimise cross-entropy loss on expert actions

***Imitation learning in LLMs***

This is basically what happens in next-token prediction, and it is essentially SFT. It is a good starting point.

Where this behaviour falls short is in changed distributions of data. So next-token prediction over reddit probably does not
perfectly generalise to wikipedia, or to a new novel. In both cases, experts were used to train the model, but the different environments
cause wildly different performance within a new environment. The model will assume it is in it's previous environment, and recapitulate
data from that environment in a highly-lossy way.

***Off-policy vs. on-policy learning***

In imitation learning, the training data itself is generated by the policy of an expert. This is not the case in pure reinforcement learning! In pure reinforcement learning,
the policy samples dictate where the state goes during training, which literally creates the next state in which the agent is trained upon. In other words, in pure reinforcment learning,
our training data is created by the policy itself! This is called 'on-policy' learning, where the agent is trained according to its own policy, rather than
the policy of an expert.

Off-policy trained models suffer from distribution shift. Imitation models are off-policy trained models.

On-policy training allows models to adapt to a new, non-expert environment.

***DAgger***

Dataset aggregation - this is on-policy imitation learning. In DAgger, the policy is used to guide the training data collection, and experts jump in to provide supervision.
Generally, we pretrain off-policy before going ahead with something like DAgger. Key insight is that we remain on-policy whilst gaining good supervision.

This is pretty akin to next-token prediction.

Can't really do something like DAgger with LLMs anyway (for direct text prediction, true we can do NTP)

**MDP Objective**

Aim is to maximise the sum of expected returns. we apply a discounted reward over time. G is the cumulative reward, r is the instantaneous reward. We apply a discount
factor between 0-1 to ensure tapering over time and ensure a finite reward; generally discount is between 0.95-0.99999. We prefer to collect rewards sooner, rather than later.

Expectation of policy refers specifically to the expected reward of an action given a certain state and action pair

Return at time t is G_t = r_t + discount*r_{t+1} + discount^2 * r_{t+2}....

FYI, the denotes of _t in G_t means the calculation of cumulative reward STARTING from time t, not ending at time t.

**State value, and State-Action value (V and Q)**

I imagine this refers to how we evaluate reward, and whether we look at purely the state (a state value function, V), or whether
we look at both the state and on-policy action (Q).

The state value function evaluates the EXPECTED (not actual) value of a state, if we followed the policy thereafter. Value of terminal state is 0.

The state-action value functions evaluates the EXPECTED (not actual) value of an action that dictate, given a state and following the policy thereafter. Value of
any action whilst in terminal state is 0. The crucial aspect here is that we DICTATE an action on the initial step.

This means we need the actions of our policy NN to be 'seen to the end' before evaluating the value of one specific action or state. When our agent is being trained, we have to sample action from the
policy across a full play-through.

***Basic properties***

V(s0) = Expected (Q(s0,a0)) if a0 was taken on policy. This is because V assume all actions are taken on policy, and the only action that could be taken off-policy,
which is a0, is constrained on on-policy behaviour. This means that V and Q are essentially identical from the second state onwards - the point of Q is to evaluate the
resultant value of an action.

thus, the only useful difference between V and Q is the off-policy action.

**Unnecessary Maths**

***1-step transition property of Vpi and Qpi***

Really really poorly explained. No intuition, reading from a slide. Great.

The essential, useful information was not even explained, but this was my takeaway: the value of future states is independent of past states. This cheapens
value/reward calculations, as we can just evaluate each state in a forward-looking way.

This was a mathematical formalisation of the markovian property. If this property didn't exist - if previous states DID matter and we couldn't observe them - we 
would have to factor previous states into our current state value-evaluation.

***Banach fixed-point theorem***

Let x be a metric space, with n-dimensions. We say an operator is contractive if two vectors passed through the operator in this metric space get closer together. 

The theorem states that this operator has a fixed point, where all outputs of the operator tend toward. This
means a vectors that are passed through a contractive operator tend toward a point (coordinate?) in the metric space.

***Bellman equation***

Let pi be a policy, assume the discount factor is less than 1. Assume the state space and action space are both finite, and the reward is a real finite number.

Then, the value function V_{pi} exists, that maps a state to a reward. 

Let B be a bellman operator that maps a function to a function. For example, it could multiply the increase the parameters of any function by 2. B in this case is defined as taking the policy and state as input and returning a value function. This
makes these two functions equivalent. Essentially, what this is attempting to show in the slides but is quite poorly communicated, is that there is a value function that is the only fixed point of the operator B.

The overall takeaway of the unnecessary maths is that the discounting is necessary to ensure the value and state-value function are tractable.

**Optimal policy and value functions**

We say a policy is optimal if the value of a state (fully played out through the value function) from that policy is equal or better than all other policies (fully played out also),
where the optimal policy is independent of the state.

That is to say, it is uniformally the maximum-reward policy across all possible states.

Optimal value function from this policy is known as V*, and the optimal state-value function is known as Q*. This is some hypothetical perfect model and evaluation methods. Apparently, the optimal value
functions V* and Q* are unique, but the optimal policy is not unique... I have no idea how to interpret this. Many different policies can lead to an optimum value?

Oh, because value functions return a real number. so many policies could result in the same number being produced. fine.

***Bellman optimality for V***

The above characterisation of optimality is V* > V for any policy, at any state. That is to say, V* returns the optimum value for any state.

Another way of characterising this is through the fixed-point equation.

V* should be equal to the argmaxQ*(S,a) for all possible actions A. That is to say, V* at state s is equal to taking the optimum action at state s and 
evaluating Q* with the state-action pairing. This is a trivial understanding that's perhaps harder to prove with maths.

Therefore, the optimal action at a given definite state can be evaluated by selecting the action that maximises Q*(s,a), which actually allows us to indirectly calculate the
optimal policy, as optimal policy is usually a neural network function that converts a state to the best possible action.

Therefore, optimal policy can be approximated by evaluated the effect of a bunch of different actions, calculating the on-policy Q, and selecting that which had the
best long-term value?

Honestly, I stopped paying attention. Genuinely awful lecture content. No insight or intuition, just reading of lecture slides.

Here are the takeaways:
1) The optimal policy is a real function that we must approximate or find
2) The optimal value function does indeed evaluate the same value as the optimum policy from state S
3) The same is true of the optimal state-action value function
4) The bellman operator is an operator that receives a function and returns a function, whilst being contractive, reducing the value
5) The bellman operator, that is contractive and returns an identical output to value function, has a fixed point at the value function's output 

the fact that the value function and optimal value function can be characterised as fixed points of the bellman operator is fundamental to the design
of new reinforcment learning algorithms.

***Discussions with Gemini on this***

Bellman was attempting to solve the problem of intractable reward for reinforcement learning. How do we evaluate the value of an action, given that it takes
several actions to know whether we performed well?

Well, first, the optimal value function of state zero should return a number equal to the maximum reward our an action in our current state, and the optimal
value function over the future state caused by that action. In other words, evaluate the optimal reward for our current state (take one action and evaluate that), and then 
evaluate the best possible reward after.

The insight is: from a given state, the maximum reward will stay fixed as long as optimum actions are taken. Imagine a maximum life-expectancy. Any unhealthy decision should
reduce that optimal life-span. One could spend their whole life living perfectly, and this maximum life-span would only remain identical. However, as soon as any sub-optimal 
action occurs, this necessarily reduces the final optimum life span.

The idea is that we are bad at evaluating our lifespan but, over the course of our life, we will get more accurate at predicting this. We track how far we have come so far, 
we can track all our unhealthy behaviours too, and interpolate our final lifespan with ever-increasing contractive accuracy.
Optimum actions "bank" the best possible rewards, taking you to the next state. 

The bellman operator, then, is the think that applies retrospective value terms GIVEN the new information we have in the future. When I was 20, I was smoking and didn't know its effect;
I thought my fit physique would lead to me to a lifespan of 100; then, when I was 50, I received a cancer diagnosis that made me realise (a) my actual lifespan is likely to be much lower (yet higher than
my current age), and (b), that I must retrospectively reevaluate the sensibleness of my smoking, as it likely led to this outcome. This is Value Iteration, and it is the key behaviour unlocked
by the bellman operator that allows a better understanding of past value given current information.

***bellman operator in this context***
The bellman operator does not prune actions, or dictate spaces. It simply allows recursive recalculation of previous state-wise values, as more information about future rewards and new states roll in.
This means that all possible actions must be taken from all possible states to allow the final recursive definition of the actual value of all given actions. This is intractable for nearly all use-cases.
Optimal value is then the value of the path that returns maximum reward.

The bellman operator means that, no matter how inaccurate our initial value-estimation function is, iteratively updating previous values using discounted rewards and improved value estimations
will improve our past, retrospective value estimation accuracy compared to our original guesses. The previous values will tend toward the correct value.

Where I am struggling: the crossover between dynamic programming, and neural networks. The bellman operator and equations are properties of
dynamic programming. I must eject neural networks from my head in order to get a generalised understanding. Bellman operators are used to correctly
understand PAST VALUE - this means they are used to improve our understanding of previous scenarios once we get some tangible reward.

**Value Iteration**

The optimal value function is a fixed point on the bellman operator space - once the bellman operator is applied to a space with full knowledge
of past actions and a terminal goal, one can recursively apply a bellman operator that redistributes value across the past action/state space until
the optimal value can be trivially collected as the path of highest reward.

Value iteration, to my understanding, is the recursive process of applying the bellman optimality operator to a series of values based on new, more current
values with more information. This means that we can estimate a value at state 0, take an action (and any given reward), and re-estimate value at our new state 1,
and use this to retroactively correct our state 0's value with that new information gained at state 1.

In reality, it is impractical to implement the bellman optimality operator, as this requires access to the action that maximises value at every possible state. This
is pretty limited to the tic-tac-toe example. despite this, value iteration and the bellman operators are instructive frameworks of thinking.

**Policy Iteration**

Similarly to value iteration, except on the action space. We first evaluate V and Q of our current state and state-action pair.
Then, we compute policy as the action that maximises our next value V. In reality, we cannot evaluate all possible actions, unless in a simple game,
so we cannot compute perfect policy iteration. Policy iteration provably converges under similar conditions.

**What goes wrong with discount, gamma?**

1) Sometimes, nothing. Weird. This often happens in LLM papers.

Essentially, problems only arise when the value function becomes undefinable. This happens because, we could take an infinite-many actions
that may lead to non-degenerating rewards, and so cannot converge toward a finite value.

However, if you only get a reward at the end of an episode, and never a reward if an episode doesn't conclude, then
this doesn't become a problem. LLM's can get cut off without a reward.

2) Exploits can be found by agents, and infinitely capitalised on.

Imagine if coins were rewarded in a game that ultimately requires an end state to be acheived to terminate.
Well then, coins may just be exploited until the end of time by our agent that sees no discount on this behaviour,
EVEN if coins only give a FRACTION of the benefit of termination.

One way around this: we consider average reward rather than maximum reward. This is more technically challenge, and is
an active area of research.