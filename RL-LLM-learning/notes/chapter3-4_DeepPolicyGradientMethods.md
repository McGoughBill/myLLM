*Deep Policy Gradient Methods*

So far, we have discussed policy evaluation, approximating V and Q using a neural network
over the policy/state. We can train value-estimating networks using monte-carlo, or temporal difference
methods that essentially encourage networks to accurately predict the reward given between two
states as the difference in estimated state value.

Now, we talk about _optimising_ policy, maximising a next-step's expected value according
to our value function.

**Intro to policy gradient methods**

We assume terminal time is finite, meaning an agent's full trajectory is calculable,
and full discounted return is a finite number too.

In terminal state, we set the action to deterministically equally an end-state action with
a probability of 1.

We have a probability distribution at terminal time, across the whole trajectory, that is equal
to the product of all probability distributions at each timepoint following each previous state's
action and reward. I think this refers to the transition probabilities, where the distribution tells
us about the final state of the model.


**Intro to policy gradient methods**

J is our policy-evaluating function; the value of our J function is the expected cumulative return 
from following our policy until terminal state.

We can find the derivative of our J function, with respect to its parameter values, to see how adjust parameter
values alters the expected cumulative reward. This derivative tells us how altering the parameter values alters 
the expected cumulative reward.

The derivative = Expected cumulative reward * gradient of the network * the log of transition probabilities across the whole trajectory. Transition probabilities
are the state-change probability distribution given possible actions given its latest state. This gradient we call little g, g.
We calculate this for multiple projectories at the same time.

Practically, little g has a very high variance which makes optimisation practically difficult. Hence, we perform some
engineering tricks to improve stability.

***Optimisation tricks***

1) _Simplify transition probability gradient_
The log of transition probabilities, across the whole trajectory, is hard and not that necessary to calculate. The only aspect
of state-change that varies with parameters is the policy network.

2) _Remove past rewards_
Technnically, each stochastic gradient is calculated by reviewing all gradients across the trajectory. However, this is unnecessary, as the
past rewards do not contribute to the gradient of the cumulative expected return. Therefore, we only consider current and future rewards.

3) _State-dependent baseline_
b is a function of state. This is a function that receives a given state and returns a value. It is independent of the
policy, and therefore has no effect upon the policy network's gradient. I believe the purpose of B is to establish a reward
basline of a state to allow us to measure deviations based on policy... I guess we will find out later..

Yeah this is gunna be the V-value function

4) _Q-estimates_
Q is a random variable that estimate value of a partial trajectory at state t, based on state t and action t.
We can substitute the actual discounted return G for Q if the Q value can measure the actual value of the policy decision.

That is to say, if our Q-estimate function can accurately predict the value of a state-action pair, we can replace the monte-carlo
calculated discounted-return value G with Q in our gradient function.

The aim is to choose V and Q value functions jointly in a way that minimises variance and clarifies training signal

Problem: We don't have access to the exact Q value function, parameterised in a neural network.

Rao-Blackwell Theorem: basically, a function of two independent random variables has a greater variance than a function of two independent random variables
that is conditioned on one of the variables. That is to say, conditioning on one of the input random variables reduces the function's output variability.

**Result of tricks**

We have an unbiased estimate of gradient of estimated cumulative returns with respect to some policy network! This is in imgs,
chapter3-1.png. it holds that the gradient of final reward value is the product of the following: expected value of the gradient of the log-policy with respect to its weights,
multiplied by the reward at every time point, and the difference between the Q-value (over the policy action) and V-value function at every timepoint.

The difference between the Q value and the V value is known as A, and it is called the advantage of an action at a given state.
If the advantage is greater than zero for a given state and action, it means the action is good and it is better than the average action of the policy network.
If the advantage is negative, it is a bad action, worse than the average action of the policy network.

This is an approximation of advantage I guess, since the V function does not directly incorporate any aspect of policy.

If an intended policy was good, then advantage is positive, then the network weights are pushed in the direction of the created policy (makes policy more likely).
IF the intended policy is bad, then the advantage is negative, and the network weights are pushed in the opposite direction of the created policy.

we apply a log to the policy network to smooth the outputs - i think. 

***Without the baseline***
We would be giving "less good" reward to bad actions and "more good" rewards to good actions. But less good is still self-encouragement
even if it is, on average, lower magnitude. Policy is effectively a probability distribution, that is normalised using the softmax function, 
so this "less good" is still effectively a negative reinforcement for bad actions, but it is unnecessary variance. Introduction of
V-value baseline should zero the mean and allow pure-variance based around the additional utility of an action. In other words,
it clarifies the training signal.

**Interpretation of this method: Actor-critic**

We have a policy network, and some value networks. the state-value V function serves as critics.

Problem: we do not know the exact Q-function for a policy defined by a neural network. We do not have a function that accurately
estimates the value of a policy at any given point.

We can replace the Q network with another neural network - but this is another layer of training complexity that is undesirable.

***final enhanmecent***

MC - k-step Temporal-difference.

We substitute the Q-value function for the V-value function with later-calculated rewards, like in temporal difference
policy evaluation methods.

this means we do not need to seperately learn Q - only the V-value function is necessary here!

The value network CAN be a randomly initialised neural network to begin with, and this is okay because of the properties of
multi-step temporal difference value estimation, the intermediate rewards are good enough signal to estimate V-value IF their is
sufficiently distributed rewards across the state space. If not, we need a larger temporal difference or a better-trained V value network
to begin with.

**Interpreting the policy gradient algorithm #1**

0) instantiate two networks: policy network, and state-value network.

while policy network not converged:
1) sample a trajectory from initial state to terminal state/a temporal difference
2) For every time point in that state, we calculate Q value of an action as the sum of rewards + the gamma discount of the final state times the final state value.
3) perform gradient ASCENT for policy network - this is the grad of network * log of action space * discount at state * advantage of action. Because convention is gradient DESCENT in pytorch, to do ascent, we simply multiply our gradient by -1.
4) perform gradient DESCENT of value network - this is the temporal difference calculation for V based on network-estimated value and the calculated Q. Remember to do stop-gradient on the Calculate Q value.
5) use some optimiser to dually optimised both networks.
end

***The log trick***

the derivative of the LOG of the action space is not there by accident, or as a variance-reduction method. It comes out of the logical manipulation of the
gradient of policy value.

We go from calculating the gradient of an expectation (hard) to the expectation of a gradient (easy, with samples - i.e., requires rollout/full trajectory).

Gradient of an expectation - hard. It is hard to calculate the gradient of an expected return, because it requires complex knowledge of how a changed network and
a complex environment interact. This is extremely hard to model but, if we could do this, we would NOT need to rollout trajectories and let our model act in the world.
We could simply mathematically represent the world as a set of differentiable equation, and directly optimise our model using these hypothetical equations.

Expectation of a gradient - easy. Instead, we can sample many trajectories over the distribution of possible actions and spaces, and let the environment
tell us what how policy and the world interact. We can see how variation in actions, weighted by probability of the network's output, results in better or worse
rewards. The log trick is what allows us to isolate the policy component from the state-policy-transition interaction - only the log of the policy is differentiable by
the policies weights, with others having no interaction and therefore dropping to zero.

The log trick is what allows us to shift the relatively placement of sum a

***The gamma trick***
Gamma is a <1 parameter that is used to attenuate rewards over time.

We have a trade-off with gamma: gamma less than 1 makes optimisation more stable, but gamma equal to 1 results in better long-term planning performance in models.
Often, people observe better performance with gamma=1 with their reinforcement learning setups. Sometimes, an artificial discount factor is added to try and achieve the best of both worlds.
I believe what this does is use discounting in the Q-value calculation, but NOT in the actual gradient adjustment calculation. When we use this artificial discounting,
the discount factor is called 'y-tilda'.

the gamma trick REMOVES discounting from the gradient calculation, which results in non-attenuated gradient adjustments later in the trajectory, but DO use it within
the q-value calculation.

**Interpreting the policy gradient algorithm #2**

0) instantiate two networks: policy network, and state-value network.

while policy network not converged:
1) sample a trajectory from initial state to terminal state/a temporal difference
2) same as above
3) perform gradient ascent WITH NO GAMMA!
4) same as above
5) same as above
end

**Algorithmic enhancement: SGD with non-uniform selection rules**

Consider F, which sums a batch of finite discounted rewards and divides by the number, returning an average.

SGD will take an IID sample (equally weighted between all samples in the batch) as a one-off learning sample.

Cyclic SGD will iteratively learn a batch until convergence WITHIN a batch. Infact, we don't even average the weights changees,
we look at each trajectory and take our learnings from each sample individually through each sample of the batch and cycle through 
until convegence.

Effectively, this is a method used for optimisation because batches are expensive to calculated in RL. If
we batch up a whole trajectory and average the weight changes, our learning will be very slow. So, instead,
in this method we don't use batches, and we learn each gradient individually. I imagine this leads to high variance?


**Interpreting the advantage actoc-critic (A2C) algorithm #3**

0) instantiate two networks: policy network, and state-value network.

while policy network not converged:
1) IF we are at terminal state: set t=0, sample an initial state for our next run, and end the run
1) ELSE: let's go
2) based on our current state, use our policy to select an action
3) let our agent's take its next action, and sample our reward and its next state from its interaction
4) repeat until terminal state OR k-steps have been reached - commonly, this is 5 steps
5) Then, calculate V and Q based on each state and state action in the collected rollouts (value is zero if in end state)
in a k-step manner. First, calculate 1-step TD gradient, then two-step, then three-step. A2C typically works backwards
from the end state; so, we calculate the value difference from the second to last state to the last state in the 1-step TD gradient,
then from the third-to-last state to the last state for the 2-step TD gradient, etc. until all five steps are packaged together for the
k-step TD gradient. We work backwards so we don't overfit on the first initial state, which would be the only state our value
network actually saw if we worked forwards (i.e., by working backwards from the endpoint, we sample each initial state uniformly).
6) calculate gradient on every single state for policy network
7) calculate gradient on every single state for value netowkr
8) optimise

**Interpreting the _asynchronous_ advantage actoc-critic (A3C) algorithm #4**

This was the original title of the paper published on A2C, and it refers to the fact that multiple computational units (GPUs, CPUs, etc.) that
could be used to independently perform gradient updates, where an individual computational unit was assigned to each k-step.

**Examples of A3C in continuous and discreet action/state spaces**

***Atari 2600 game***

Action space: 18 possible game actions in this game. 60fps, making the game discreet
Standard trick for games is to reduce the time-resolution of games (60fps is a lot!), especially
as human reactions are much slower than 1/60th of a second (implying redundancy in time resolution)

Humans used this trick: agent repeats the same action for 4 frames - 15 actions per second.

Standard preprocessing: Remove flickering (artefact of old graphics) - make game black and white, and downscale image resolution.
Also, inline with the repeated action every 4 frames, inputs to the network are stacked 4 most-recent frames - in the atari game, 
this allows for the notion of velocity.

Note: atari games are actually partially observable MDP, as the game screen does not capture the full game state. Also, the atari
screen.

Use a 4-channel CNN, leading to a fully connected layer treated with a softmax activation.

***Generating training data for this example***

Form a neural network that maps state to action (like the conv net)
step 1: sample an action given the state; retrieve BOTH the most-probable action (or sample from the possibilities)
and the probability of the most probable action (we are NOT interested in the probabilities of non-optimal actions)

step 2: backprop on the selected most-probable action (i.e., gradient descent of all parameters will only be assessed
according to its relationship to the selected most-probable action)


***MuJoCo***

This is a general physics engine environment, where we have different articulated 'robotic' structures,
that have a set of joints, actuators, and sensors as their state. The action space is the force applied by each
of the actuators within the physics environment. (this can be extended to real environments in the same way, where
PID controllers that then ensure the correct demanded force by the RL brain is actually applied by the physical motors.)

MuJoCo's state and action space is (somewhat) continuous, with some sensible maximum and minimum input bounds (not infinitely precise,
cannot apply infinite force).

***Generating training data for this example***

Assume the actions space is a single float between -1 and 1.

step 1: Form a neural network that maps the state to TWO real numbers. This is just the mean and (analogously) variance
of the continuous number. In reality, the variance is e^(2pi*SECONDNUMBER*FIRSTNUMBER).

We select our actual continuous output by randomly sampling the gaussian distribution, and applying some activation funtion I guess.

step 2: backprop on the value. Now, the actual maths for optimising through a gaussian distribution requires some thought,
but essentially it is two step - we want to align the mean (trivial - MSE equivalent) and expand the variance if far off, or 
retract the variance if near to the answer.

OOO nice - we can use the reparameterisation trick. We can sample from a normal distribution and multiply by the variance number, add the mean number.
This is brilliant, now we can optimise our network through the values of mean and variance by multiply by two pretty-much random numbers. The network will
see these as non-differentiable noise (it is!), but will eventually learn how to manipulate this noise to get desirable outcomes.


**Sampling Efficiency**

Essentially, this refers to the ability of a model to learn from few samples. Segmentation training is highly sample-efficient, next-token prediction has
very low sampling efficiency.

in RL - sampling efficiency is relatively high, with the compute costs of rendering the environment and calculating rollout trajectories being the main computational
bottleneck.

The question in RL with regards to sampling efficiency is: how can we learn more from an episode/rollout?

One of the answers to this need is the _surrogate objective_. In terms of objectives, this is adding an extra task to solve, to increase the learning
signal per timestep taken. In other words, this is multi-task learning.

What is happening, is that we have our policy network (START_P_NETWORK), that dictate's to us a set of actions that it would take along a trajectory,
and a value network (START_V_NETWORK) that lets us know the advantage of its actions over time. Here, then, we have a set of actions we did infact take.

Now, we have a new policy, let's say the START_P_NETWORK, and we train it on the gradients of a trajectory ORIGINAL_TRAJ to make its good actions more likely, and
bad actions less likely. Let's call this 1-batch trained network 1B_P_NETWORK. We can add an additional layer of learning by inspecting what this new policy would
have done along the same rollout trajectory ORIGINAL_TRAJ, comparing its actions to those of the START_P_NETWORK (within the 1B_P_NETWORK trajectory!), and training the network using the 
advantage of the new policy using the old network's value function (START_V_NETWORK). We can then ratio the amount of learning in correspondence to how much better the new
policy is (1B_P_NETWORK) than the old policy (START_P_NETWORK) along the exact same trajectory.

Good actions, defined by a positive advantage, make their own good actions more likely.
Bad actions, defined by a negative advantage, make their own bad actions less likely.
The ratio of new:old policy confidence prediction governs the magnitude of learning. Therefore,
this weighting increase the magnitude of changes that tend toward new policy. if the policy is bad (as judged by advantage), we need to discourage this
action strongly - this ratio weighting is what helps us; because our new policy made the bad action MORE likely, we need to STRONGLY steer away. Conversely, if the
action was good, and the new policy made it more likely, we should steer toward it more!. If the new policy didn't make the new action more or less likely,
we should conservatively be cautious about any weight changes, even if the action was good or bad, because our new policy is not more likely to make it happen.

The combination of the above object - good/bad action (defined on a scale like Value) - and likelihood, looks a lot like an expectation. it is!

Two learning signals for the price of one rollout! Technically, according to the derivation, we should actually be comparing the START_P_NETWORK's actions to the
1B_P_NETWORK's actions along a new trajectory chartered by the 1B_P_NETWORK (rather than using ORIGINAL_TRAJ). But the point of this was rollout efficiency,
and after 1 batch, we can presume sufficient similarity of policy (especially as we are using the old value function, START_V_FUNCTION) that we can be rollout-frugal
and just use the old rollout to guide our work. 

Using the simplification stated above, we can show that the gradients of the objective are identical to the actual objective in the first order,
but not the second order (curvature). This limits that extent to which we can apply learning from the previous rollout trajectory using the surrogate objective - 
we have to use a low learning rate to remain close to the actual gradient we are trying to optimise for.

Also, due to the simplification above, if we want to use the surrogate objective, we should try to constrain the new policy function 1B_P_NETWORK to being
'close' to the old policy START_P_NETWORK, so that the trajectories and actions are similar enough to compare improvements.

**Trust-region policy optimisation**
This leads to the concept of a 'trust region' - defined by KL-divergence - which is a measure of distribution similarity

Trust-region is defined by the KL-divergence of the policy space. So, treating each action output from the START_P_NETWORK and
1B_P_NETWORK as distributions of probabilities (which, due to the softmax activation function, is accurate), KL-divergence measures the alignment
of these distributions, and can be used as a regulariser perhaps?

Here is the algorithm (chapter3-2.png in imgs):

start with a START_NETWORK, and an improved policy network IMPROVED, and a value network
While not converged:
1) Sample n trajectories from START_NETWORK
2) Get gradients that maximise the likelihood of good actions by IMPROVED network, as judged by value network, weighted by the similarity between good and bad actions
Also, constraint this to only optimise over IMPROVED actions that have a low KL-divergence with START_NETWORK actions
3) set the previous IMPROVED to the start network, and the new IMPROVED network to the now-optimised IMPROVED network
4) improve the value network (via TD or monte carlo)
end

**Proximal policy optimisation**
The TRPO is hard to implement because the trust-region is hard to define and not exactly intuitive mathematically due to its attempt to tackle
second-order derivative conjugate gradients for the trust region.

PPO is a pure gradient approach that uses trust-region ideas of "proximal" policy. There are two PPO methods: clipping, and penalty. clipping
literally limits the gradient changes based on KL-divergence, penalty uses KL-divergence as an added term in optimisation, as a regulariser. Most commonly,
nowadays, PPO-clip is used instead of PPO-penalty.

How PPO-clip explicitly works is it limits the output of the ratio of the action-likelihood weight. Remember, this was the weight that we multiplied to the advantage
in the expected return calcation for the policy evaluation (see img chapter3-3.png). The weight was a scalar, and PPO-clip simply limits the variability of this scalar.

Here is the algorithm (chapter3-4.png in imgs):

start with a START_NETWORK, and an improved policy network IMPROVED, and a value network
While not converged:
1) Sample n trajectories from START_NETWORK
2) Get gradients that maximise the likelihood of good actions by IMPROVED network, as judged by value network, weighted by the similarity between good and bad actions but CLIPPED if this weighting deviates too strongly from 1.
3) set the previous IMPROVED to the start network, and the new IMPROVED network to the now-optimised IMPROVED network
4) improve the value network (via TD or monte carlo)
end

The purpose of the PPO is to keep the improved policy CLOSE to the old policy, to allow stable optimisation. In the initial paper, an epsilon of 0.8 was used. This limited the weighting
fraction to between 0.8 and 1.2. Often, we would train over a batch 10 times (repeat some combination of a batches gradients 10 times per batch in PPO)

**Bias-variance tradeoff in advantage calculation**

In temporal difference learning with k-steps, it can be shown that, as you increase k from 1 to infinity (monte carlo), the value of the current
time point should have the same mean, and the variance should increase. This, for me, was counter-intuitive, but it is true because our systems are non-deterministic.
Each state transition requires an interaction between an action and the environment, and the environment is impssible to perfectly predict. It it randomness in the 
environment that produces the greater variability as the number of temporal steps increases.

This has an effect on our temporal difference k choice! the longer we leave TD calculations, the less sure we can be about our reward signals proximity to the true mean.
This is important - if there is randomness in the environment, this corrupts the use of long-distance rewards in our reinforcement learning!

However, if our value estimation network is completely off, then this will be a source of strong bias in the advantage calculation.

Therefore, where the environment has randmoness (or unobserved elements that affect the state), we have low bias/high variability in 
long-distance TD, and high bias/low variability in low-distance variability. Therefore, if we have a random environment/partially observed state,
we should be careful not to treat monte-carlo based solutions/long TD rewards as the gospel.

The discount factor, gamma, is what can be used to minimise this long-term variability from environmental randomness.

How should we choose k, though, and maybe even gamma, to tackle this problem? Especially for k, that can only take discreet integer values.

***Generalised Advantage Estimation (GAE)***
This method is intended to tackle the optimum selection of k throughout the various stages of training a value network

GAE uses an exponential weighted average of all TD advantages (ie, advantage as calculated by various k values).

I find this super boring and dry. It essentially does a discounted sum of advantages over all values of k, where the discounting is aggressively
increased as k increases. This 'averages' the final estimate of advantage as weighted sum of advantages over various values of k, where smaller
values have higher weights.

The GAE calculation uses a different discounting of various values of k, lambda. Large lambda increase variance and reduces bias, by placing
greater emphasis on the later-stage TD estimates; small lambda decreases variance and increases bias, by placing greater emphasis on earlier-stage 
TD estimates.

The result of this is that we have a new parameter to tune - lambda - great. But that, overall, GAE does a better job at calculating advantage.
The temporal difference calculations also, naturally, make uses of gamma-discounting, that reduces the value of latterly-gained rewards.

Common values of lambda and gamma are 0.96 and 0.995, respectively. GAE is used in PPO and TRPO.

**Policy advantage**

This is the notion that of quantifying how much better a different policy is than another policy in the same trajectory; that is to say, we
follow on policy through a rollout and we assess how much more advantageous a different policy would have been in that same rollout.

We generate the trajectory from the original policy. then we assess how much better an alternate policy over the same trajectory would perform.

This is an alternate perspective on interpreting TRPO and PPO. An optimal policy exists when no other policy could produce an advantage over it.
A policy can have no advantage over itself, so this policy advantage is zero.

***Policy iteration***

This is the process of argmax-ing policy decisions; we have some baseline action that can assumed as normal, but we search the action space for the
action that maximises the policy advantage.

In other words, we have a reference policy defined by some network or hash-table, and we seek to improve it via a search. The action argmax of the advantage is identical 
to the action argmax of Q (as V in the advantage calcs does not depend upon action)

To implement policy iteration, we can use importance sampling (?) to form the notion of 'likelihood' in our argmax function. The sampled argmax would
put the best policy as a probability of '1' and the rest at zero, destroying a weighting calculation based on this likelihood. Importance sampling, i imagine,
is what allows us to address uncertainty around the actual argmax? - pick this up later, timepoint 55.08.

This involves sampling trajectories using our reference policy, and searching possible other actions one could have taken,
and seeing those actions through to some temporal difference that allows us to evaluate these alternate-action values.

Alternatively, we could optimise expected advantage directly by training a surrogate neural network at every timestep to maxmise advantage over
a specific trajectory, producing a more-optimal policy for that trajectory, that establishes a direction of travel for gradient learning.

Both of the approaches above fail though, because our next-step policy network needs to be close to our reference policy network, hence
the previously-discussed trust region and PPO-clip etc. In order to solve in this approach, we need to constraint argmax policy iteration
optimisation such that our optimised policy and reference policy are not too dissimilar.

In TRPO and PPO, we learn two networks simultaneously: policy and value.

Why do we need to learn the value network? In order to calculate advantage - in other words,
to assign the correct sign to our gradients during learning (are we going in the right way or not?)
The value function is not used really at all after the policy network is trained...

***No Value Function***

What if we didnt have a value function? (known as 'no baseline') I mean, technically the value is drawn from cumulative rewards. The value function
is essentially just there to estimate those rewards at low-k TD, or with delayed rewards. It provides a useful bias to learn from.

Now, imagine a scenario where we don't have a value function. IF we monte-carlod our reward signal, this is actually identical to no
having no value function - so we don't even need it here! The problem is that monte-carlo could be prohibitively expensive.

Okay, so we attempt to TD without a value function - where does this go wrong? Well, if all of our rewards are stacked at the end of a trajectory,
and our TD learning step k does not include that end, then we essentially get no proper learning signal within a finite small time difference. this is bad!
Statistically, this means that our reward signal has unnacceptably high variance (this is because we are dependent upon very-late stage states for our reward,
and our environment is complex and non-deterministic, AND OUR ACTIONS ARE SEQUENTIALLY STACKED, so our reward will be super variable, 
obfuscating the value of specific early state actions).

The essential problem the value function is trying to solve is: how valuable is a specfic action, that leads to a specific state-change, for long term rewards?
Without a value function, we are trying to assign raw late-stage reward credit to all actions. Late stage rewards are highly variable (like a swaying tower), and
it is not clear which of the bricks that built the tower best-contributed to its enabled height.


This high variance means that either (a) learning is impossible, or (b) learning is extremely slow.

**Group Relative Policy Optimisation**

This attempts to tackle the variance problem, leading to difficulty learning in no-value-function scenarios (Especially with later stage rewards)

This is designed specifically for large language models.

In the example we are learning from, we get ONE reward per trajectory, evaluated only at the end of the trajectory, where there is no
discounting. This is the worst case for learning without a baseline function.

Idea is, we collect N trajectories (and N terminal rewards) with our policy. Then, instead of advantage, our network is trained to predict
(WITHIN THE N BATCHES!!) the relative performance of different scenarios (with some weighting clipping being applied, such that we do not
stray too far from the original policy). In other words, from the same initial condition, we train our model on the relative difference in reward
within a batch.

So, for each batch on N trajectories from the same initial conditions, we get N different terminal rewards. We can then sample the mean and std dev
of these rewards from the batch. We can then normalise the disitrbution of rewards (zero mean, 1 std dev). The reward signal, then, is the normalised
relative performance of different trajectories - the sign of gradients from the better-than-average trajectories are positive, and the
sign of gradients from the worse-than-average trajectories are negative.

Therefore, in GRPO, advantage is computed as the relative performance versus to the average of other policy decisions from the same initial state.
This seems great! But also like it would be a high-noise signal (we stumble into the attribution problem for early states).

GRPO has only been used for LLMs, not other applications. IT isn't the first or only method to forego a baseline function, either.
But, this is open for improvement in the future! It got a lot of attention because of DeepSeek's performance.


**why not other methods?**

well, the action space of LLMs is discrete and finite. Other methods are designed for a continuous action space, so they are not explored on this course.

LLMs only really use the methods above. Methods like DQN seek to train the Q network, and use some greedy search algorithm
to argmax the DQN when deployed. In other words, the policy is not explicitly defined or parameterised, it is defined as the
action that maximises the Q network.

It is not yet clear how we establish a DQN for a pretrained policy like an LLM, or how we might otherwise integrate the idea
of DQN, greedy search, and LLMs, to improve upon GRPO. This is active research!