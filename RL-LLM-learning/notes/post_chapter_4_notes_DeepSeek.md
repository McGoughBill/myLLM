*Summarising my understanding of GRPO used in DeepSeek paper*

My understanding of GRPO is that we use the relative performance as the training signal, with the
average performance being used as substitute for the value function.

This means that we see each trajectory out until its complete terminal state, and get an undiscounted final reward for each trajectory.

variations in trajectory come from noise in the policy, AKA temperature. What happens is, we let the LLM loose on the same initial state
with some non-zero temperature, and generate a set of trajectories that varies slightly.

Then, using the importance sampling, we collect how much more likely our updatable policy is than our old policy at generating each token (each action)
within each trajectory as a ratio of the updatable policy and the old policy. On the first batch, this will be 1 for each token in every trajectory,
as the updatable policy IS the old policy.

Then, we collect the relative reward (normalised by all reward values). The mean of the reward value approximately represents the value of the initial state,
and the variation represents how much better a model performed than the mean.

Our objective is then SUM of products of the improved likelihoods (on first batch this is just 1) and the normalised reward.

Then, we want to optimise the objective by adjusting our trainable policy. To do this: differentiate the objective by the trainable
network weights, and adjust our network using these gradients. Which elements of the objective are differentiable? Well, the rewards
are gunna stay identical within a rollout - they were technically dictated by 'old policy'. So, too, were the probabilities generated
by 'old policy'. Therefore the only differentiable term is the trainable policy's parameters. The rewards are a scalar to tell us how much
more/less of these gradients we should use relative to each other trajectory.

So, we backprop through the differentiated per-token probabilities (now, becoming log probabilities for stability) in each trajectory to get the gradients that generate each of the tokens given the
previous information. Then, we uniformly attribute more of those gradients if they were in a well-rewarded trajectory, and uniformly
take away the gradients of poorly-performing trajectories.

This is a HELL of a lot of memory. one gradient for each parameter in the network (could be billions, or trillions) and for each
token generated in the then LLM response. Then, there is the credit attribution problem - we are just uniformly upvoting all tokens in
the good answers... great. 

Ahwell, this is GRPO.

We repeat the training steps with the same set of rollouts, but this time grabbing new activations using our trainable network. 
after the first batch, the tokens in the good events should become more likely, leading 
to a necessary reweighting of likelihoods to keep the new network accurately learning from the old rollouts (as we are using the same
outcomes and reward values from the old policy) using the ratio of likelihoods from the trainable policy to the old policy.
