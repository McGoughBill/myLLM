*Deep Policy Evaluation*

I imagine this is the process of evaluating the expected value of an action (dictated by a policy)
using a deep neural network.

This lecture will cover how to evaluate a policy choice, and then how to optimise a policy.

**First method: Monte Carlo policy evaluation**

We simply to a rollout of n independent trajectories, and compute the cumulative reward of each trajetctory
The value of a state is the average (??) of the cumulative value over all trajectories - what? Why?

If the set S is small, and the number of possible states is small, we can calculate the value of all possible states.

When the state space is large or infinite, we cannot resolve the value function for all possible states. Instead, we
can only approximate the value function with a small number of states. We do this approximation using a neural network.

So we approximate Vpi (the true value of a state selected by our policy) with Vphi (a neural network). Conceptually, the loss
function is the MSE between the value of a state (predicted by a neural network) and the true value of a state. 

The problem with this conceptual loss function is that we require access to the true value function still, as our learning signal. 
I may have lost some understanding, but apparently the gradient of the true value function is tractable?

Here is how the value function approximation works (using a neural network):

While Vpi != Vphi:

1) Sample trajectory from an initial state and a policy (policy could be an NN, could be a look-up table) and complete until terminal state.
2) Compute g using a trajectory: g = (Vphi(state 0) - (discounted cumulative value up until terminal state)) * (the gradients in the neural network wrt. value at state 0)
3) Update neural weights using g with an optimizer. g is a stochastic gradient upon every Vphi parameter.

end

If we initially evaluate the value of state zero too highly, then this means we need to reduce our predictions next time. Hence,
we need to reduce our parameters according to the gradients of our neural network wrt. to value. If the value was too low, then we need to predict
higher, this flips the polarity of the gradients in the opposite directions - we need to go higher!

Here is the expanded version of the same algorithm...

While Vpi != Vphi:

1) select our current state as an initial state, set time = 0
2) while not currently in terminal state:
   1) evaluate the action of from our current state using the policy
   2) find the next state and resultant reward from this action, set this as our current state
   3) set t = t+1
3) t = T, terminal time
4) Calculate g as stated before
5) update vphi using g, parameter-wise gradients on value estimation.

Essentially, g, the gradient of Vphi with respect to its predicted value, measures how much each parameters contributes to making the value higher or lower. This tells
us in which direction we should tweak the parameters in order to get a higher or lower value. We get the true value function from the complete rollout of a system through its trajectory - we manually assess it -
for individual states, one at a time! Or, batched, but the premise remains the same; we actually calculate value manually by seeing our model
through to completion. Then, the difference between our original estimate of value and our final actual assessment of value is our error that we should use to guide weight adjustment.

Monte-carlo policy evaluation requires complete rollout, which is relatively inefficient.

**Second method: Temporal difference policy evaluation**

In temporal-difference learning, we use the reward of a single transition.

Because of the one-step transition property, the value of state 0 SHOULD be equal to the reward at zero + value of state one.
Therefore, one-step transition looks at taking an arbitrary number of single steps, calculates the next-step value,
and uses this as a learning signal for the state 0 evaluation. 

We average all the values of the next step from state zero. Are these next steps all taken from our policy? 

However, inorder to use next-step value as a signal, we still need a true value function. This is a problem!
If we rely on the true value function, for the next step evaluation as well, this remains intractable.

Instead, we approximate the true value function with a neural network - the value network. So, the value network guesses
value at state 0, then we take the policy's action and proceed to the next state, state 1. In state 1, we receive some reward and re-evaluate
value. The state difference between the state 1 value + reward and the state 0 value should be 0. Differences from zero are learned
as incorrect! So the error is the difference between value 0 and (value 1 + reward at 1). In other words, our target is the reward,
that the network is predicting as the difference between two temporal states.

Temporal difference has no access to the true value of a state, instead using temporal differences in reward as a signal that should be learned.
As a result, this algorithm can breakdown in practical settings. This algorithm especially breaks down where rewards are not distributed across a trajectory/
episode, but held out until the end of the episode.

***How should we compute the approximate stochastic gradient above?***

Option 1: Calculate the reward (target; scalar), the value of state 0 and 1 (predictions; scalar), and the gradients of value state 0 with respect to
the value network's weights (vector). This is fairly trivial to understand, and correct, but apparently cumbersome relative to option 2. This is the
biased stochastic gradient approximation.

Option 2: We can forward evaluate the MSE of the reward and the values at state 0 and 1 and backprop as normal across the loss function.
This is actually incorrect, essentially because if we does this we accidentally differentiate the value network at state 1 as well. This is incorrect.
We are only optimising for the value function at state 0, as we only have the reward at state 1.

The insight comes from the fact that we have no actual real label or real value function. We have an approximated target
using the neural network that we are using to guess the answer in the first place. Therefore, if we just differentiate the MSE as we do in supervised learning,
we will pull out the network's gradients with respect to value at state 1 AND two. This is incorrect!! 
We only want the gradients of the value network for evaluating value at state 0, as this is the only point for which we have a supervision label.

The state one's gradients between value and network weights are treate as irrelevant for this task, but will get averaged into the pytorch,
adding noise and ruining an already-filmsy optimisation procedure.

IF we leave the gradient in at state one, then what we are trying to do is optimising the value at state 0 and state 1 to pull-together with a distance of 'reward'.
IF we stop gradient at state 0, we are saying it is state 0's guess that is more-wrong, and therefore we should only pay attention to pulling state 0's value assignment
toward a fixed state 1 value assignment.

To make TD optimisation work with MSE, we have to prevent gradient calculation ( .detach() ) during the value prediction at state 1.

****An aside: semi gradient commentary****

Technically, both option 1 and 2 of the gradient approximator  is actually not technically a gradient descent, because
it is not a gradient of any true function apparently. So we call this a semi-gradient method. Probably, this is a result of a lack of
true labels - we are not descending any true loss curve, just a presumed one.

If we get rid of the stop-gradient operator on the value prediction at state 1, we arrive at an actual gradient method, apparently. This
 is the equivalent of learning the minimising error of value assignment, the bellman error. But this has tended to perform worse in practical settings,
apparently.

**Third method: k-step policy evaluation**

I imagine this is just the difference between value at an arbitrary depth (not just one time point/action in the future).

I suppose k-step is a generalisation between monte-carlo (infinite step) and 1-step.

Have to take into account that we may accidentally terminate before k-steps too, in which this does just become the Monte-Carlo estimation.

Then, to fit a value function with an objective, we are trying to predict the difference between value at an arbitrary time step and
time step 0, at state zero, using a much-better elaborated reward. that has been collected inbetween across several steps. To calculate the target,
we must apply a summation of gamma-reductive rewards and then finally the gamma-reduced value estimate (from the value function) at our state k. Remember
to apply the stop-gradient at value evaluation at step k!


**Monte-Carlo (MC) vs Temporal Difference (TD, or bootstrapping)**

The goal of both networks is to train a network to accurately learning the value function.

TD completely replaces the concept of the value function with the value network. MC completes the value
assessment by calculating value at each state until terminal state.

Early on in the training loop, bootstrapping can cause instabilities, especially where rewards are heavily delayed.

Later on in the training loop, our network becomes pretty good, and so waiting until completion to learn from rewards
is unnecessarily arduous.

BTW; as a rule of thumb, k-step TD is usually about 5-steps into the future.

**Policy evaluation with the Q function instead**

This is basically the same as in the V-function case, but remember we are also conditioning value upon a selected action within a state.

We either see actions until the end with a given policy, and get a final reward value that way (MC), or we replace the notion of a true valeu function
with the Q value network-action pair (following a given policy) and use the reward as our signal, which should be equal to the temporal difference of value estimates
at state0,action0 and state_k,action_k.