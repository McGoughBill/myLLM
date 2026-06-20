*AlphaGo, test-time compute, and expert iteration*

**AlphaGo - 2-player zero sum games**

In a two-player, zero sum game, the advantage gained by one player is exactly
the same as the loss attributable to the other player's moves.

Chess, Rock-paper-scissors.

Difference between chess and RPS: sequential vs. simultaneous discreet game choices.
Players take turns in chess, they move identically in time in RPS.

Minimax optimisation: we have a single reward system, and we are trying
to minimise one set of parameters relative to the reward, and maximise another
set of parameters relative to the reward:

classic example: GAN. We are trying to maximise discrimination error with the generator,
and minimise discrimination error with the discriminator. This sets up an oppositional 
'dual network' training over an identical scenario, as if the generator is playing AGAINST the discriminator.

Nash equilibrium: if deviations in the minimising agent's policy increases the reward
AND simultaneously deviations in the maximising agent's policy decreases the reward. In summary,
if either policy loses from a unilateral change in policy, then they have achieved a settled solution.

GANs are quite hard to stabilise, in a training optimisation sense. Many two-player zero-sum
games are very hard to stabilise, apparently. Even more delicate than single-agent RL, as you can imagine.

**Simple algorithm: SGD-A**

Simultaneous gradient descent-ascent (ascent and descent on each of the minimax algorithms).

This is so unstable though! Most common failure mode: cycling dynamics. This is where one agent thinks it spots
and exploitable bias in the opposing agent, and adapts its policy to exploit that bias. this generates a real bias in its
own policy exploitable by the other agent, which then is overcorrected by the first agent, etc. etc.

***problem***
The problem here is off too-rapid divergence of policy, and hasty assumption of bias in the opposing policy. I play rock,
so you DEFINITELY play paper next round (whereas mine is random), but now i see that for maybe two games in a row and I DEFINITELY
play scissors, now maybe for 3 games in a row, etc. etc. The problem is that cyclical dynamics poison the training data.

***solution #1***
This is wild. For every timestep, we calculate a gradient updated policy BUT DO NOT MAKE THIS OUR CANONICAL POLICY.
Instead, we make this our temporary policy, and calculate the gradient of our temporary policy with respect to the same move.
Essentially, we perform gradient update as normal, but then imagine taking the same move in the same position and using THIS as our gradient update.

this is called 'extragradient method', and it essentially involves not committing to the first gradient update, but instead
evaluating how this update would influence performance in the same scenario AS WELL AS how the UPDATE OPPONENT would perform
in this scenario

Weird - we are using gradient from a temporary network to influence the policy of the original network - a different policy. But this works
because we are seeing what our opponent took from our behaviour and learning to negate that.

***solution #2***
More intuitive - this is weight decay. Essentially, we are progressively decreaseing the parameter
values by some small amount (0.99), regardless of how good they are. This essentially acts as a regulariser.

**Antisymmetric two player payoff games**


