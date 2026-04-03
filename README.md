# dqn_trader
A DQN agent that reads stock market data, decides to buy / hold / sell, and learns over thousands of simulated trading days which decisions make it the most money.

It learns entirely by trial and error — no human tells it the right answer. It just gets rewarded or punished based on what happens to its portfolio.

The 6 building blocks we'll build in order
1. The Environment — the "world" the agent lives in. It feeds the agent market data, accepts its decisions, and hands back a reward. This is the most important piece and where lookahead bias can destroy you if you're not careful.
2. The Neural Network — a simple feedforward net that takes in market state and outputs Q-values (scores for each action).
3. The Replay Buffer — a memory bank that stores past experiences so the agent can learn from them later, in random batches.
4. The DQN Agent — the brain that ties the network, buffer, and decisions together. Includes epsilon-greedy exploration.
5. The Training Loop — the engine that runs episodes, collects experience, and triggers learning.
6. Evaluation & Visualization — plotting performance vs. a buy-and-hold baseline so you can actually see if your agent learned something real.


What is a Q-value?
Q(state, action) = "how much total future reward do I expect if I take this action right now, and then play optimally forever after?" The network learns to predict these. The agent just picks the action with the highest Q-value.
What is the Bellman equation?
This is how the agent updates its beliefs. If you take action A in state S, get reward R, and land in state S', then:

Q(S, A) = R + γ × max(Q(S', all actions))

γ (gamma) is the discount factor — future rewards are worth a little less than immediate ones. This single equation is the backbone of all DQN learning.
What is epsilon-greedy?
Early in training the agent knows nothing, so it explores randomly. Over time it shifts toward exploiting what it has learned. Epsilon is the probability of taking a random action. We start it at 1.0 (fully random) and decay it toward 0.01.
What is the replay buffer?
If the agent learned from each experience immediately and in order, it would overfit to recent data catastrophically. Instead we store experiences in a big pool and sample random mini-batches to train on. This breaks the correlation between consecutive experiences.
What is the target network?
A second copy of the neural net, frozen for N steps, used only to compute the "target" Q-value in the Bellman equation. Without it, you're chasing a moving target — the network is trying to predict values produced by itself, which causes training to spiral. The target net gets synced to the main net every N steps.