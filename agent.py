import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import DQNNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    # ─────────────────────────────────────────────────────────────────────────
    # The agent is the brain. It owns the neural network, the replay buffer,
    # and all the logic for deciding what to do and learning from experience.
    #
    # The two core methods are:
    #   select_action() — given a state, choose an action
    #   learn()         — sample from buffer and update network weights
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        state_size,
        action_size,
        lr           = 1e-4,     # learning rate — how big each weight update step is
        gamma        = 0.99,     # discount factor — how much future rewards are valued
        epsilon      = 1.0,      # starting exploration rate — 1.0 = fully random
        epsilon_min  = 0.01,     # minimum exploration rate — always explore a little
        epsilon_decay= 0.995,    # how fast epsilon shrinks each step
        batch_size   = 64,       # how many experiences to learn from at once
        buffer_size  = 100_000,  # max experiences stored in replay buffer
        target_update= 100,      # sync target network every N steps
    ):
        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps_done    = 0   # counts total steps taken across all episodes

        # ─────────────────────────────────────────────────────────────────────
        # Device selection.
        # torch.cuda.is_available() returns True if a CUDA GPU is present.
        # On Colab with GPU runtime, this will be True and training will be
        # dramatically faster. On your Mac locally, this will be False and
        # PyTorch uses CPU. The code works identically on either.
        # ─────────────────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")

        # ─────────────────────────────────────────────────────────────────────
        # Two networks — this is the key architectural decision in DQN.
        #
        # main_net: the network being actively trained. Makes predictions,
        #   receives gradient updates every step.
        #
        # target_net: a frozen copy of main_net. Used ONLY to compute the
        #   "target" Q-value in the Bellman equation. Gets synced to main_net
        #   every target_update steps but is otherwise frozen.
        #
        # Why two networks?
        # If we used main_net for both predicting AND computing targets,
        # we'd be chasing a moving target — the network would be trying to
        # predict values that it itself keeps changing every step.
        # This causes catastrophic instability (the loss explodes or never
        # converges). The frozen target_net provides stable training targets.
        # ─────────────────────────────────────────────────────────────────────
        self.main_net   = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)

        # Copy main_net's weights into target_net so they start identical.
        # state_dict() returns a dictionary of all parameters (weights, biases).
        # load_state_dict() loads those parameters into another network.
        self.target_net.load_state_dict(self.main_net.state_dict())

        # ─────────────────────────────────────────────────────────────────────
        # target_net.eval() puts it in evaluation mode permanently.
        # This disables dropout and batch norm updates.
        # We never train the target net — it only provides stable predictions.
        # ─────────────────────────────────────────────────────────────────────
        self.target_net.eval()

        # ─────────────────────────────────────────────────────────────────────
        # Optimizer: Adam (Adaptive Moment Estimation).
        # The optimizer is responsible for updating the network's weights
        # based on the computed gradients.
        #
        # Adam is the standard choice for deep learning. It adapts the
        # learning rate for each parameter individually using estimates of
        # first and second moments of the gradients. More robust than plain
        # gradient descent — works well without much tuning.
        #
        # lr=1e-4 means each update step moves weights by at most 0.0001
        # in the direction that reduces loss. Too large → unstable.
        # Too small → trains forever.
        #
        # We only pass main_net's parameters — we never optimize target_net.
        # ─────────────────────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)

        # ─────────────────────────────────────────────────────────────────────
        # Huber loss (SmoothL1Loss in PyTorch).
        # Loss measures how wrong the network's Q-value predictions are.
        # We want to minimize this during training.
        #
        # Why Huber and not plain MSE (mean squared error)?
        # MSE squares the error — large errors get enormous gradients which
        # can destabilize training. Huber loss acts like MSE for small errors
        # but like absolute error for large errors. This clips extreme
        # gradients and makes training more stable in noisy environments
        # like financial markets.
        # ─────────────────────────────────────────────────────────────────────
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)


    def select_action(self, state):
        # ─────────────────────────────────────────────────────────────────────
        # Epsilon-greedy action selection.
        # This is how the agent balances exploration vs exploitation.
        #
        # Exploration: take a random action. Discovers new strategies.
        #   Necessary early in training when the network knows nothing useful.
        #
        # Exploitation: use the network to pick the best known action.
        #   Necessary later when the network has learned something valuable.
        #
        # Epsilon starts at 1.0 (always random) and decays toward 0.01
        # (almost always use the network). The agent gradually shifts from
        # exploring randomly to exploiting what it has learned.
        # ─────────────────────────────────────────────────────────────────────
        if np.random.random() < self.epsilon:
            # Random action — uniform choice among SELL, HOLD, BUY
            return np.random.randint(0, self.action_size)

        # ─────────────────────────────────────────────────────────────────────
        # Greedy action — let the network decide.
        #
        # torch.FloatTensor(state) converts numpy array to PyTorch tensor.
        # .unsqueeze(0) adds a batch dimension: (state_size,) → (1, state_size)
        #   The network expects (batch_size, state_size) — even for one sample.
        # .to(self.device) moves the tensor to GPU if available.
        #
        # torch.no_grad() tells PyTorch not to track gradients here.
        #   We're just making a prediction, not training. Skipping gradient
        #   tracking saves memory and speeds up inference significantly.
        #
        # .argmax(1) returns the index of the highest Q-value along dimension 1
        #   (across actions). That index IS the action (0=SELL, 1=HOLD, 2=BUY).
        # .item() converts a single-element tensor to a plain Python integer.
        # ─────────────────────────────────────────────────────────────────────
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_net(state_tensor)
        return q_values.argmax(1).item()


    def store_experience(self, state, action, reward, next_state, done):
        # Simple wrapper — push one transition into the replay buffer.
        self.memory.push(state, action, reward, next_state, done)


    def learn(self):
        # ─────────────────────────────────────────────────────────────────────
        # The learning step — the heart of DQN.
        # Called every step of training once the buffer has enough experiences.
        #
        # The goal: adjust main_net's weights so its Q-value predictions
        # get closer to what the Bellman equation says they should be.
        # ─────────────────────────────────────────────────────────────────────

        # Don't learn until we have enough experiences to sample a full batch
        if len(self.memory) < self.batch_size:
            return None

        # ─── Sample a batch from the buffer ───────────────────────────────────
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # ─── Convert numpy arrays to PyTorch tensors ──────────────────────────
        # Everything needs to be a tensor on the correct device for the network.
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # ─── Compute current Q-values ─────────────────────────────────────────
        # main_net(states) → shape (batch_size, action_size) — Q-values for ALL actions
        # .gather(1, actions.unsqueeze(1)) selects only the Q-value for the
        #   action that was actually taken in each experience.
        #   gather() picks values along a dimension using indices.
        #   actions.unsqueeze(1) reshapes (batch_size,) → (batch_size, 1)
        #   so gather knows which column (action) to pick from each row (experience).
        # .squeeze(1) removes the extra dimension → back to (batch_size,)
        current_q = self.main_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ─── Compute target Q-values using the Bellman equation ───────────────
        # The Bellman equation is the mathematical foundation of all Q-learning:
        #
        #   Q(s, a) = r + γ * max_a'[Q(s', a')] * (1 - done)
        #
        # In plain English:
        # "The true value of taking action a in state s equals the immediate
        #  reward r PLUS the discounted value of the best possible action in
        #  the next state s'."
        #
        # γ (gamma=0.99) is the discount factor. Future rewards are worth
        # slightly less than immediate rewards — a dollar today beats a dollar
        # next year. 0.99 means we value tomorrow's reward at 99% of today's.
        #
        # (1 - done) zeros out the future term when the episode is over —
        # there IS no future state after a terminal state, so future reward = 0.
        #
        # torch.no_grad() — we're computing targets, not training target_net.
        # target_net(next_states) → Q-values for all actions in next state.
        # .max(1)[0] → the highest Q-value across actions, shape (batch_size,)
        with torch.no_grad():
            next_q       = self.target_net(next_states).max(1)[0]
            target_q     = rewards + self.gamma * next_q * (1 - dones)

        # ─── Compute loss ─────────────────────────────────────────────────────
        # Loss measures how far off our current predictions are from the targets.
        # We want current_q to match target_q as closely as possible.
        loss = self.loss_fn(current_q, target_q)

        # ─── Backpropagation ──────────────────────────────────────────────────
        # This is how the network actually learns — three lines that do
        # everything:
        #
        # optimizer.zero_grad(): clear gradients from the previous step.
        #   Gradients accumulate by default in PyTorch — we must clear them
        #   manually before each backward pass.
        #
        # loss.backward(): compute the gradient of the loss with respect to
        #   every weight in main_net. PyTorch traces back through every
        #   operation in the forward pass and applies the chain rule.
        #
        # clip_grad_norm_: gradient clipping. Caps the total gradient magnitude
        #   at 1.0. Prevents "exploding gradients" — a situation where one
        #   very wrong prediction causes a massive weight update that destroys
        #   everything the network has learned. Common in RL with noisy rewards.
        #
        # optimizer.step(): update every weight by moving it a small amount
        #   in the direction that reduces the loss. The learning rate controls
        #   how big that step is.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1.0)
        self.optimizer.step()

        # ─── Decay epsilon ────────────────────────────────────────────────────
        # After each learning step, reduce epsilon slightly.
        # max() ensures it never goes below epsilon_min.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ─── Sync target network ──────────────────────────────────────────────
        # Every target_update steps, copy main_net's weights into target_net.
        # This gives the target net a fresh snapshot to provide stable targets
        # for the next N steps.
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()