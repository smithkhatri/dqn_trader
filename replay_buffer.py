import random
import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Why do we need a replay buffer?
#
# If the agent learned from each experience immediately and in sequence,
# two problems arise:
#
# 1. Correlation: consecutive experiences are highly correlated — day 2 looks
#    almost identical to day 1. Training on correlated data causes the network
#    to overfit to recent patterns and "forget" older ones.
#
# 2. Instability: learning from each step immediately, one at a time, causes
#    the loss to oscillate wildly. The network chases its own tail.
#
# The solution: store all experiences in a large pool and sample RANDOM
# mini-batches from it. Random sampling breaks correlation. Batching stabilizes
# the gradient updates. This is one of the two key innovations of DQN
# (the other being the target network in agent.py).
# ─────────────────────────────────────────────────────────────────────────────


class ReplayBuffer:

    def __init__(self, capacity=100_000):
        # ─────────────────────────────────────────────────────────────────────
        # deque = double-ended queue. Like a list but optimized for appending
        # and removing from either end.
        #
        # maxlen=capacity is the key feature: when the deque is full and you
        # append a new item, it automatically removes the OLDEST item from the
        # other end. No manual management needed — it's a sliding window of
        # the most recent 100,000 experiences.
        #
        # Why 100,000? Large enough to have diverse experiences from many
        # different market conditions. Small enough to fit in RAM.
        # ─────────────────────────────────────────────────────────────────────
        self.buffer = deque(maxlen=capacity)


    def push(self, state, action, reward, next_state, done):
        # ─────────────────────────────────────────────────────────────────────
        # Store one experience tuple in the buffer.
        # An experience is everything that happened in one step:
        #   state      — what the agent observed before acting
        #   action     — what the agent decided to do (0, 1, or 2)
        #   reward     — how good or bad that decision was
        #   next_state — what the agent observed after acting
        #   done       — whether the episode ended after this step
        #
        # Together these are called a "transition" in RL literature.
        # The Bellman equation needs all 5 pieces to compute the target Q-value.
        # ─────────────────────────────────────────────────────────────────────
        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size):
        # ─────────────────────────────────────────────────────────────────────
        # Random sample batch_size experiences from the buffer.
        #
        # random.sample(population, k) returns k UNIQUE items chosen randomly
        # from population. No repeats within a single batch.
        #
        # zip(*batch) is a Python trick to transpose a list of tuples.
        # If batch = [(s1,a1,r1), (s2,a2,r2), (s3,a3,r3)]
        # then zip(*batch) = [(s1,s2,s3), (a1,a2,a3), (r1,r2,r3)]
        # This lets us unpack all states together, all actions together, etc.
        # ─────────────────────────────────────────────────────────────────────
        batch                                      = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # ─────────────────────────────────────────────────────────────────────
        # Convert each group to a numpy array.
        # The neural network and PyTorch expect numpy arrays (which then get
        # converted to tensors in the agent).
        #
        # np.array(states)      → shape (batch_size, state_size)
        # np.array(actions)     → shape (batch_size,)
        # np.array(rewards)     → shape (batch_size,)
        # np.array(next_states) → shape (batch_size, state_size)
        # np.array(dones)       → shape (batch_size,) — bool/float values
        # ─────────────────────────────────────────────────────────────────────
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )


    def __len__(self):
        # ─────────────────────────────────────────────────────────────────────
        # __len__ is a special Python method. Defining it lets you call
        # len(buffer) on your object, just like len(list) or len(string).
        # We use this in the training loop to check if the buffer has enough
        # experiences before we start sampling from it.
        # ─────────────────────────────────────────────────────────────────────
        return len(self.buffer)