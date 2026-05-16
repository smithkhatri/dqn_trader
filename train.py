import torch
import numpy as np
from environment import TradingEnvironment
from agent import DQNAgent


def train(
    ticker      = "SPY",
    start       = "2010-01-01",
    end         = "2022-12-31",
    episodes    = 300,
    window_size = 20,
):
    # ─────────────────────────────────────────────────────────────────────────
    # Create the training environment.
    # train=True means we use the first 80% of the data for training.
    # ─────────────────────────────────────────────────────────────────────────
    env = TradingEnvironment(
        ticker      = ticker,
        start       = start,
        end         = end,
        window_size = window_size,
        train       = True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Create the agent, passing in the environment's state and action sizes.
    # This ensures the network is exactly the right shape for our environment.
    # ─────────────────────────────────────────────────────────────────────────
    agent = DQNAgent(
        state_size  = env.state_size,
        action_size = env.action_size,
    )

    best_net_worth = 0
    rewards_log    = []

    for episode in range(1, episodes + 1):
        # ─────────────────────────────────────────────────────────────────────
        # Start a fresh episode. reset() resets portfolio state and returns
        # the initial state vector for the agent to observe.
        # ─────────────────────────────────────────────────────────────────────
        state        = env.reset()
        total_reward = 0
        done         = False

        while not done:
            # ─────────────────────────────────────────────────────────────────
            # 1. Agent observes the state and chooses an action.
            # 2. Environment executes the action and returns the outcome.
            # 3. Agent stores the experience in its replay buffer.
            # 4. Agent samples a batch from the buffer and learns from it.
            # 5. State advances to next_state.
            # This loop is the entire RL training cycle.
            # ─────────────────────────────────────────────────────────────────
            action                          = agent.select_action(state)
            next_state, reward, done        = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            loss                            = agent.learn()
            state                           = next_state
            total_reward                   += reward

        rewards_log.append(total_reward)

        # ─────────────────────────────────────────────────────────────────────
        # Save the model if this episode produced the best portfolio value.
        # torch.save() serializes the network weights to disk.
        # We save main_net.state_dict() — just the weights, not the whole
        # network class — which is the standard PyTorch saving convention.
        # ─────────────────────────────────────────────────────────────────────
        if env.net_worth > best_net_worth:
            best_net_worth = env.net_worth
            torch.save(agent.main_net.state_dict(), "best_model.pth")

        # Print progress every 10 episodes
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_log[-10:])
            print(
                f"Episode {episode:4d} | "
                f"Avg Reward: {avg_reward:8.4f} | "
                f"Net Worth: ${env.net_worth:10.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print(f"\nTraining complete. Best net worth: ${best_net_worth:.2f}")
    return agent, env


if __name__ == "__main__":
    train()