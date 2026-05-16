import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import TradingEnvironment
from agent import DQNAgent
from model import DQNNetwork


def evaluate(
    ticker      = "SPY",
    start       = "2010-01-01",
    end         = "2022-12-31",
    window_size = 20,
    model_path  = "best_model.pth",
):
    # ─────────────────────────────────────────────────────────────────────────
    # Create the TEST environment — train=False uses the last 20% of data
    # that the agent has never seen during training.
    # This is the honest evaluation of whether the agent actually learned
    # something general, or just memorized the training data.
    # ─────────────────────────────────────────────────────────────────────────
    env = TradingEnvironment(
        ticker      = ticker,
        start       = start,
        end         = end,
        window_size = window_size,
        train       = False,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Recreate the agent and load the saved best weights.
    # torch.load loads the state dictionary from disk.
    # map_location=device handles the case where you trained on GPU (Colab)
    # but are evaluating on CPU (your Mac) — it remaps the tensors correctly.
    # load_state_dict() copies those weights into the network.
    # ─────────────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.main_net.load_state_dict(torch.load(model_path, map_location=device))

    # ─────────────────────────────────────────────────────────────────────────
    # model.eval() switches the network to evaluation mode.
    # This disables Dropout — all neurons are active at full strength.
    # During training, dropout randomly zeros neurons to prevent overfitting.
    # During evaluation, we want the full network making predictions.
    # ─────────────────────────────────────────────────────────────────────────
    agent.main_net.eval()

    # ─────────────────────────────────────────────────────────────────────────
    # Force epsilon to 0 — pure exploitation, zero random exploration.
    # During evaluation we want to see what the agent has actually learned,
    # not random actions mixed in.
    # ─────────────────────────────────────────────────────────────────────────
    agent.epsilon = 0.0

    # ─── Run one full episode on test data ────────────────────────────────────
    state        = env.reset()
    done         = False
    agent_values = [env.initial_cash]  # track portfolio value each day
    actions_log  = []                   # track what action was taken each day

    while not done:
        action              = agent.select_action(state)
        next_state, _, done = env.step(action)
        state               = next_state
        agent_values.append(env.net_worth)
        actions_log.append(action)

    # ─── Compute buy-and-hold baseline ────────────────────────────────────────
    # Buy-and-hold: invest all cash on day 1, hold until the end.
    # This is the benchmark every trading strategy must beat to be meaningful.
    # If your agent can't beat buy-and-hold, it's not adding value.
    initial_price  = env.prices.iloc[env.window_size]
    bah_values     = [
        env.initial_cash * (env.prices.iloc[i] / initial_price)
        for i in range(env.window_size, env.window_size + len(agent_values))
    ]

    # ─── Performance metrics ──────────────────────────────────────────────────
    agent_returns  = np.diff(agent_values) / np.array(agent_values[:-1])
    total_return   = (agent_values[-1] - agent_values[0]) / agent_values[0] * 100

    # Sharpe Ratio: risk-adjusted return.
    # = (mean daily return / std of daily returns) * sqrt(252)
    # sqrt(252) annualizes it — 252 trading days in a year.
    # Higher is better. >1 is decent. >2 is very good.
    sharpe         = (np.mean(agent_returns) / (np.std(agent_returns) + 1e-9)) * np.sqrt(252)

    # Max Drawdown: largest peak-to-trough drop in portfolio value.
    # Measures how bad the worst losing streak was.
    # Lower (less negative) is better.
    peak           = np.maximum.accumulate(agent_values)
    drawdown       = (np.array(agent_values) - peak) / peak
    max_drawdown   = drawdown.min() * 100

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS ON TEST DATA")
    print(f"{'='*50}")
    print(f"Agent final value:      ${agent_values[-1]:,.2f}")
    print(f"Buy & Hold final value: ${bah_values[-1]:,.2f}")
    print(f"Total return:           {total_return:.2f}%")
    print(f"Sharpe ratio:           {sharpe:.3f}")
    print(f"Max drawdown:           {max_drawdown:.2f}%")

    # ─── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Top chart: portfolio value comparison
    ax1.plot(agent_values, label="DQN Agent",      color="blue",  linewidth=1.5)
    ax1.plot(bah_values,   label="Buy & Hold",     color="gray",  linewidth=1.5, linestyle="--")
    ax1.set_title("Portfolio Value — Agent vs Buy & Hold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom chart: price with buy/sell markers
    prices = env.prices.iloc[env.window_size:env.window_size + len(actions_log)].values
    ax2.plot(prices, color="black", linewidth=1, label="Price")

    # Mark buy actions with green up-triangles, sell actions with red down-triangles
    for i, action in enumerate(actions_log):
        if action == TradingEnvironment.BUY:
            ax2.scatter(i, prices[i], marker="^", color="green", s=50, zorder=5)
        elif action == TradingEnvironment.SELL:
            ax2.scatter(i, prices[i], marker="v", color="red",   s=50, zorder=5)

    ax2.set_title("Price Chart with Agent Actions")
    ax2.set_ylabel("Price ($)")
    ax2.set_xlabel("Trading Days")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=150)
    plt.show()
    print("\nChart saved as evaluation.png")


if __name__ == "__main__":
    evaluate()