from environment import TradingEnvironment
import numpy as np

env = TradingEnvironment(ticker="SPY", start="2015-01-01", end="2022-12-31")


print(f"\nState size: {env.state_size}")
print(f"Action size: {env.action_size}")
print(f"Training rows: {len(env.data)}")

state = env.reset()
print(f"\nInitial state shape: {state.shape}")
print(f"First few state values: {state[:5]}")

# Run 5 random steps
for i in range(5):
    action = np.random.randint(0, 3)
    next_state, reward, done = env.step(action)
    env.render()
    print(f"  Action: {['SELL','HOLD','BUY'][action]} | Reward: {reward:.6f} | Done: {done}")

