# Change is Brewing: An RL-based solution to dynamic pricing models in local coffee shops

This project implements a custom reinforcement learning environment for simulating dynamic pricing at a campus coffee shop. Using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and the `gymnasium` framework, we train a PPO agent to set coffee prices throughout the day to maximize revenue under realistic demand fluctuations.

Customers at a campus coffee shop respond to price, time of day, weather, and day of the week (weekday vs. Friday). We simulate customer demand across 20 time intervals (8am–6pm in 30-minute steps), model price sensitivity using logistic functions, and compare PPO-learned pricing strategies to fixed-price baselines.

## Features of Our Algorithm

- **Custom Gym environment** (`CoffeePricingEnv`) with:
  - Price elasticity per time slot
  - Rain and weekday/weekend variation
  - Customer volume tied to survey-based estimates
  - Temporal features including sinusoidal encodings
- **Reinforcement Learning with PPO**
  - Optimized using `optuna`
  - Reward includes revenue, entropy bonus, and price variance penalty
- **Baseline comparison**
  - Evaluates all fixed prices from \$3.00 to \$8.00
  - Tracks PPO performance relative to a \$5.00 baseline
- **Visualization**
  - Smoothed PPO training performance over time
  - Revenue comparison across fixed vs. learned policies
  - Daily price/demand curves

## Getting Started

### 1. Install dependencies

```bash
pip install gymnasium stable-baselines3 matplotlib numpy torch optuna
