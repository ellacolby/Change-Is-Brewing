import gymnasium as gym
from gymnasium import spaces
import numpy as np
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import copy
import optuna
# The following code uses code from Stable-Baselines-3 for PPO algorithm and BaseCallback. We also Optuna for hyperparameter tuning. 
# Both of the github repositories for the source code are referenced in our final report. 

# allows us to control a variance penalty throughout timesteps
GLOBAL_STEP = 0


# keeps track of the fixed price reward for every day we train model on
class RewardTrackerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ppo_rewards = []
        self.baseline_rewards = []

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            # gets the ppo revenue from the model, easy bc already done
            info = self.locals["infos"][0]
            ppo_revenue = info.get("total_revenue", 0.0)

            # we chose fixed price of $5.00 bc high reward
            fixed_price_cents = 500              

            # gets $5.00 index (sub 300 bc of our bounds)
            baseline_index = (fixed_price_cents - 300) // 10   

            # make a matching clone and run it
            env = self.training_env.envs[0]
            if hasattr(env, "env"):
                # if Monitor, unwrap
                env = env.env 

            # clone the baseline environment and run a day
            baseline_env = env.get_baseline_clone(fixed_price_cents)
            done         = False
            while not done:
                _, _, done, _, _ = baseline_env.step(baseline_index)

            # record results
            baseline_revenue = baseline_env.total_revenue
            self.ppo_rewards.append(ppo_revenue)
            self.baseline_rewards.append(baseline_revenue)

            print(f"Appending PPO revenue: {ppo_revenue:.2f}, "
                  f"Baseline ($5.00): {baseline_revenue:.2f}")
        return True


# performs an evaluation of the PPO model every 5000 steps
class EvalRolloutCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_counter = 0

    # only performs at right eval frequency
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.eval_counter += 1
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = self.eval_env.step(action)
            print(f"\n Evaluation Rollout #{self.eval_counter} at step {self.n_calls}")
        return True

# over the course of all of the episodes, plots PPO's performance
# compared to the baseline of $5.00 with a smoothing window
def plot_reward_history(ppo_rewards, baseline_rewards=None, smoothing_window=1000):

    # takes in the moving 100 episode average of the difference
    smoothed = []
    for i in range(len(ppo_rewards)):
        start = max(0, i - smoothing_window + 1)
        smoothed.append(np.mean(ppo_rewards[start:i+1]))


    plt.figure(figsize=(10, 5))

    # plots the smoothed gains or losses over the fixed price
    if baseline_rewards is not None:
        gains = [ppo - base for ppo, base in zip(ppo_rewards, baseline_rewards)]
        smoothed_gains = []
        for i in range(len(gains)):
            start = max(0, i - smoothing_window + 1)
            smoothed_gains.append(np.mean(gains[start:i+1]))

        plt.plot(np.array(gains), label='PPO Gain Over Baseline', alpha=0.4)
        plt.plot(np.array(smoothed_gains), label=f'Smoothed Gain (window={smoothing_window})', linewidth=2)
        plt.ylabel("Gain Over Fixed Price ($)")
    else:
        plt.plot(ppo_rewards, label='Episode Reward', alpha=0.4)
        plt.plot(smoothed, label=f'Smoothed (window={smoothing_window})', linewidth=2)
        plt.ylabel("Total Reward ($)")


    plt.xlabel("Episode")
    plt.title("PPO Performance Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# definition of our simulator of the coffee shop
class CoffeePricingEnv(gym.Env):
    def __init__(self):
        super(CoffeePricingEnv, self).__init__()
    
        # store open from 8AM - 6PM, and since increments of 30 mins
        # we have 20 intervals
        self.num_intervals = 20
        # proper formatting of the time bins
        self.time_bins = [f"{8 + i//2}:{'00' if i%2==0 else '30'}" for i in range(self.num_intervals)]

        # dictionary of the rewards for all the different intervals
        self.interval_rewards = defaultdict(list)

        # one hot encoded buckets for splitting the time of day **
        # prev-day price and revenue
        self.extra_feats   = 2 

        self.num_coarse_buckets   = 4
        self.interval2bucket = np.repeat(
            np.arange(self.num_coarse_buckets),
            self.num_intervals // self.num_coarse_buckets + 1
        )[:self.num_intervals]

        self.obs_len = 8 + self.num_coarse_buckets + self.extra_feats + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.obs_len,), dtype=np.float32)


        # represents $3.00 to $8.00 in $0.10 steps
        self.action_space = spaces.Discrete(51)  

        # helps keep track of previous day prices, we have a buffer array
        self.prev_day_prices   = np.zeros(self.num_intervals, dtype=np.float32)
        self.prev_day_revenue  = np.zeros(self.num_intervals, dtype=np.float32)
        self.last_day_prices   = np.zeros(self.num_intervals, dtype=np.float32)
        self.last_day_revenue  = np.zeros(self.num_intervals, dtype=np.float32)

        # based on the data we collected - 6000 students, 50% drink coffee, 
        # 30% of those 50% drink at coffee club on prospect
        self.student_population = 6000
        self.coffee_drinkers = int(0.5 * self.student_population)
        self.CC_drinkers = int(0.3 * self.coffee_drinkers)
        self.price_memory_days = 3
        self.recent_avg_price = 500.0  # start at neutral $5.00 price
        self.recent_prices = [500.0] * self.price_memory_days
        self.high_price_days = 0  # how many recent days had high average price
        self.max_customers = self.CC_drinkers  # initial max customer base

        self.reset()


    def _get_obs(self):
        # gets numbers between 0 and 1
        scalars = np.array([
            self.current_interval / self.num_intervals,
            self.last_purchases / self.my_customers,
            self.previous_price / 800.0,
            float(self.is_friday_flag),
            float(self.is_raining),
            self.recent_avg_price / 800.0,
            self.high_price_days / 10.0,
            self.max_customers / self.CC_drinkers
        ], dtype=np.float32)


        # one-hot coarse bucket
        bucket_onehot = np.zeros(self.num_coarse_buckets, dtype=np.float32)
        b = self.interval2bucket[min(self.current_interval, self.num_intervals - 1)]
        bucket_onehot[b] = 1.0

        # gets the extra features of the previous price/revenue at same time slot in previous day
        i = min(self.current_interval, self.num_intervals - 1)
        extras = np.array([
            self.prev_day_prices[i] / 800.0,
            self.prev_day_revenue[i] / 20.0
        ], dtype=np.float32)

        # smooths the intervals
        phi = 2 * np.pi * self.current_interval / self.num_intervals
        time_feat = np.array([np.sin(phi), np.cos(phi)], dtype=np.float32)

        obs = np.concatenate((scalars, bucket_onehot, time_feat, extras))
        return obs
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 20% chance of friday
        self.day_type="friday" if random.random() < 0.2 else "weekday"

        self.is_friday_flag = 1 if self.day_type == "friday" else 0
        
        # each interval has slightly different elasticity
        self.hour_elasticity = np.random.uniform(0.6, 1.2, size=self.num_intervals)

        if hasattr(self, "last_day_prices"):
            self.prev_day_prices[:]  = self.last_day_prices
            self.prev_day_revenue[:] = self.last_day_revenue

       
        # clear today’s trackers
        self.last_day_prices[:]   = 0.0
        self.last_day_revenue[:]  = 0.0

        self.total_revenue = 0.0
        self.current_interval = 0
        self.last_purchases = 0

        # baseline price, so makes sense to start with this

        self.previous_price = 500
        self.price_counter = defaultdict(int)
        self.pricing_history = []
        self.purchase_history = []

        self.recent_prices.pop(0)
        self.recent_prices.append(np.mean(self.last_day_prices))
        self.recent_avg_price = np.mean(self.recent_prices)

        # track number of consecutive high-price days
        if self.recent_avg_price > 650:  # above $6.50 average
            self.high_price_days += 1
        else:
            self.high_price_days = max(0, self.high_price_days - 1)

        # apply churn: shrink max customer base over time
        churn_rate = 0.02  # lose 2% of base per high-price day
        churned_fraction = 1 - churn_rate * self.high_price_days
        churned_fraction = max(0.6, churned_fraction)  # don't drop below 60% of original base
        self.max_customers = int(self.CC_drinkers * churned_fraction)


        # every day has 20% chance of rain
        self.is_raining = np.random.rand() < 0.2

        # makes customer base adjustments based on survey to coffee drinkers
        self.my_customers = self.max_customers
        if self.is_raining:
            self.my_customers = int(0.8 * self.my_customers)
    
        if self.is_friday_flag:
            self.my_customers = int(0.8 * self.my_customers)
            # peaks more sensitive, different peaks for friday vs others
            self.hour_elasticity[[2,3,4,5]] += 0.5
        else:
            self.hour_elasticity[[0, 1, 2, 3, 14, 15]] += 0.5

        # gets hourly distribution of customers
        self.hourly_distribution = self._get_hourly_distribution()
        self.revenue_history = []

        return self._get_obs(), {} 


    def _get_hourly_distribution(self):
      
      # 20 time intervals (0 to 19), make random noise so every interval has some customer activity
      base = np.random.uniform(0.5, 1.5, size=self.num_intervals) 

        # customer peaks based on survey
      if self.day_type == "friday": 
          base[2] += 6  # 9 - 9:30
          base[3] += 6  # 9:30 - 10
          base[4] += 6  # 10 - 10:30
          base[5] += 6  # 10:30 - 11
      else:
          # weekday peaks between 8–10 am and 3–4 pm
          base[0] += 3  # 8 - 8:30
          base[1] += 3  # etc
          base[2] += 6  
          base[3] += 6  
          base[14] += 6 
          base[15] += 6 

      # normalizing to form probability distribution
      prob_dist = base / base.sum()

      return prob_dist
    
    def get_baseline_clone(self, fixed_price_cents):
        baseline_env = copy.deepcopy(self)

        # 2) reset only the trajectory‐specific bits:
        baseline_env.current_interval = 0
        baseline_env.purchases_so_far = 0
        baseline_env.total_revenue    = 0.0
        baseline_env.previous_price   = fixed_price_cents
        baseline_env.price_counter    = defaultdict(int)
        baseline_env.pricing_history  = []
        baseline_env.purchase_history = []
        baseline_env.revenue_history  = []

        return baseline_env


    
    # changes how many customers will buy at a certain time interval based on price
    def _price_sensitivity_multiplier(self, price):
        return 2.0 / (1 + np.exp(self.hour_elasticity[self.current_interval] * (price - 6.50)))


    def step(self, action):

        # references the global variable step
        global GLOBAL_STEP
        GLOBAL_STEP += 1
        # 0 → 1 over first 50 k env steps
        ramp = min(GLOBAL_STEP / 50_000, 1.0)   

        # sets the price
        action = int(action)
        price_dollars = (300 + 10 * action) / 100.0

        # tracking how often each price is chosen and rewards exploration slightly
        key = int(300 + 10 * action)        
        self.price_counter[key] += 1
        total_actions = sum(self.price_counter.values())
        freq = self.price_counter[key] / total_actions
        entropy_bonus = -np.log(freq + 1e-8)

        # base demand from hourly distribution
        base_volume = self.my_customers * self.hourly_distribution[self.current_interval]
        memory_penalty = max(0, (self.recent_avg_price - 600) / 100.0)
        base_volume *= (1 - 0.1 * memory_penalty)

        # penalizes or rewards customer count depending on price
        multiplier = self._price_sensitivity_multiplier(price_dollars)
        base_volume *= multiplier

        # sample num purchases from poisson distribution
        num_purchases = int(np.random.poisson(base_volume))
        revenue_dollars = price_dollars * num_purchases        

        # the reward is the reveue from this time increment
        money_reward = revenue_dollars

        # gets this info for observations
        self.last_day_prices[self.current_interval] = 300 + 10 * action 
        self.last_day_revenue[self.current_interval] = revenue_dollars


        # logging for later plots
        self.pricing_history.append((self.current_interval, price_dollars))
        self.purchase_history.append((self.current_interval, num_purchases))
        self.revenue_history.append((self.current_interval, revenue_dollars))

        # look back 4 intervals, punish significant variance as the steps progress (ramp increases with GLOBAL_STEP)
        recent_prices = [p for _, p in self.pricing_history[-4:]] + [price_dollars]
        variance_pen  = ramp * np.var(recent_prices) / 1e4 
        reward = money_reward - variance_pen + 0.05 * entropy_bonus

        # bookkeeping for observations or logging 
        self.last_purchases = num_purchases
        self.total_revenue    += revenue_dollars
        self.previous_price    = 300 + 10 * action   # real cents
        self.current_interval += 1

        # checks whether done
        done = self.current_interval == self.num_intervals


        return self._get_obs(), reward, done, False, {"total_revenue": self.total_revenue}

# makes a coffee environment, don't need but easy to keep for simplicity
def sample_env():
    return CoffeePricingEnv()

# plots price change and demand change over the course of one day
def plot_price_and_demand(env):

    # never have been reached but prevents unexpected exceptions
    if not hasattr(env, "pricing_history") or not env.pricing_history:
        print("No pricing history found in the environment.")
        return
    if not hasattr(env, "purchase_history") or not env.purchase_history:
        print("No purchase history found in the environment.")
        return

    # gets what we're plotting
    intervals = [i for i, _ in env.pricing_history]
    prices = [p for _, p in env.pricing_history]
    purchases = [n for _, n in env.purchase_history]
    labels = env.time_bins

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel("Time Interval")
    ax1.set_ylabel("Price ($)", color=color)
    ax1.plot(intervals, prices, color=color, label="Price", marker="o")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(ticks=range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylim(3, 8)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel("Number of Purchases", color=color)
    ax2.plot(intervals, purchases, color=color, label="Purchases", marker="x")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 300)

    fig.tight_layout()
    plt.title("Price and Customer Activity Over the Day")
    plt.grid(True)
    plt.show()

# takes in a policy and the environments we're testing on to get the average reward over the environments
def evaluate_policy(policy_fn, seed_envs, seed_obs):
        revenues = []
        for e0, o0 in zip(seed_envs, seed_obs):
            env = copy.deepcopy(e0)
            obs = o0.copy()
            done = False
            while not done:
                action = policy_fn(env, obs)
                obs, _, done, _, _ = env.step(action)
            revenues.append(env.total_revenue)
        avg_rev = np.mean(revenues)
        return revenues, avg_rev


if __name__ == "__main__":
    # makes the environment and the evaluation environment for the model
    env = make_vec_env(sample_env, n_envs=1)
    eval_env_instance = make_vec_env(sample_env, n_envs=1)
    eval_callback = EvalRolloutCallback(eval_env_instance, eval_freq=5000)
    callback = RewardTrackerCallback()

    # got these optial hyperparameters by using optuna
    model = PPO(
    "MlpPolicy", env,
    learning_rate=0.0009879484691513014,
    n_steps=512,
    batch_size=256,
    ent_coef=0.00611244359108216, 
    vf_coef=0.11958835015472258,
    gamma=0.9771365508801672,
    gae_lambda=0.8136338839538674,
    policy_kwargs=dict(net_arch=[512,256, 128]),
    verbose=1,
    device="cpu"           
)
    # train PPO 
    model.learn(total_timesteps=2000000, callback=[callback, eval_callback])
    
    # makes 10 consistent environments so we can test performance across prices and dynamic agent with same environments
    seed_envs = []
    seed_obs  = []
    seed_envs, seed_obs = [], []
    for _ in range(10):
        e = sample_env()
        o = e._get_obs()
        seed_envs.append(e)
        seed_obs.append(o)

    # define the two policy fns, fixed and dynamic
    def ppo_policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    def fixed_policy(idx):
        return lambda env, obs: idx

    # evaluate PPO
    ppo_revenues, ppo_avg = evaluate_policy(ppo_policy, seed_envs, seed_obs)
    print(f"PPO avg revenue over seeds: ${ppo_avg:.2f}")

    # evaluate all fixed prices
    fixed_prices   = list(range(300, 801, 10))
    fixed_avg_revs = []
    for price_cents in fixed_prices:
        idx = (price_cents - 300)//10
        _, avg = evaluate_policy(fixed_policy(idx), seed_envs, seed_obs)
        fixed_avg_revs.append(avg)

    # plot the average revenue of all fixed prices compared to the dynamic pricing
    plt.figure(figsize=(10,6))
    plt.plot([p/100 for p in fixed_prices], fixed_avg_revs,
            label="Avg Fixed Price Revenue", marker='o')
    plt.axhline(ppo_avg, color='red', linestyle='--',
                label="Avg Dynamic Pricing (PPO)")
    plt.xlabel("Price ($)")
    plt.ylabel("Avg Total Revenue ($)")
    plt.title("Fixed vs Dynamic Pricing (same 10 seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plots reward history through training compared to $5.00
    plot_reward_history(callback.ppo_rewards, baseline_rewards=callback.baseline_rewards)

    # makes one last environment to plot the price and demand changes over the day
    ppo_env = CoffeePricingEnv() 
    obs = ppo_env._get_obs()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = ppo_env.step(action)
        total_reward += reward

    plot_price_and_demand(ppo_env)
