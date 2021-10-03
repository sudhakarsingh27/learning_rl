## Code adapted from: https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f


import numpy as np
import gym
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np

from jax.scipy.special import logsumexp


# Init the keys
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [2, 3]
param_scale = 0.1
step_size = 1
num_epochs = 10
batch_size = 1
n_targets = 3
params = init_network_params(layer_sizes, random.PRNGKey(0))

def relu(x):
    return jnp.maximum(0, x)

def q_value(params, x):
    # per-example predictions
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
  
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


def loss(params, next_state, reward, cur_state_rewards, action, discount, learning):
    future_rewards = q_value(params, jnp.array([next_state[0], next_state[1]]))
    return learning*(reward + discount*jnp.max(future_rewards) - 
                                 cur_state_rewards[action])

def update(params, next_state, reward, cur_state_rewards, action, discount, learning):
    grads = grad(loss)(params, next_state, reward, cur_state_rewards, action, discount, learning)
    return [(w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)]

def terminal_state_loss(reward, action, cur_state_rewards):
    return reward - cur_state_rewards[action]
    
def update_terminal_state(params, reward, action, cur_state_rewards):
    grads = grad(terminal_state_loss)(reward, action, cur_state_rewards)
    return [(w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)]

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes, params):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # Initialize Q table
    # Q = np.random.uniform(low = -1, high = 1, 
    #                       size = (num_states[0], num_states[1], 
    #                               env.action_space.n))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        cur_state = env.reset()
        
        # Discretize state
        # state_adj = (state - env.observation_space.low)*np.array([10, 100])
        # state_adj = np.round(state_adj, 0).astype(int)
    
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 20):
                env.render()
            # print("new state")

            # In any case, get the rewards for current state
            cur_state_rewards = q_value(params, jnp.array([cur_state[0], cur_state[1]]))

            print(cur_state_rewards)
            
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                print("exploiting")
                action = int(np.argmax(cur_state_rewards)) 
            else:
                print("exploring ...")
                action = np.random.randint(0, env.action_space.n)

            # print(action, type(action))

            # Get next state and reward
            next_state, reward, done, info = env.step(action) 

            #Allow for terminal states
            if done and next_state[0] >= 0.5:
                
                ## UPDATE NN based on q_value
                params = update_terminal_state(params, reward, action, cur_state_rewards)

            # Adjust Q value for current state
            else:
                params = update(params, next_state, reward, cur_state_rewards, action, discount, learning)

            # Update variables
            tot_reward += reward
            cur_state = next_state

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 100 == 0:    
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
            
    env.close()
    
    return ave_reward_list

# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 5000, params)

# Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')     
plt.close()  
