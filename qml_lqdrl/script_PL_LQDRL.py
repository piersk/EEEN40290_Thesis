#script_PL_LQDRL.py
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import numpy as np
from pennylane import draw
from pennylane import grad
#from pennylane.optimize import AdamOptimizer
from pennylane.fourier import circuit_spectrum
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
#import optax
from collections import deque
import random
import time

# Function to plot the latest UAV position as a point as well as it's previous course as a line
def plot_uav_trajectory(env, uav_trajectory, ep, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for gu in env.legit_users:
        gu_pos = gu.position
        ax.scatter(gu_pos[0], gu_pos[1], gu_pos[2], label=f'GU {gu.id}', color="green")
    centroid = np.mean([gu.position for gu in env.legit_users], axis=0)

    uav_positions = np.array(env.uavs[0].history)
    ax.plot(uav_positions[:,0], uav_positions[:,1], uav_positions[:,2], label="UAV Path", color="blue")
        
    for uav in env.uavs:
        uav_position = uav.position
        ax.scatter(uav_position[0], uav_position[1], uav_position[2], label="UAV Positions", color="cyan")
    ax.scatter(*centroid, label="GU Centroid", color="red", marker="X", s=100)
    plt.legend()
    plt.savefig(f'eg_plots/test24/uav_trajectory_{ep}_timestep_{t}.png')
    plt.close()

# Importing modules required for experiments
from uav_lqdrl_env import UAV_LQDRL_Environment
from quantum_models import QuantumActor, QuantumCritic
from replay_buffer import ReplayBuffer
from pennylane.optimize import AdamOptimizer
from prioritised_experience_replay import SumTree, Memory

env = UAV_LQDRL_Environment()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

#actor = QuantumActor(n_qubits=state_dim, m_layers=3)
#critic = QuantumCritic(n_qubits=state_dim + action_dim, m_layers=3)
actor = QuantumActor(n_qubits=state_dim, m_layers=3)
critic = QuantumCritic(n_qubits=state_dim + action_dim, m_layers=3)

buffer = ReplayBuffer()
#actor_opt = AdamOptimizer(stepsize=0.01)
#critic_opt = AdamOptimizer(stepsize=0.01)
import optax

actor_opt = optax.adam(learning_rate=0.01)
critic_opt = optax.adam(learning_rate=0.01)

actor_opt_state = actor_opt.init(actor.theta)
critic_opt_state = critic_opt.init(critic.theta)

episodes = 20
batch_size = 30
#episodes = 2
#batch_size = 1
gamma = 0.99
#max_act_scale = 1e14
max_act_scale = 1e1

time_step = 1

time_arr = []

tot_reward_arr = []
step_rewards_arr = []
actor_losses = []
critic_losses = []
ep_distances_to_centroid = []

uav_pos_arr = []

total_runtime_start = time.time()

for ep in range(episodes):
    ep_start_time = time.time()
    time_var = 0
    i = 0
    state, _ = env.reset()
    done = False
    total_reward = 0
    ep_uav_trajectory = []
    dist_to_centroid_arr = []
    break_var = 0

    while not done:
        uav_pos = env.get_uav_position()
        ep_uav_trajectory.append(uav_pos)

        uav_energy = env.get_remaining_energy()

        uav_energy_perc = uav_energy / env.E_MAX

        print("Remaining UAV Energy: ", uav_energy, " J")

        print("Percentage of Remaining UAV Energy: ", uav_energy_perc * 100, "%")

        print("UAV Position Co-ordinates: ", uav_pos)

        gu_centroid = np.mean([gu.position for gu in env.legit_users], axis=0)
        print("GU Centroid Co-ordinates: ", gu_centroid)

        dist_to_centroid = np.linalg.norm(uav_pos - gu_centroid)
        dist_to_centroid_arr.append(dist_to_centroid)
        print("Distance of UAV from GU Centroid: ", dist_to_centroid, "m")

        step_start_time = time.time()
        #state_tensor = np.array(state, requires_grad=False)
        state_tensor = jnp.array(state)
        action = actor(state_tensor)
        print("Action: ", action)
        action = jnp.tanh(jnp.array(action)) * max_act_scale
        print("Action Scaled along Hyperbolic Tangent: ", action)
        action = np.clip(np.array(action), -1, 1)
        print("Clipped Action: ", action)

        next_state, reward, done, _, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        step_rewards_arr.append(reward)

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

            # Replace the critic_loss and actor_loss functions with:

            def critic_loss(theta):
                q_vals = []
                targets = []
                for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                    sa = jnp.concatenate([jnp.array(s), jnp.array(a)])
                    q_val = critic.qnode(sa, theta)
                    q_val = critic.decode_op(q_val)

                    na = actor(jnp.array(ns))
                    nsa = jnp.concatenate([jnp.array(ns), jnp.array(na)])
                    q_val_next = critic.qnode(nsa, theta)
                    q_val_next = critic.decode_op(q_val_next)

                    target = r + gamma * q_val_next * (1 - d)
                    q_vals.append(q_val)
                    targets.append(target)
                q_vals = jnp.array(q_vals)
                targets = jnp.array(targets)
                return jnp.mean((q_vals - targets) ** 2)

            def actor_loss(theta):
                q_vals = []
                for s in states:
                    a = actor(jnp.array(s), theta)
                    sa = jnp.concatenate([jnp.array(s), jnp.array(a)])
                    q_val = critic.qnode(sa, critic.theta)  # Use critic's current theta
                    q_val = critic.decode_op(q_val)
                    q_vals.append(q_val)
                return -jnp.mean(jnp.array(q_vals))

            actor_losses.append(actor_loss(actor.theta))
            critic_losses.append(critic_loss(critic.theta))

            #actor.theta, actor_loss_val = actor_opt.step_and_cost(actor_loss, actor.theta)
            #actor_losses.append(actor_loss_val)
            #g = grad(actor_loss)(actor.theta)
            #print("Computed Gradient: ", g)
            time_var += time_step
        time_arr.append(time_var)
        if i % 10 == 0 or done:
            plot_uav_trajectory(env, ep_uav_trajectory, ep, i)
        step_end_time = time.time()
        step_time = step_start_time - step_end_time 
        print(f"Time taken for step {i} to execute: ", abs(step_time), " seconds")
        i += 1
        # Break out of episode early (for debugging purposes)
        #break_var += 1
        #if break_var >= 50:
        #    break

    ep_end_time = time.time()
    ep_time = ep_start_time - ep_end_time
    print("Time taken for episode to execute: ", abs(ep_time), " seconds")
    plot_uav_trajectory(env, ep_uav_trajectory, ep, i)
    tot_reward_arr.append(total_reward)
    print(f"Episode {ep} | Total reward: {total_reward:.10f}")
    uav_pos_arr.append(ep_uav_trajectory)
    ep_distances_to_centroid.append(dist_to_centroid_arr)

# Plot rewards and losses
#plt.plot(total_rewards)
plt.plot(tot_reward_arr)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig("eg_plots/test24/rewards_over_episodes.png")
plt.close()

plt.plot(actor_losses, label="Actor Loss")
plt.plot(critic_losses, label="Critic Loss")
plt.legend()
plt.title("Actor and Critic Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.savefig("eg_plots/test24/losses.png")
plt.close()

fig, ax = plt.subplots(5, 4, figsize=(20, 16))
idx = 0
for i in range(5):
    for j in range(4):
        ax[i, j].plot(ep_distances_to_centroid[idx], label=f"Episode {idx} Distance of UAV-BS to Centroid")
        ax[i, j].set_ylabel("Distance") 
        ax[i, j].set_xlabel("Time")
        ax[i, j].set_title(f"Episode {idx}")
        idx += 1
fig.suptitle("UAV-BS Distances to GU Centroid Across Episodes")
plt.tight_layout()
plt.savefig("eg_plots/test24/distances_to_centroid.png")
plt.close()

print("All good so far")
total_runtime_end = time.time()
total_runtime = abs(total_runtime_end - total_runtime_start)
print("Total Time Taken for Experiment to Run: ", total_runtime)
