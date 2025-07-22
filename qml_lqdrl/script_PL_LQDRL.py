import jax
import pennylane as qml
from pennylane import numpy as np
from pennylane import draw
from pennylane import grad
from pennylane.optimize import AdamOptimizer
from pennylane.fourier import circuit_spectrum
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
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
    plt.savefig(f'eg_plots/uav_trajectory_{ep}_timestep_{t}.png')

from uav_lqdrl_env import UAV_LQDRL_Environment
from quantum_models import QuantumActor, QuantumCritic
from replay_buffer import ReplayBuffer
from pennylane.optimize import AdamOptimizer
from prioritised_experience_replay import SumTree, Memory

env = UAV_LQDRL_Environment()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = QuantumActor(n_qubits=state_dim, m_layers=3)
critic = QuantumCritic(n_qubits=state_dim + action_dim, m_layers=3)

buffer = ReplayBuffer()
actor_opt = AdamOptimizer(stepsize=0.01)
critic_opt = AdamOptimizer(stepsize=0.01)

episodes = 2
batch_size = 1
gamma = 0.99

time_step = 1

time_arr = []

tot_reward_arr = []
step_rewards_arr = []
actor_losses = []
critic_losses = []

uav_pos_arr = []

for ep in range(episodes):
    ep_start_time = time.time()
    time_var = 0
    i = 0
    state, _ = env.reset()
    done = False
    total_reward = 0
    ep_uav_trajectory = []
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
        print("Distance of UAV from GU Centroid: ", dist_to_centroid, "m")

        step_start_time = time.time()
        state_tensor = np.array(state, requires_grad=False)
        action = actor(state_tensor)
        action = np.clip(np.array(action), -1, 1)

        next_state, reward, done, _, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        step_rewards_arr.append(reward)

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

            def critic_loss(theta):
                critic.update_params(theta)
                loss = 0
                for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                    sa = np.concatenate([s, a])
                    q_val = critic(sa)
                    q_val = critic.decode_op(q_val)
                    print("Critic Q-Value: ", q_val)

                    na = actor(ns)
                    nsa = np.concatenate([ns, na])
                    q_val_next = critic(nsa)
                    q_val_next = critic.decode_op(q_val_next)

                    target = r + gamma * q_val_next * (1 - d)
                    print("Critic Target: ", target)
                    loss += (q_val - target) ** 2
                return loss / batch_size

            critic.theta, critic_loss_val = critic_opt.step_and_cost(critic_loss, critic.theta)
            critic_losses.append(critic_loss_val)

            def actor_loss(theta):
                actor.update_params(theta)
                loss = 0
                for s in states:
                    a = actor(s)
                    sa = np.concatenate([s, a])
                    q_val = critic(sa)
                    q_val = critic.decode_op(q_val)
                    print("Actor Q-Value: ", q_val)
                    loss -= q_val
                return loss / batch_size

            actor.theta, actor_loss_val = actor_opt.step_and_cost(actor_loss, actor.theta)
            actor_losses.append(actor_loss_val)
            time_var += time_step
        time_arr.append(time_var)
        #plot_uav_position(env)
        plot_uav_trajectory(env, ep_uav_trajectory, ep, i)
        step_end_time = time.time()
        step_time = step_start_time - step_end_time 
        print(f"Time taken for step {i} to execute: ", step_time, " seconds")
        i += 1
        break_var += 1
        if break_var >= 10:
            break

    ep_end_time = time.time()
    ep_time = ep_start_time - ep_end_time
    print("Time taken for episode to execute: ", ep_time, " seconds")
    plot_uav_trajectory(env, ep_uav_trajectory, ep, i)
    tot_reward_arr.append(total_reward)
    print(f"Episode {ep} | Total reward: {total_reward:.2f}")
    uav_pos_arr.append(ep_uav_trajectory)

print("All good so far")
