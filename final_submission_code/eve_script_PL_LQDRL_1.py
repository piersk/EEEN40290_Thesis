#script_PL_LQDRL.py
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import numpy as np
from pennylane import draw
from pennylane import grad
from pennylane.fourier import circuit_spectrum
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
from collections import deque
import random
import time
import math

# Function to plot the latest UAV position as a point as well as it's previous course as a line
def plot_uav_trajectory(env, uav_trajectory, layer, ep, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for gu in env.legit_users:
        gu_pos = gu.position
        ax.scatter(gu_pos[0], gu_pos[1], gu_pos[2], label=f'GU {gu.id}', color="green")
    centroid = np.mean([gu.position for gu in env.legit_users], axis=0)

    for eve in env.eves:
        eve_pos = eve.position
        ax.scatter(eve_pos[0], eve_pos[1], eve_pos[2], label=f'Eve {eve.id}', color="magenta")

    uav_positions = np.array(env.uavs[0].history)
    ax.plot(uav_positions[:,0], uav_positions[:,1], uav_positions[:,2], label="UAV Path", color="blue")
        
    for uav in env.uavs:
        uav_position = uav.position
        ax.scatter(uav_position[0], uav_position[1], uav_position[2], label="UAV Positions", color="cyan")
    ax.scatter(*centroid, label="GU Centroid", color="red", marker="X", s=100)
    plt.legend()
    plt.savefig(f'eve_outputs/plots_eve_outputs/test3/{layer+1}_layers_uav_trajectory_{ep}_timestep_{t}.png')
    plt.close()

def gradient_norm(grad):
    return jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in grad]))

# Importing modules required for experiments
from eve_uav_lqdrl_env import UAV_LQDRL_Environment
from quantum_models import QuantumActor, QuantumCritic
from replay_buffer import ReplayBuffer
from pennylane.optimize import AdamOptimizer
from prioritised_experience_replay import SumTree, Memory

overall_start_time = time.time()

m_layers = 1
for m in range(m_layers):
    print(f"============ Experiment with {m+1} Layers in Ansatz ============")
    env = UAV_LQDRL_Environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = QuantumActor(n_qubits=state_dim, m_layers=m+1)
    critic = QuantumCritic(n_qubits=state_dim + action_dim, m_layers=m+1)

    capacity = 100000
    buffer = Memory(capacity)

    import optax

    actor_opt = optax.adam(learning_rate=0.01)
    critic_opt = optax.adam(learning_rate=0.01)

    actor_opt_state = actor_opt.init(actor.theta)
    critic_opt_state = critic_opt.init(critic.theta)

    episodes = 30
    batch_size = 30
    gamma = 0.99
    max_act_scale = 1e15
    #max_act_scale = 1

    time_step = 1
    diff = 0

    time_arr = []

    tot_reward_arr = []
    rewards_across_eps_arr = []
    actor_losses = []
    critic_losses = []
    ep_distances_to_centroid = []
    ep_sum_rate_arr = []
    ep_energy_eff_arr = []
    ep_timesteps_arr = []
    ep_remaining_energy_arr = []
    ep_energy_cons_arr = []
    ep_secrecy_rates_arr = []

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
        step_rewards_arr = []
        step_sum_rate_arr = []
        step_energy_eff_arr = []
        step_remaining_energy_arr = []
        step_energy_cons_arr = []
        step_secrecy_rates_arr = []
        break_var = 0

        while not done:
            step_start_time = time.time()
            uav_pos = env.get_uav_position()
            ep_uav_trajectory.append(uav_pos)

            uav_energy = env.get_remaining_energy()
            energy_cons = env.E_MAX - uav_energy 
            step_energy_cons_arr.append(energy_cons)

            uav_energy_perc = uav_energy / env.E_MAX

            print("Remaining UAV Energy: ", uav_energy, " J")

            print("Percentage of Remaining UAV Energy: ", uav_energy_perc * 100, "%")

            print("UAV Position Co-ordinates: ", uav_pos)

            gu_centroid = np.mean([gu.position for gu in env.legit_users], axis=0)
            gu_centroid[2] += 10
            print("GU Centroid Co-ordinates: ", gu_centroid)

            dist_to_centroid = np.linalg.norm(uav_pos - gu_centroid)
            dist_to_centroid_arr.append(dist_to_centroid)
            print("Distance of UAV from GU Centroid: ", dist_to_centroid, "m")

            energy_efficiency = env.get_energy_efficiency()
            print(f"Energy Efficiency for step {i}: ", energy_efficiency)

            step_remaining_energy_arr.append(uav_energy)
            step_energy_eff_arr.append(energy_efficiency)

            #state_tensor = np.array(state, requires_grad=False)
            state_tensor = jnp.array(state)
            action = actor(state_tensor)
            print("Action: ", action)
            action = jnp.tanh(jnp.array(action)) * max_act_scale
            print("Action Scaled along Hyperbolic Tangent: ", action)
            action = np.clip(np.array(action), -1, 1)
            print("Clipped Action: ", action)

            next_state, reward, done, _, _ = env.step(action)
            #buffer.push(state, action, reward, next_state, done)
            buffer.store([state, action, reward, next_state, done])
            total_reward += reward
            state = next_state

            step_rewards_arr.append(reward)

            sum_rates = env.get_sum_rates()
            step_sum_rate_arr.append(sum_rates)

            secrecy_rates = [0.0, 0.0, 0.0, 0.0]
            secrecy_rates = env.get_secrecy_rates()
            for eve_m in range(len(secrecy_rates)):
                sr = secrecy_rates[eve_m]
                if math.isnan(sr) is True:
                    secrecy_rates[eve_m] = 0.0
            secrecy_rates = [float(sr) if np.isfinite(sr) else 0.0 for sr in secrecy_rates]
            step_secrecy_rates_arr.append(secrecy_rates)
            print(f"Secrecy Rates for Step {i}: ", secrecy_rates)

            if len(buffer) >= batch_size:
                b_idx, batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

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
                    td_err = abs(q_vals - targets)
                    loss_val = jnp.mean((q_vals - targets) ** 2)
                    return loss_val, td_err

                def actor_loss(theta):
                    q_vals = []
                    for s in states:
                        a = actor(jnp.array(s), theta)
                        sa = jnp.concatenate([jnp.array(s), jnp.array(a)])
                        q_val = critic.qnode(sa, critic.theta)  # Use critic's current theta
                        q_val = critic.decode_op(q_val)
                        q_vals.append(q_val)
                    return -jnp.mean(jnp.array(q_vals))

                loss_val, td_errors = critic_loss(critic.theta)
                critic_losses.append(loss_val)
                actor_losses.append(actor_loss(actor.theta))
                buffer.batch_update(b_idx, np.array(td_errors))

                time_var += time_step
            time_arr.append(time_var)
            #if i % 10 == 0 or done:
                #plot_uav_trajectory(env, ep_uav_trajectory, m, ep, i)
            step_end_time = time.time()
            step_time = step_start_time - step_end_time 
            print(f"Time taken for step {i} to execute: ", abs(step_time), " seconds")
            i += 1
            # Break out of episode early (for debugging purposes)
            #if i == 10:
            #    break
            if dist_to_centroid_arr[i-2] is not None:
                diff = dist_to_centroid_arr[i-2] - dist_to_centroid_arr[i-1]
                if (diff <= 0.1):
                    break_var += 1
            if break_var >= 250:
                break

        ep_end_time = time.time()
        ep_time = ep_start_time - ep_end_time
        print("Time taken for episode to execute: ", abs(ep_time), " seconds")
        plot_uav_trajectory(env, ep_uav_trajectory, m, ep, i)
        print(f"Episode {ep} | Total reward: {total_reward:.10f}")
        tot_reward_arr.append(total_reward)
        ep_sum_rate_arr.append(step_sum_rate_arr)
        uav_pos_arr.append(ep_uav_trajectory)
        ep_energy_eff_arr.append(step_energy_eff_arr)
        ep_distances_to_centroid.append(dist_to_centroid_arr)
        rewards_across_eps_arr.append(step_rewards_arr)
        ep_timesteps_arr.append(i)
        ep_remaining_energy_arr.append(step_remaining_energy_arr)
        ep_energy_cons_arr.append(step_energy_cons_arr)
        ep_secrecy_rates_arr.append(step_secrecy_rates_arr)

    # Plot rewards and losses
    #plt.plot(total_rewards)
    plt.plot(tot_reward_arr)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_rewards_over_episodes.png")
    plt.close()

    #plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.legend()
    plt.title("Actor and Critic Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_layers_losses.png")
    plt.close()

    fig, ax = plt.subplots(5, 6, figsize=(20, 16))
    colour_codes = ['b', 'orange', 'g', 'r']
    gu_labels = [f"GU {i}" for i in range(env.num_legit_users)]
    idx = 0
    for i in range(5):
        for j in range(6):
            for gu_id in range(len(env.legit_users)):
                gu_rates = [sr[gu_id] for sr in ep_sum_rate_arr[idx]]
                ax[i, j].plot(gu_rates, color=colour_codes[gu_id])
                ax[i, j].set_xlabel("Timesteps")
                ax[i, j].set_ylabel("Sum Rates (bps)")
            ax[i, j].set_title(f"Episode {idx} Sum Rates")
            idx += 1
    fig.legend(gu_labels, loc='upper right', ncol=len(env.legit_users), fontsize=12)
    fig.suptitle(f"Sum Rates for All Legitimate GUs Across Episodes with {m+1} Layers")
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_layers_sum_rates.png")
    plt.close()

    fig, ax = plt.subplots(5, 6, figsize=(20, 16))
    #fig, ax = plt.subplots(5, 2, figsize=(20, 16))
    colour_codes = ['b', 'orange', 'g', 'r']
    gu_labels = [f"GU {i}" for i in range(env.num_legit_users)]
    idx = 0
    for i in range(5):
        for j in range(6):
            for gu_id in range(len(env.legit_users)):
                gu_rates = [sr[gu_id] for sr in ep_secrecy_rates_arr[idx]]
                ax[i, j].plot(gu_rates, color=colour_codes[gu_id])
                ax[i, j].set_xlabel("Timesteps")
                ax[i, j].set_ylabel("Secrecy Rates (bps)")
            ax[i, j].set_title(f"Episode {idx} Secrecy Rates")
            idx += 1
    fig.legend(gu_labels, loc='upper right', ncol=len(env.legit_users), fontsize=12)
    fig.suptitle(f"Secrecy Rates for All Legitimate GUs Across Episodes with {m+1} Layers")
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_layers_secrecy_rates.png")
    plt.close()

    fig, ax = plt.subplots(5, 6, figsize=(20, 16))
    idx = 0
    for i in range(5):
        for j in range(6):
            ax[i, j].plot(ep_distances_to_centroid[idx], label=f"Episode {idx} Distance of UAV-BS to Centroid")
            ax[i, j].set_ylabel("Distance (m)") 
            ax[i, j].set_xlabel("Time")
            ax[i, j].set_title(f"Episode {idx}")
            idx += 1
    fig.suptitle("UAV-BS Distances to GU Centroid Across Episodes")
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_layers_distances_to_centroid.png")
    plt.close()

    fig, ax = plt.subplots(5, 6, figsize=(20, 16))
    idx = 0
    for i in range(5):
        for j in range(6):
            ax[i, j].plot(rewards_across_eps_arr[idx], label=f"Episode {idx} Rewards")
            ax[i, j].set_ylabel("Reward") 
            ax[i, j].set_xlabel("Timestep")
            ax[i, j].set_title(f"Episode {idx}")
            idx += 1
    fig.suptitle("Allocated Reward Curves Across Episodes")
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_layers_episodewise_rewards.png")
    plt.close()

    fig, ax = plt.subplots(5, 6, figsize=(20, 16))  # One subplot per episode
    idx = 0
    for i in range(5):
        for j in range(6):
            rewards = rewards_across_eps_arr[idx]
            distances = ep_distances_to_centroid[idx]
            min_len = min(len(rewards), len(distances))  # In case lengths mismatch
            ax[i, j].plot(distances[:min_len], rewards[:min_len])
            ax[i, j].set_xlabel("Distance to Centroid")
            ax[i, j].set_ylabel("Reward")
            ax[i, j].set_title(f"Episode {i}: Reward vs Distance")
            idx += 1
    fig.suptitle("Reward vs Distance to Centroid")
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_layers_reward_vs_distance.png")
    plt.close()

    energy_eff_all = np.concatenate([np.array(ep) for ep in ep_energy_eff_arr])
    energy_cons_all = env.E_MAX - energy_eff_all
    sum_rates_all = np.concatenate([np.array(ep) for ep in ep_sum_rate_arr], axis=0)  # shape: (total_steps, num_GUs)

    def global_stats(data):
        return data.max(), data.min(), data.mean()

    energy_eff_stats = global_stats(energy_eff_all)
    energy_cons_stats = global_stats(energy_cons_all)

    sum_rate_stats_perGU = [global_stats(sum_rates_all[:, gu]) for gu in range(env.num_legit_users)]

    # ----------- Per-timestep Stats -----------
    min_len = min(len(ep) for ep in ep_energy_eff_arr)  # shortest episode length

    # Per-timestep arrays: shape (num_eps, min_len, ...)
    energy_eff_steps = np.array([ep[:min_len] for ep in ep_energy_eff_arr])
    energy_cons_steps = env.E_MAX - energy_eff_steps
    sum_rates_steps = np.array([ep[:min_len] for ep in ep_sum_rate_arr])  # shape: (num_eps, min_len, num_GUs)

    # Stats over episodes (axis=0)
    energy_eff_step_max = energy_eff_steps.max(axis=0)
    energy_eff_step_min = energy_eff_steps.min(axis=0)
    energy_eff_step_avg = energy_eff_steps.mean(axis=0)

    energy_cons_step_max = energy_cons_steps.max(axis=0)
    energy_cons_step_min = energy_cons_steps.min(axis=0)
    energy_cons_step_avg = energy_cons_steps.mean(axis=0)

    sum_rate_step_max = sum_rates_steps.max(axis=0)   # shape: (min_len, num_GUs)
    sum_rate_step_min = sum_rates_steps.min(axis=0)
    sum_rate_step_avg = sum_rates_steps.mean(axis=0)

    # ----------- Plot Global Stats -----------
    plt.figure()
    plt.bar(["Max", "Min", "Mean"], energy_eff_stats)
    plt.title(f"Global Energy Efficiency Stats ({m+1} Layers)")
    plt.ylabel("Energy Efficiency (bps/Hz/J)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_global_energy_eff.png")
    plt.close()

    plt.figure()
    plt.bar(["Max", "Min", "Mean"], energy_cons_stats)
    plt.title(f"Global Energy Consumption Stats ({m+1} Layers)")
    plt.ylabel("Energy Consumption (J)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_global_energy_cons.png")
    plt.close()

    plt.figure()
    for gu in range(env.num_legit_users):
        stats = sum_rate_stats_perGU[gu]
        plt.bar([f"Max GU{gu}", f"Min GU{gu}", f"Mean GU{gu}"], stats)
    plt.title(f"Global Sum Rate Stats per GU ({m+1} Layers)")
    plt.ylabel("Sum Rate (bps)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_global_sum_rates_perGU.png")
    plt.close()

    # ----------- Plot Per-timestep Stats -----------
    timesteps = range(min_len)

    plt.figure()
    plt.plot(timesteps, energy_eff_step_max, label="Max")
    plt.plot(timesteps, energy_eff_step_min, label="Min")
    plt.plot(timesteps, energy_eff_step_avg, label="Mean")
    plt.legend()
    plt.title(f"Per-timestep Energy Efficiency Stats ({m+1} Layers)")
    plt.xlabel("Timestep")
    plt.ylabel("Energy Efficiency (bps/Hz/J)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_timestep_energy_eff.png")
    plt.close()

    plt.figure()
    plt.plot(timesteps, energy_cons_step_max, label="Max")
    plt.plot(timesteps, energy_cons_step_min, label="Min")
    plt.plot(timesteps, energy_cons_step_avg, label="Mean")
    plt.legend()
    plt.title(f"Per-timestep Energy Consumption Stats ({m+1} Layers)")
    plt.xlabel("Timestep")
    plt.ylabel("Energy Consumption (J)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_timestep_energy_cons.png")
    plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(20, 16))
    idx = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].plot(timesteps, sum_rate_step_max[:, idx], label="Max")
            ax[i, j].plot(timesteps, sum_rate_step_min[:, idx], label="Min")
            ax[i, j].plot(timesteps, sum_rate_step_avg[:, idx], label="Mean")
            ax[i, j].set_title(f"Max, Min & Mean Sum Rates for GU {idx}")
            ax[i, j].set_xlabel("Timestep")
            ax[i, j].set_ylabel("Sum Rate (bps)")
            ax[i, j].legend()
            idx += 1
    fig.suptitle(f"Maximum, Minimum & Mean Sum Rates for All GUs")
    #fig.legend(gu_labels, loc='upper right', ncol=len(env.legit_users), fontsize=12)
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_timestep_sum_rate_all_GUs.png")
    plt.close()


    for gu in range(env.num_legit_users):
        plt.figure()
        plt.plot(timesteps, sum_rate_step_max[:, gu], label="Max")
        plt.plot(timesteps, sum_rate_step_min[:, gu], label="Min")
        plt.plot(timesteps, sum_rate_step_avg[:, gu], label="Mean")
        plt.legend()
        plt.title(f"Per-timestep Sum Rate Stats GU {gu} ({m+1} Layers)")
        plt.xlabel("Timestep")
        plt.ylabel("Sum Rate (bps)")
        plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_timestep_sum_rate_GU{gu}.png")
        plt.close()


    ep_count = len(ep_sum_rate_arr)
    ep_nums = np.arange(ep_count)


    secrecy_rates_steps = np.array([ep[:min_len] for ep in ep_secrecy_rates_arr])  # (num_eps, min_len, num_GUs)
    secrecy_rate_step_max = secrecy_rates_steps.max(axis=0)
    secrecy_rate_step_min = secrecy_rates_steps.min(axis=0)
    secrecy_rate_step_avg = secrecy_rates_steps.mean(axis=0)

    fig, ax = plt.subplots(2, 2, figsize=(20, 16))
    idx = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].plot(timesteps, secrecy_rate_step_max[:, idx], label="Max")
            ax[i, j].plot(timesteps, secrecy_rate_step_min[:, idx], label="Min")
            ax[i, j].plot(timesteps, secrecy_rate_step_avg[:, idx], label="Mean")
            ax[i, j].set_title(f"Max, Min & Mean Secrecy Rates for GU {idx}")
            ax[i, j].set_xlabel("Timestep")
            ax[i, j].set_ylabel("Secrecy Rate (bps)")
            ax[i, j].legend()
            idx += 1
    fig.suptitle(f"Maximum, Minimum & Mean Secrecy Rates for All GUs")
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_timestep_secrecy_rate_all_GUs.png")
    plt.close()

    # Individual GU plots
    for gu in range(env.num_legit_users):
        plt.figure()
        plt.plot(timesteps, secrecy_rate_step_max[:, gu], label="Max")
        plt.plot(timesteps, secrecy_rate_step_min[:, gu], label="Min")
        plt.plot(timesteps, secrecy_rate_step_avg[:, gu], label="Mean")
        plt.legend()
        plt.title(f"Per-timestep Secrecy Rate Stats GU {gu} ({m+1} Layers)")
        plt.xlabel("Timestep")
        plt.ylabel("Secrecy Rate (bps)")
        plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_timestep_secrecy_rate_GU{gu}.png")
        plt.close()

    # ----------- Per-episode Stats -----------
    # Compute max, min, mean for sum rates (shape: num_eps, num_GUs)
    sum_rate_max = np.zeros((ep_count, env.num_legit_users))
    sum_rate_min = np.zeros((ep_count, env.num_legit_users))
    sum_rate_mean = np.zeros((ep_count, env.num_legit_users))

    energy_eff_max = []
    energy_eff_min = []
    energy_eff_mean = []

    energy_cons_max = []
    energy_cons_min = []
    energy_cons_mean = []

    extrapolated_rewards = []

    for ep in range(ep_count):
        sum_rates = np.array(ep_sum_rate_arr[ep])  # shape: (timesteps, num_GUs)
        energy_cons = np.array(ep_energy_cons_arr[ep])
        energy_eff = np.array(ep_energy_eff_arr[ep])
        rewards = np.array(rewards_across_eps_arr[ep])

        # Sum rate stats per GU
        for gu in range(env.num_legit_users):
            sum_rate_max[ep, gu] = sum_rates[:, gu].max()
            sum_rate_min[ep, gu] = sum_rates[:, gu].min()
            sum_rate_mean[ep, gu] = sum_rates[:, gu].mean()

        # Energy stats
        energy_eff_max.append(energy_eff.max())
        energy_eff_min.append(energy_eff.min())
        energy_eff_mean.append(energy_eff.mean())

        energy_cons_max.append(energy_cons.max())
        energy_cons_min.append(energy_cons.min())
        energy_cons_mean.append(energy_cons.mean())

        # ----------- Extrapolated reward calculation -----------
        remaining_energy = env.get_remaining_energy_from_episode(ep) if hasattr(env, 'get_remaining_energy_from_episode') else env.get_remaining_energy()
        last_10_eff = energy_cons[-10:] if len(energy_cons) >= 10 else energy_cons
        avg_energy_last_10 = np.mean(last_10_eff) if len(last_10_eff) > 0 else 0
        max_reward_last_10 = np.max(rewards[-10:]) if len(rewards) > 0 else 0

        if avg_energy_last_10 > 0:
            energy_blocks = remaining_energy / avg_energy_last_10
            extrapolated_reward = rewards.sum() + max_reward_last_10 * energy_blocks
        else:
            extrapolated_reward = rewards.sum()  # No extrapolation if no energy consumption data

        extrapolated_rewards.append(extrapolated_reward)

    # ----------- Plot Sum Rates (2x2 subplot) -----------
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    gu_labels = [f"GU {i}" for i in range(env.num_legit_users)]
    idx = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].plot(ep_nums, sum_rate_max[:, idx], label="Max")
            ax[i, j].plot(ep_nums, sum_rate_min[:, idx], label="Min")
            ax[i, j].plot(ep_nums, sum_rate_mean[:, idx], label="Mean")
            ax[i, j].set_title(f"Episode Sum Rate Stats - {gu_labels[idx]}")
            ax[i, j].set_xlabel("Episode")
            ax[i, j].set_ylabel("Sum Rate (bps)")
            ax[i, j].legend()
            idx += 1
    plt.suptitle(f"Per-Episode Sum Rate Stats ({m+1} Layers)")
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_episode_sum_rate_stats.png")
    plt.close()

    # ----------- Per-episode Secrecy Rate Stats -----------
    secrecy_rate_max = np.zeros((ep_count, env.num_legit_users))
    secrecy_rate_min = np.zeros((ep_count, env.num_legit_users))
    secrecy_rate_mean = np.zeros((ep_count, env.num_legit_users))

    for ep in range(ep_count):
        sec_rates = np.array(ep_secrecy_rates_arr[ep])  # (timesteps, num_GUs)
        for gu in range(env.num_legit_users):
            secrecy_rate_max[ep, gu] = sec_rates[:, gu].max()
            secrecy_rate_min[ep, gu] = sec_rates[:, gu].min()
            secrecy_rate_mean[ep, gu] = sec_rates[:, gu].mean()

    # Plot 2x2 secrecy rate stats per GU
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    idx = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].plot(ep_nums, secrecy_rate_max[:, idx], label="Max")
            ax[i, j].plot(ep_nums, secrecy_rate_min[:, idx], label="Min")
            ax[i, j].plot(ep_nums, secrecy_rate_mean[:, idx], label="Mean")
            ax[i, j].set_title(f"Episode Secrecy Rate Stats - GU {idx}")
            ax[i, j].set_xlabel("Episode")
            ax[i, j].set_ylabel("Secrecy Rate (bps)")
            ax[i, j].legend()
            idx += 1
    plt.suptitle(f"Per-Episode Secrecy Rate Stats ({m+1} Layers)")
    plt.tight_layout()
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_episode_secrecy_rate_stats.png")
    plt.close()

    # ---------- Per-timestep Distance to GU Stats ----------
    distances_steps = np.array([ep[:min_len] for ep in ep_distances_to_centroid])  # shape: (num_eps, min_len)
    distance_step_max = distances_steps.max(axis=0)
    distance_step_min = distances_steps.min(axis=0)
    distance_step_avg = distances_steps.mean(axis=0)

    plt.figure()
    plt.plot(timesteps, distance_step_max, label="Max")
    plt.plot(timesteps, distance_step_min, label="Min")
    plt.plot(timesteps, distance_step_avg, label="Mean")
    plt.legend()
    plt.title(f"Per-timestep UAV-BS Distance to GU Centroid Stats ({m+1} Layers)")
    plt.xlabel("Timestep")
    plt.ylabel("Distance to GU Centroid (m)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_timestep_distance_to_centroid.png")
    plt.close()

    # ---------- Per-episode Distance to GU Stats ----------
    ep_count = len(ep_distances_to_centroid)
    distance_max_ep = np.zeros(ep_count)
    distance_min_ep = np.zeros(ep_count)
    distance_mean_ep = np.zeros(ep_count)

    for ep in range(ep_count):
        distances = np.array(ep_distances_to_centroid[ep])
        distance_max_ep[ep] = distances.max()
        distance_min_ep[ep] = distances.min()
        distance_mean_ep[ep] = distances.mean()

    plt.figure()
    plt.plot(ep_nums, distance_max_ep, label="Max")
    plt.plot(ep_nums, distance_min_ep, label="Min")
    plt.plot(ep_nums, distance_mean_ep, label="Mean")
    plt.legend()
    plt.title(f"Per-episode UAV-BS Distance to GU Centroid Stats ({m+1} Layers)")
    plt.xlabel("Episode")
    plt.ylabel("Distance to GU Centroid (m)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_episode_distance_to_centroid.png")
    plt.close()

    # ----------- Plot Energy Efficiency (single plot) -----------
    plt.figure(figsize=(10, 6))
    plt.plot(ep_nums, energy_eff_max, label="Max")
    plt.plot(ep_nums, energy_eff_min, label="Min")
    plt.plot(ep_nums, energy_eff_mean, label="Mean")
    plt.legend()
    plt.title(f"Per-Episode Energy Efficiency Stats ({m+1} Layers)")
    plt.xlabel("Episode")
    plt.ylabel("Energy Efficiency (bps/Hz/J)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_episode_energy_eff.png")
    plt.close()

    # ----------- Plot Energy Consumption (single plot) -----------
    plt.figure(figsize=(10, 6))
    plt.plot(ep_nums, energy_cons_max, label="Max")
    plt.plot(ep_nums, energy_cons_min, label="Min")
    plt.plot(ep_nums, energy_cons_mean, label="Mean")
    plt.legend()
    plt.title(f"Per-Episode Energy Consumption Stats ({m+1} Layers)")
    plt.xlabel("Episode")
    plt.ylabel("Energy Consumption (J)")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_episode_energy_cons.png")
    plt.close()

    # ----------- Plot Extrapolated Rewards -----------
    plt.figure(figsize=(10, 6))
    plt.plot(ep_nums, extrapolated_rewards, label="Extrapolated Reward", color='purple')
    plt.plot(ep_nums, tot_reward_arr, label="Actual Reward", linestyle='--', color='gray')
    plt.legend()
    plt.title(f"Extrapolated vs Actual Rewards ({m+1} Layers)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_episode_extrapolated_rewards.png")
    plt.close()

    critic_losses_inv_arr = []
    for l in range(len(critic_losses)):
        critic_loss_inv = 1 / critic_losses[l]
        critic_losses_inv_arr.append(critic_loss_inv)
    plt.plot(critic_losses_inv_arr, label="Critic Loss")
    plt.legend()
    plt.title("Actor and Critic Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.savefig(f"eve_outputs/plots_eve_outputs/test3/{m+1}_inv_layers_losses.png")
    plt.close()

    print("All good so far")
    total_runtime_end = time.time()
    total_runtime = abs(total_runtime_end - total_runtime_start)
    print(f"Total Time Taken for Experiment with {m+1} Layers to Run: ", total_runtime)

overall_end_time = time.time()
overall_time = abs(overall_end_time - overall_start_time)
print("Total Time Taken for Experiment to Run: ", overall_time)
