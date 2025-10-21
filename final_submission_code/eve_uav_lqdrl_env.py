# uav_lqdrl_env.py
import numpy as np
import gymnasium as gym
import math
from gymnasium import spaces
from scipy.stats import rice
from scipy.special import iv

# === PID Controller Class ===
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# === Ground User Base Classes ===
class GroundUser:
    def __init__(self, gu_id, position, cluster_id):
        self.id = gu_id
        self.position = np.array(position)
        self.cluster_id = cluster_id
        self.channel_gain = 1.0
        self.subcarrier_allocated = False

class LegitimateUser(GroundUser):
    def __init__(self, gu_id, position, cluster_id):
        super().__init__(gu_id, position, cluster_id)
        self.subcarrier_allocated = True
        self.secret_rate = 0.0

        # LGU filter generated additive noise signature from UAV-BS
        def filter_additive_noise_signature(self, signal, noise_pattern):
            filt = signal - noise_pattern * 0.95
            return filt

class Eavesdropper(GroundUser):
    def __init__(self, gu_id, position, cluster_id):
        super().__init__(gu_id, position, cluster_id)
        self.eve_rate = []

# === UAV Base Classes ===
class UAV:
    def __init__(self, uav_id, position, velocity, tx_power, energy, num_links, mass):
        self.id = uav_id
        self.position = np.array(position, dtype=np.float32)
        self.velocity = velocity
        self.tx_power = tx_power
        self.energy = energy
        self.history = [self.position.copy()]
        self.num_links = num_links  # Changed from links to num_links
        self.mass = mass
        self.prev_energy_consumption = 0    # Previous energy consumption initialised to 0 J
        self.prev_dist_to_centroid = 0
        self.prev_tx_power = 0
        self.prev_velocity = 0
        self.zeta = 1   # Default zeta value = 1
        self.yaw = 0.0
        self.pitch = 0.0
        self.curr_sum_rates = []
        self.energy_efficiency = 0

    # Function to move UAV in 3-D Cartesian Space
    def move(self, delta_pos, dist, bounds):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        proposed_pos = self.position + delta_pos
        clipped_pos = np.clip(proposed_pos, [xmin, ymin, zmin], [xmax, ymax, zmax])
        #delta_clipped = clipped_pos - self.position
        delta_clipped = self.position - clipped_pos

        self.position += delta_pos
        print("Change in UAV Position: ", delta_pos)
        print("Clipped Change in UAV Position: ", delta_clipped)
        self.velocity = dist
        print("UAV Velocity: ", self.velocity)
        self.history.append(self.position.copy())

    def get_distance_travelled(self):
        if len(self.history) < 2:
            return 0
        return np.linalg.norm(self.history[-1] - self.history[-2])

    def update_orientation_and_move(self, yaw_cmd, pitch_cmd, throttle, delta_t, bounds, velocity):
        self.velocity = velocity
        self.yaw += yaw_cmd * delta_t
        self.pitch += pitch_cmd * delta_t
        self.pitch = np.clip(self.pitch, -np.pi/2, np.pi/2)

        velocity = throttle * self.velocity
        dx = velocity * np.cos(self.pitch) * np.cos(self.yaw)
        dy = velocity * np.cos(self.pitch) * np.sin(self.yaw)
        dz = velocity * np.sin(self.pitch)

        delta_pos = np.array([dx, dy, dz])
        velocity *= delta_t
        return delta_pos, velocity, bounds

    # 100 J/s avionics power from d'Andrea et al (2014)
    def compute_energy_consumption(self, tx_power_arr, sum_rate_arr, g=9.81, k=6.65, num_rotors=4, rho=1.225, theta=0.0507, Lambda=100):
        c_t = self.get_distance_travelled()
        c_t = abs(c_t)
        n_sum = self.mass
        travel = (n_sum * g * c_t) / (k * num_rotors)
        hover = ((n_sum * g) ** 1.5) / np.sqrt(2 * num_rotors * rho * theta)
        min_v = 5
        eff_v = max(self.velocity, min_v)
        avionics = Lambda * c_t / (eff_v) # Avoid dividing by 0
        comms = 0
        for k in range(len(sum_rate_arr)):
            tx_power = 10**(tx_power_arr[k]/10)/1000
            comms += tx_power * sum_rate_arr[k]
        energy_cons = travel + hover + avionics + comms
        return energy_cons

class UAVBaseStation(UAV):
    def __init__(self, *args, coverage_radius=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_radius = coverage_radius
        self.noise_signatures = []
        self.legitimate_users = []
        self.secrecy_rate = []

    # Function to generate additive noise signature
    # Additive noise signature should increase as Eve proximity to LGU decreases
    # Create function for Legit GU to filter the noise signature out
    # Detail signalling mechanism for LGU filtering in thesis report 
    def generate_additive_noise_signature(self, noise_factor, freq, duration):
        noise_pwr = self.tx_power * noise_factor
        freq_bands = [freq * ((1 + i) * noise_factor) for i in range(1)]

        pattern = {
            'frequencies': np.random.choice(freq_bands, size=10),
            'durations': np.random.uniform(0.1, 0.5, size=10),
            'power': noise_pwr
        }

        self.noise_signature = pattern
        return pattern

# TODO: Include relayed links between UAVs as a list
class UAVRelay(UAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# TODO: Include eavesdropping GUs being interfered with as a list
class UAVJammer(UAV):
    def __init__(self, *args, noise_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_power = noise_power

# State Space:
# Energy consumed per timestep as per the equation: E_remain(t) = E_remain(t-1) - E_cons
# UAV Position (q_UAV in thesis and Zhang et. al (2025) but denoted as c(t) in Silviaranti et. al (2025))
# GU clustering u_K
# Dimensions: 2K, where K=number of GUs + 1 for energy consumption + 1 for UAV position - amounting to 2K+4 state space dimensions
# Action Space:
# UAV Trajectory
# Dynamic power and resource allocation
# NOMA user group grouping
class UAV_LQDRL_Environment(gym.Env):
    def __init__(self):
        super().__init__()
        self.time = 0
        self.delta_t = 1
        self.num_uavs = 1
        self.num_legit_users = 4
        self.num_eves = 2
        self.K_FACTOR = 10
        self.SHADOWING_SIGMA = 4
        self.NOISE_LOS = -100 # -100 dBm 
        self.NOISE_NLOS = -80 # -80 dBm 
        self.A1 = 4
        self.A2 = 0.1
        self.PATHLOSS_COEFF = 3 # Empirical value for urban terrain

        self.P_MAX = 30     # 30 dBm
        self.E_MAX = 50e03  # 50kJ 
        self.R_MIN = 9.5e06 # R_min = 9.5 Mbps
        self.V_MAX = 50     # 50 m/s 
        self.xmin, self.ymin, self.zmin = 0, 0, 10 
        self.xmax, self.ymax, self.zmax = 150, 150, 122
        # Penalty values are for scaling the reward based on constraint violations
        self.pwr_penalty = self.alt_penalty = self.range_penalty = \
        self.min_rate_penalty = self.energy_penalty = self.velocity_penalty = 0.15

        self.f_carr = 1e06  # 1 MHz carrier frequency
        
        self.uavs = [
            UAVBaseStation(0, [0, 0, 0], 0, self.P_MAX, self.E_MAX, num_links=self.num_legit_users, mass=1.46)
        ]

        self.legit_users = [
            LegitimateUser(i, [np.random.uniform(self.xmin, self.xmax), 
                               np.random.uniform(self.ymin, self.ymax), 
                               0], 
                           cluster_id = 0) 
            for i in range(self.num_legit_users)
        ]

        self.eves = [
            Eavesdropper(i, [np.random.uniform(self.xmin, self.xmax),
                              np.random.uniform(self.ymin, self.ymax),
                              0],
                        cluster_id = 1)
            for i in range(self.num_eves)
        ]

        # 2K+4 Dimensional Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((2 * self.num_legit_users) + 4,), dtype=np.float32
        )

        # 5-D Action Space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self.yaw_pid = PIDController(kp=5, ki=1.2, kd=1.875)
        self.pitch_pid = PIDController(kp=5, ki=1.2, kd=1.875)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for uav in self.uavs:
            uav.position = np.random.uniform([self.xmin, self.ymin, self.zmin], [self.xmax, self.ymax, self.zmax])
            uav.energy = self.E_MAX
            uav.history = [uav.position.copy()]
            uav.yaw = 0.0
            uav.pitch = 0.0
            uav.prev_distance_to_centroid = None
        return self._get_obs(), {}

    # Function returns the observation state space of the environment (s)
    def _get_obs(self):
        uav_pos = np.concatenate([uav.position for uav in self.uavs])
        gu_pos = np.concatenate([gu.position[:2] for gu in self.legit_users])
        uav_energy = np.array([self.uavs[0].energy], dtype=np.float32)
        return np.concatenate([uav_pos, gu_pos, uav_energy]).astype(np.float32)

    def dbm_to_watt(self, dbm):
        return (10 ** ((dbm) / 10)) / 1000

    def compute_zeta(self, dist_to_centroid):
        zeta = 1 - ((self.xmax - dist_to_centroid) / (self.xmax - self.xmin))
        return zeta

    # Function to compute the velocity of the UAV for any timestep t
    # Zeta must be computed as a variable between 0 and 1 to scale against V_MAX
    def compute_velocity(self, zeta):
        min_v = 5
        v = zeta * self.V_MAX
        v = max(v, min_v)
        print("UAV Velocity: ", v, " m/s")
        return v

    def get_uav_position(self):
        for uav in self.uavs:
            position = uav.position
        return position

    def get_remaining_energy(self):
        for uav in self.uavs:
            e_remain = uav.energy
        return e_remain 

    def get_energy_efficiency(self):
        for uav in self.uavs:
            energy_eff = uav.energy_efficiency
        return energy_eff

    def get_energy_cons(self):
        for uav in self.uavs:
            e_cons = abs(self.E_MAX - uav.energy)
        return e_cons

    def get_uav_history(self):
        hist_arr = []
        for uav in self.uavs:
            hist = uav.history
            hist_arr.append(hist)
        return hist_arr

    def get_sum_rates(self):
        return self.curr_sum_rates

    def get_secrecy_rates(self):
        for uav in self.uavs:
            secrecy_rates = uav.secrecy_rate
        return secrecy_rates

    def compute_awgn(self):
        return np.random.normal(0, 0.5)

    def compute_snr(self, tx_power, channel_gain, noise_power):
        #return 10 * np.log10(tx_power / noise_power**2 + 1e-9)
        snr = (tx_power * abs((channel_gain**2))) / (noise_power**2)
        return snr

    def compute_r_k(self, subchan_bw, snr):
        sum_rate_k = subchan_bw * np.log2(1 + snr)
        return sum_rate_k

    def compute_power_coefficients(self, channel_gain, channel_gain_arr):
        # Set minimum channel gain to 0.01
        h_tot = 0
        for n in range(len(channel_gain_arr)):
            h_tot += abs(channel_gain_arr[n])
        hnorm = abs(channel_gain) / h_tot
        h_prop = 1 - hnorm
        delta = (1 / (self.num_legit_users)) * h_prop
        return delta

    def compute_power_allocation(self, pwr_coeff):
        power_alloc = self.P_MAX * pwr_coeff
        return power_alloc

    def apply_noma_grouping(self, action_scalar):
        group_id = int(np.clip(action_scalar * self.num_legit_users, 0, self.num_legit_users - 1))
        for i, gu in enumerate(self.legit_users):
            gu.cluster_id = group_id

    def compute_pathloss(self, f_carrier, uav_pos, gu_pos):
        dist = np.linalg.norm(uav_pos - gu_pos) 
        fs_ploss = 20 * np.log10(dist) + 20 * np.log10(f_carrier) + 20 * np.log10((4 * np.pi) / 2.99e08)
        return fs_ploss

    def rician_channel(self, distance, uav_pos, gu_pos, pl_coeff):
        ref_pwr_gain = (self.dbm_to_watt(self.P_MAX) / self.num_legit_users) * 1**(-pl_coeff) # 0.25 W
        tx_pwr_gain = ref_pwr_gain * (distance**(-pl_coeff))
        theta = np.arcsin(uav_pos[2] / distance)
        K = self.A1 * np.exp(self.A2 * theta)
        #g = np.sqrt(K / (K + 1)) * 1 + np.sqrt(1 / (K + 1)) * self.compute_awgn()
        g = np.sqrt(K / (K + 1)) * 1 - np.sqrt(1 / (K + 1)) * self.compute_awgn()
        channel_gain = g * (distance**(-pl_coeff))
        return channel_gain

    # FUNCTION NOT IN USE ANYMORE
    def compute_channel_gain(self, pathloss, awgn):
        h = pathloss * awgn
        return h

    def compute_subcarrier_allocation(self, f_carrier):
        i = 0
        bw_arr = [0 for i in range(self.num_legit_users)]
        for gus in self.legit_users:
            bw_arr[i] = f_carrier / self.num_legit_users
            i += 1
        return bw_arr

    def compute_sum_rate(self, bw_arr, snr):
        sum_rate_arr = []
        for k in range(self.num_legit_users):
            r_k = bw_arr[k] * np.log2(1 + abs(snr))
            sum_rate_arr.append(r_k)
        return sum_rate_arr

    def compute_eve_noise_factor(self, eve_dist_to_gu):
        d_max = np.sqrt((self.xmax**2) + (self.ymax**2))
        eve_noise_factor = d_max / (d_max - eve_dist_to_gu)
        return eve_noise_factor

    def compute_secrecy_rate(self, sum_rate, worst_case_eve_rate):
        secrecy_rate = abs(sum_rate - worst_case_eve_rate)
        return secrecy_rate 

    def _compute_energy_efficiency(self, sum_rate, energy_cons):
        return sum_rate / (energy_cons)

    def _compute_gu_centroid(self, gu_positions):
        gu_centroid = np.mean(gu_positions, axis=0)
        gu_centroid[2] += 10
        return gu_centroid 

    def _compute_gu_pos_diff(self, gu_positions):
        gu_diffs = []
        other_positions = len(gu_positions) - 1
        for i in range(len(gu_positions)):
            for j in range(i, other_positions):
                gu_diff = abs(gu_positions[i] - gu_positions[j + 1])
                gu_diffs.append(gu_diff)
        return gu_diffs

    # Function to compute the action (a) taken by the UAV agent(s) based on state (s)
    def step(self, action):
        action = np.clip(action, -1, 1)
        #action += np.random.normal(0, 5e-02, size=action.shape)
        energy_cons_penalty = 0
        reward_boost = 0
        zeta = 1
        for i, uav in enumerate(self.uavs):
            gu_positions = np.array([gu.position for gu in self.legit_users])
            gu_centroid = self._compute_gu_centroid(gu_positions)
            dist_to_centroid = np.linalg.norm(uav.position - gu_centroid)
            # Only slow down speed when reasonably close to the GU centroid
            if dist_to_centroid <= 25:
                zeta = self.compute_zeta(dist_to_centroid)
            else:
                zeta = 1
            dist = self.compute_velocity(zeta) * self.delta_t
            print("UAV Velocity (step): ", dist, "m/s")
            
            # Direction vector from UAV to centroid (normalized)
            direction_to_centroid = gu_centroid - uav.position
            direction_to_centroid /= (np.linalg.norm(direction_to_centroid) + 1e-8)  # avoid div by 0

            # Weighted combination of policy action and centroid direction
            steering_ratio = 0.75  # higher = more directed toward centroid
            action_vector = (steering_ratio * direction_to_centroid + (1 - steering_ratio) * action[:3])
            action_vector /= (np.linalg.norm(action_vector) + 1e-8)  # normalize again

            # Compute delta movement
            delta = action_vector * dist

            # Apply movement
            uav.move(delta, dist, [self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax])

            #delta = action[i*3:(i+1)*3] * v
            delta = action[:3] * dist
            uav.move(delta, dist, [self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax])

            channel_gain_arr = []
            pwr_delta_arr = []
            awgn_arr = []
            dist_from_gu_arr = []
            i = 0
            for gu in self.legit_users:
                uav_pos = uav.position
                gu_pos = gu.position
                dist_from_gu = np.linalg.norm(uav_pos - gu_pos)
                dist_from_gu_arr.append(dist_from_gu)
                #pathloss = self.compute_pathloss(self.f_carr, uav_pos, gu_pos)
                awgn = self.compute_awgn()
                # Ensure the AWGN value is positive
                awgn = np.sqrt(awgn**2)
                awgn_arr.append(awgn)
                channel_gain = self.rician_channel(dist_from_gu, uav_pos, gu_pos, self.PATHLOSS_COEFF)
                i += 1
                print(f"GU {i} Channel Gain: ", channel_gain)
                channel_gain_arr.append(channel_gain)
                print(f"Distance between UAV & GU {i}: ", dist_from_gu, " m")

            tx_power_arr = []
            snr_arr = []
            bw_arr = self.compute_subcarrier_allocation(self.f_carr)
            sum_rate_arr = []

            eve_channel_gain_arr = []
            eve_tx_pwr_arr = []
            bw_arr = self.compute_subcarrier_allocation(self.f_carr)
            eve_bw_arr = bw_arr[:self.num_eves]
            noise_factor_arr = [] 
            all_eves_noise_signature_arr = []
            all_eves_dist_to_gus_arr = []
            all_eves_snr_arr = []
            eve_rate_arr = []
            for k, gu in enumerate(self.legit_users):
                eve_dist_to_gu_arr = []
                noise_signature_arr = []
                eve_snr_arr = []
                for j, eve in enumerate(self.eves):
                    eve_dist_to_gu = abs(np.linalg.norm(gu.position - eve.position))
                    eve_dist_to_gu_arr.append(eve_dist_to_gu)
                    eve_noise_factor = self.compute_eve_noise_factor(eve_dist_to_gu)
                    noise_factor_arr.append(eve_noise_factor)
                    uav_pos = uav.position
                    eve_pos = eve.position
                    eve_dist_to_uav = abs(np.linalg.norm(uav.position - eve.position))
                    eve_channel_gain = self.rician_channel(eve_dist_to_gu, uav_pos, eve_pos, self.PATHLOSS_COEFF)
                    print(f"Eve {j} Channel Gain for GU {k}: ", eve_channel_gain)
                    eve_tx_pwr = self.P_MAX / self.num_eves
                    noise_signature = uav.generate_additive_noise_signature(eve_noise_factor, self.f_carr, self.delta_t)
                    noise_signature_arr.append(noise_signature['power'])
                    noise_env = self.dbm_to_watt(self.NOISE_LOS)
                    tot_noise = noise_signature['power'] + noise_env
                    eve_snr = self.compute_snr(eve_tx_pwr, eve_channel_gain, tot_noise)
                    print(f"Eve {j} SNR for GU {k}: ", eve_snr)
                    eve_snr_arr.append(eve_snr)
                    eve_subchan_bw = self.f_carr / self.num_legit_users
                    eve_rate = self.compute_r_k(eve_subchan_bw, eve_snr)
                    print(f"Eavesdropping Rate {j} for GU {k}: ", eve_rate)
                    eve_rate_arr.append(eve_rate)

            print("Eavesdropper SNR: ", eve_snr_arr)
            eve_rate_arr.sort(reverse=True)
            print("Eavesdropping Rate: ", eve_rate_arr)
            used_noise_signatures = noise_signature_arr[:self.num_legit_users]
            print("Used Noise Signatures: ", used_noise_signatures)
            used_noise_signatures.sort(reverse=True)
            uav.noise_signatures = used_noise_signatures 
            worst_case_eve_rate_arr = eve_rate_arr[:self.num_legit_users]

            print("Worst-Case Eavesdropping Rate: ", worst_case_eve_rate_arr)
            
            for k, gu in enumerate(self.legit_users):
                print("=================================")
                gain = channel_gain_arr[k]
                pwr_delta = self.compute_power_coefficients(gain, channel_gain_arr)
                print(f"Power scaling variable {k}: ", pwr_delta)
                tx_power = self.compute_power_allocation(pwr_delta)
                print(f"Transmit Power {k}: ", tx_power, "dBm")
                tx_power = self.dbm_to_watt(tx_power)
                print(f"Transmit Power {k}: ", tx_power, "W")
                tx_power_arr.append(tx_power)
                noise_kn = self.dbm_to_watt(self.NOISE_LOS)
                print(f"AWGN {k}: ", noise_kn)
                snr_legit = self.compute_snr(tx_power, gain, noise_kn)
                print(f"SNR {k}: ", snr_legit)
                print(f"SNR {k}: ", 20 * np.log10(snr_legit), "dB")
                snr_arr.append(snr_legit)
                bw_subchan = bw_arr[k]
                print(f"Subchannel Bandwidth {k}: ", bw_subchan, "Hz")
                r_kn = self.compute_r_k(bw_subchan, snr_legit)
                print(f"Data rate {k}: ", r_kn, "bps")
                sum_rate_arr.append(r_kn)
            self.curr_sum_rates = sum_rate_arr

            secrecy_rate_arr = []
            for m in range(self.num_legit_users):
                secrecy_rate = sum_rate_arr[m] - eve_rate_arr[m]
                print(f"Secrecy Rate {m}: ", secrecy_rate)
                secrecy_rate_arr.append(secrecy_rate)

            secrecy_rate_arr = [float(sr) if np.isfinite(sr) else 0.0 for sr in secrecy_rate_arr]
            assert len(secrecy_rate_arr) == self.num_legit_users, (
                f"Expected {self.num_legit_users} secrecy rates, got {len(secrecy_rate_arr)}"
            )
            uav.secrecy_rate = secrecy_rate_arr
            print("Secrecy Rate: ", secrecy_rate_arr)
            print("Secrecy Rate: ", uav.secrecy_rate)

        # Energy consumption should only occur once per step for 1 UAV and once per UAV per step if multiple UAV-BSs are to be used
        sum_rate_hz_arr = []
        for i in range(len(sum_rate_arr)):
            sum_rate_hz = (sum_rate_arr[i] / self.f_carr)
            sum_rate_hz_arr.append(sum_rate_hz)

        uav_energy_cons = uav.compute_energy_consumption(tx_power_arr, sum_rate_hz_arr)
        print("UAV Energy Consumption: ", uav_energy_cons)
        uav.energy -= uav_energy_cons

        gu_diffs = []
        gu_diffs = self._compute_gu_pos_diff(dist_from_gu_arr)

        # Reward boost for UAV becoming more equidistant between the GUs
        for i in range(len(gu_diffs)):
            if gu_diffs[i] <= 40:
                reward_boost += 0.05
            if gu_diffs[i] <= 20:
                reward_boost += 0.075
            if gu_diffs[i] <= 10:
                reward_boost += 0.1

        if dist_to_centroid <= 30 and uav.position[2] >= self.zmin and dist_to_centroid >= 10:
            reward_boost += 0.5

        if (dist_to_centroid <= uav.prev_dist_to_centroid) and not (uav.position[2] <= self.zmin):
            reward_boost += 0.1

        r_sum = np.sum(sum_rate_arr, axis=0)
        reward = self._compute_reward(sum_rate_arr, uav_energy_cons)
        reward += reward_boost * reward
        if (uav.position[2] <= 0):
            energy_cons_penalty += 0.95
        uav.prev_dist_to_centroid = dist_to_centroid
        done = any(uav.energy <= 0 for uav in self.uavs)
        penalties = self.check_constraints()
        total_penalty = sum(v * p for v, p in zip(penalties.values(), [
            self.pwr_penalty, self.alt_penalty, self.range_penalty,
            self.min_rate_penalty, self.energy_penalty, self.velocity_penalty
        ]))
        print("Reward Boost Factor: ", reward_boost)
        print("Energy Consumption Penalty Factor: ", energy_cons_penalty)
        print("Total Penalties Factor: ", total_penalty)
        reward -= reward * (total_penalty + energy_cons_penalty)
        print("Reward for step: ", reward)
        
        return self._get_obs(), reward, done, False, {}

    def _compute_reward(self, sum_rate_arr, energy_consumption):
        # Compute reward for the UAV agent
        reward = 0
        grant_reward = False
        masr = np.sum(sum_rate_arr, axis=0)
        uav = self.uavs[0]
        uav.prev_energy_consumption = energy_consumption
        energy_eff = self._compute_energy_efficiency(masr, energy_consumption)
        uav.energy_efficiency = energy_eff 
        print("Energy Efficiency: ", energy_eff)
        j = 0
        for k in range(0, self.num_legit_users):
            sum_rate = sum_rate_arr[k]
            print(f"Sum Rate {k}: ", sum_rate)
            if sum_rate > self.R_MIN:
                reward += energy_eff / self.num_legit_users
                j += 1
                if k == (self.num_legit_users - 1) and k == (j - 1):
                    grant_reward = True

        print("Reward Allocated: ", grant_reward)

        if grant_reward == True:
            reward = energy_eff
            print("All users above R_min")
        else:
            #reward -= (self.num_legit_users - j) * reward
            print("Not all users above R_min")

        return reward

    def check_constraints(self):
        violations = {
            "range": False,
            "altitude": False,
            "energy": False,
            "velocity": False,
            "power": False,
            "min_rate": False,
        }

        for uav in self.uavs:
            x, y, z = uav.position
            if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
                violations["range"] = True
            if uav.energy <= 0:
                violations["energy"] = True
            if uav.energy > self.E_MAX:
                violations["energy"] = True
            if not (self.zmin <= z <= self.zmax):
                violations["altitude"] = True
            if uav.velocity > self.V_MAX:
                violations["velocity"] = True

        return violations

    def render(self):
        pass
