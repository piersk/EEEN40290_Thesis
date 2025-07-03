# uav_bs_env.py
from rician_channel import rician_fading_params, rician_channel_gain
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# === GU & Eaves ===
class GroundUser:
    def __init__(self, gu_id, position, cluster_id):
        self.id = gu_id
        self.position = np.array(position)
        self.cluster_id = cluster_id
        self.channel_gain = 1.0

class LegitimateUser(GroundUser):
    def __init__(self, gu_id, position, cluster_id):
        super().__init__(gu_id, position, cluster_id)
        self.subcarrier_allocated = True
        self.secret_rate = 0.0

class Eavesdropper(GroundUser):
    def __init__(self, gu_id, position):
        super().__init__(gu_id, position, cluster_id=-1)
        self.snr_intercept = 0.0

# === UAV Base Classes ===
class UAV:
    def __init__(self, uav_id, position, velocity, tx_power, energy, num_links, mass):
        self.id = uav_id
        self.position = np.array(position, dtype=np.float32)
        self.velocity = velocity
        self.tx_power = tx_power
        self.energy = energy
        self.history = [self.position.copy()]
        self.num_links = num_links
        self.mass = mass
        self.prev_energy_consumption = 0
        self.prev_distance_to_centroid = None

    def move(self, delta_pos):
        self.position += delta_pos
        self.velocity = np.linalg.norm(delta_pos)
        self.history.append(self.position.copy())

    def get_distance_travelled(self):
        if len(self.history) < 2:
            return 0
        return np.linalg.norm(self.history[-1] - self.history[-2])

    def compute_energy_consumption(self, g=9.81, k=6.65, num_rotors=4, rho=1.225, theta=0.3, Lambda=0.15):
        c_t = self.get_distance_travelled()
        n_sum = self.mass
        term1 = (n_sum * g * c_t) / (k * num_rotors)
        term2 = ((n_sum * g) ** 1.5) / np.sqrt(2 * num_rotors * rho * theta)
        term3 = Lambda * c_t / (self.velocity + 1e-6)
        R_kn = 1.0
        term4 = self.tx_power * R_kn
        return term1 + term2 + term3 + term4

# === UAV Base Station ===
class UAVBaseStation(UAV):
    def __init__(self, *args, coverage_radius=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_radius = coverage_radius

# === Environment Class ===
class UAVSecrecyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_uavs = 1
        self.num_legit_users = 5
        self.num_eavesdroppers = 1

        self.P_MAX = 30
        self.E_MAX = 1000
        self.R_MIN = 0.75
        self.V_MAX = 30
        self.xmin, self.ymin, self.zmin = 0, 0, 10
        self.xmax, self.ymax, self.zmax = 150, 150, 122
        #self.xmax, self.ymax, self.zmax = 5000, 5000, 122

        # Penalty weights
        self.pwr_penalty = self.alt_penalty = self.range_penalty = \
        self.min_rate_penalty = self.energy_penalty = self.velocity_penalty = 10

        self.uavs = [UAVBaseStation(0, [0, 0, 0], 0, 10, 1000, num_links=5, mass=2000)]

        self.legit_users = [
            LegitimateUser(i, [np.random.uniform(self.xmin, self.xmax),
                               np.random.uniform(self.ymin, self.ymax), 0], cluster_id=0)
            for i in range(self.num_legit_users)
        ]

        self.eavesdroppers = [
            Eavesdropper(i, [np.random.uniform(self.xmin, self.xmax),
                             np.random.uniform(self.ymin, self.ymax), 0])
            for i in range(self.num_eavesdroppers)
        ]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_uavs * 3 + 3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_uavs * 3,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for uav in self.uavs:
            uav.position = np.random.uniform([0, 0, 10], [200, 200, 100])
            uav.energy = 1000
            uav.history = [uav.position.copy()]
            uav.prev_distance_to_centroid = None
        return self._get_obs(), {}

    def _get_obs(self):
        uav_pos = np.concatenate([uav.position for uav in self.uavs])
        gu_centroid = np.mean([gu.position for gu in self.legit_users], axis=0)
        return np.concatenate([uav_pos, gu_centroid]).astype(np.float32)

    def compute_awgn(self):
        return np.random.normal(0, 1)

    def compute_snr(self, tx_power, noise_power):
        return 10 * np.log10(tx_power / (noise_power**2 + 1e-9))

    def compute_masr(self, alpha, snr_legit, snr_eaves):
        R_legit = alpha * np.log2(1 + snr_legit)
        R_eaves = np.log2(1 + snr_eaves)
        return max(0, R_legit - R_eaves)

    def step(self, action):
        action = np.clip(action, -1, 1)
        energy_cons_penalty = 0

        for i, uav in enumerate(self.uavs):
            # TODO: UPDATE TO SET THE VELOCITY FROM V_MAX TO A MORE DYNAMIC VELOCITY 
            delta = action[i*3:(i+1)*3] * 30.0
            uav.move(delta)
            uav_energy_cons = uav.compute_energy_consumption()
            uav.energy -= uav_energy_cons

            if uav_energy_cons > uav.prev_energy_consumption:
                energy_cons_penalty += 10

        reward = self._compute_reward()
        done = any(uav.energy <= 0 for uav in self.uavs)

        penalties = self.check_constraints()
        total_penalty = sum(v * p for v, p in zip(penalties.values(), [
            self.pwr_penalty, self.alt_penalty, self.range_penalty,
            self.min_rate_penalty, self.energy_penalty, self.velocity_penalty
        ]))
        reward -= (total_penalty + energy_cons_penalty)
        return self._get_obs(), reward, done, False, {}


    def _compute_reward(self):
        bs = self.uavs[0]
        reward = 0
        noise = self.compute_awgn()
        snr_legit = self.compute_snr(bs.tx_power, noise)

        gu_positions = np.array([gu.position for gu in self.legit_users])
        centroid = np.mean(gu_positions, axis=0)
        distance_to_centroid = np.linalg.norm(bs.position - centroid)

        # === Reward closeness to GU cluster centroid ===
        #distance_reward = np.exp(-0.01 * (distance_to_centroid) / 20)  # [0,1], sharper falloff
        #reward += 50 * distance_reward  # scaled to be dominant initially
        distance_reward = (1 / np.log1p(distance_to_centroid))
        reward += 100 * distance_reward

        # === Penalty for sudden moves away from centroid ===
        if len(bs.history) >= 2:
            prev_dist = np.linalg.norm(bs.history[-2] - centroid)
            if distance_to_centroid > prev_dist:
                reward -= 50  # discourage moving away

            if distance_to_centroid < prev_dist:
                reward += 20

        # === Bonus rewards for stability near centroid ===
        # Penalty for large distance from centroid
        if distance_to_centroid >= 50:
            reward -= 50

            '''
        if distance_to_centroid < 200:
            reward += 10
        if distance_to_centroid < 100:
            reward += 25
        if distance_to_centroid < 50:   # hover threshold
            reward += 50                # bonus for hovering close
            '''
        if distance_to_centroid < 25:
            reward += 20 
        if distance_to_centroid < 10:
            reward += 50 
        if distance_to_centroid < 5:
            reward += 75

        # === Energy efficiency reward ===
        energy_consumption = bs.compute_energy_consumption()
        bs.prev_energy_consumption = energy_consumption
        reward += 10 / (1.0 + energy_consumption)  # inverse relation

        return reward

    '''
    def _compute_reward(self):
        bs = self.uavs[0]
        reward = 0
        noise = self.compute_awgn()
        snr_legit = self.compute_snr(bs.tx_power, noise)

        gu_positions = np.array([gu.position for gu in self.legit_users])
        centroid = np.mean(gu_positions, axis=0)
        dist_to_centroid = np.linalg.norm(bs.position - centroid)

        # --- REWARD COMPONENTS ---
        if bs.prev_distance_to_centroid is not None:
            if dist_to_centroid < bs.prev_distance_to_centroid:
                reward += 50 # positive for improvement
            else:
                reward -= 10  # penalty for divergence
        bs.prev_distance_to_centroid = dist_to_centroid

        if dist_to_centroid < 100:
            reward += 75 # bonus for staying close to centroid

        reward -= 0.05 * dist_to_centroid

        energy_eff = 1.0 / (1.0 + bs.compute_energy_consumption())
        reward += 10 * energy_eff
        bs.prev_energy_consumption = bs.compute_energy_consumption()

        for gu in self.legit_users:
            d = np.linalg.norm(bs.position - gu.position)
            if d < bs.coverage_radius:
                snr_eaves = max(self.compute_snr(bs.tx_power, noise) for eve in self.eavesdroppers)
                masr = self.compute_masr(alpha=1, snr_legit=snr_legit, snr_eaves=snr_eaves)
                reward += 8 * masr

        return reward
    '''

    def check_constraints(self):
        violations = {
            "power": sum(uav.tx_power for uav in self.uavs) > self.P_MAX,
            "altitude": False, "range": False, "min_rate": False,
            "energy": False, "velocity": False
        }
        for uav in self.uavs:
            x, y, z = uav.position
            if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
                violations["range"] = True
            if not (self.zmin <= z <= self.zmax):
                violations["altitude"] = True
            if uav.energy <= 0 or uav.energy > self.E_MAX:
                violations["energy"] = True
            if uav.velocity > self.V_MAX:
                violations["velocity"] = True
        return violations

    def render(self):
        pass
