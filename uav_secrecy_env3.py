# uav_secrecy_env.py
from rician_channel import rician_fading_params, rician_channel_gain
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# === Ground User Base Classes ===
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
        self.num_links = num_links  # Changed from links to num_links
        self.mass = mass
        self.prev_energy_consumption = 0

    def move(self, delta_pos):
        self.position += delta_pos
        self.velocity = np.linalg.norm(delta_pos)
        self.history.append(self.position.copy())

    def get_distance_travelled(self):
        if len(self.history) < 2:
            return 0
        return np.linalg.norm(self.history[-1] - self.history[-2])

    def compute_energy_consumption(self, g=9.81, k=6.65, num_rotors=4, rho=1.225, theta=0.3, Lambda=5):
        c_t = self.get_distance_travelled()
        n_sum = self.mass
        term1 = (n_sum * g * c_t) / (k * num_rotors)
        term2 = ((n_sum * g) ** 1.5) / np.sqrt(2 * num_rotors * rho * theta)
        term3 = Lambda * c_t / (self.velocity + 1e-6)
        R_kn = 1.0
        term4 = self.tx_power * R_kn
        return term1 + term2 + term3 + term4

class UAVBaseStation(UAV):
    def __init__(self, *args, coverage_radius=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_radius = coverage_radius
        self.legitimate_users = []

class UAVRelay(UAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class UAVJammer(UAV):
    def __init__(self, *args, noise_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_power = noise_power


# === Environment Class ===
class UAVSecrecyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_uavs = 3
        self.num_legit_users = 5
        self.num_eavesdroppers = 3

        # === Constraints ===
        self.P_MAX = 30
        self.E_MAX = 1000
        self.R_MIN = 0.75
        self.V_MAX = 20
        self.xmin, self.ymin, self.zmin = 0, 0, 10
        self.xmax, self.ymax, self.zmax = 1500, 1500, 122
        self.pwr_penalty = self.alt_penalty = self.range_penalty = \
        self.min_rate_penalty = self.energy_penalty = self.velocity_penalty = 10

        # === UAVs ===
        self.uavs = [
            UAVBaseStation(0, [0, 0, 0], 0, 10, 1000, num_links=5, mass=2000),
            UAVRelay(1, [50, 50, 0], 0, 10, 1000, num_links=2, mass=2000),
            UAVJammer(2, [0, 0, 0], 0, 10, 1000, num_links=0, mass=2000, noise_power=2.0)
        ]

        # === Users ===
        self.legit_users = [
            LegitimateUser(i, [np.random.uniform(self.xmin, self.xmax),
                               np.random.uniform(self.ymin, self.ymax),
                               0], cluster_id=0) for i in range(self.num_legit_users)
        ]

        self.eavesdroppers = [
            Eavesdropper(i, [np.random.uniform(self.xmin, self.xmax),
                             np.random.uniform(self.ymin, self.ymax),
                             0]) for i in range(self.num_eavesdroppers)
        ]

        # === Communication Topology ===
        self.link_topology = {
            # UAV ID : list of connected entity IDs (e.g. UAVs or GUs)
            0: list(range(self.num_legit_users)),     # BS connects to all GUs
            1: [0],                                   # Relay connects to BS
            2: [0, 1]                                 # Jammer targets BS and Relay
        }

        # === Spaces ===
        # TODO: POSSIBLE CHANGES REQUIRED FOR THE STATE SPACE DIMENSIONS/SHAPES
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_uavs * 3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_uavs * 3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for uav in self.uavs:
            uav.position = np.random.uniform([0, 0, 10], [200, 200, 100])
            uav.energy = 1000
            uav.history = [uav.position.copy()]
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([uav.position for uav in self.uavs]).astype(np.float32)

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
            delta = action[i*3:(i+1)*3] * 5.0
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

        for uav in self.uavs:
            if uav.position[0] <= self.xmax:
                reward += 1
            if uav.position[1] <= self.ymax:
                reward += 1
            if uav.position[2] <= self.zmax:
                reward += 1

            energy_consumption = uav.compute_energy_consumption()

            if energy_consumption <= self.E_MAX:
                reward += 1
            if energy_consumption < uav.prev_energy_consumption:
                reward += 10
            
            if uav.velocity <= self.V_MAX:
                reward += 10

        for gu in self.legit_users:
            d = np.linalg.norm(bs.position - gu.position)

        for gu in self.legit_users:
            d = np.linalg.norm(bs.position - gu.position)
            if d < bs.coverage_radius:
                snr_eaves = max(self.compute_snr(bs.tx_power, noise) for eve in self.eavesdroppers)
                masr = self.compute_masr(alpha=1, snr_legit=snr_legit, snr_eaves=snr_eaves)
                reward += 10 * masr

        prev = energy_consumption
        uav.prev_energy_consumption = prev

        return reward

    def check_constraints(self):
        violations = {
            "power": sum(uav.tx_power for uav in self.uavs) > self.P_MAX,
            "altitude": False,
            "range": False,
            "min_rate": False,
            "energy": False,
            "velocity": False
        }
        for uav in self.uavs:
            x, y, h = uav.position
            if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
                violations["range"] = True
            if not (self.zmin <= h <= self.zmax):
                violations["altitude"] = True
            if uav.energy <= 0 or uav.energy > self.E_MAX:
                violations["energy"] = True
            if uav.velocity > self.V_MAX:
                violations["velocity"] = True
        return violations

    def render(self):
        pass
