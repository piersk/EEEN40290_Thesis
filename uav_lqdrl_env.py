# uav_lqdrl_env.py
# TODO: Merge problem from LQ-DRL problem & Secrecy Rate Problem
# Implementation of the proposed LQ-DRL algorithm presented in Silvirianti et al. (2025)
# All GU instantiation should just be handled by GroundUser parent class for this program
# Possibly remove the Relay & Jammer UAVs for this implementation/version
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# TODO: Update the UAVs & GUs for this to only include the Legitimate GUs (no subclasses required for this prototype) 
# Do not consider secrecy for this model
# === Ground User Base Classes ===
class GroundUser:
    def __init__(self, gu_id, position, cluster_id):
        self.id = gu_id
        self.position = np.array(position)
        self.cluster_id = cluster_id
        self.channel_gain = 1.0
        self.subcarrier_allocated = False

# TODO: Remove this subclass if possible
class LegitimateUser(GroundUser):
    def __init__(self, gu_id, position, cluster_id):
        super().__init__(gu_id, position, cluster_id)
        self.subcarrier_allocated = True
        self.secret_rate = 0.0

# TODO: Only UAV-BS considered for this program
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
        #self.links = np.array(num_links)
        self.mass = mass
        self.prev_energy_consumption = 0    # Previous energy consumption initialised to 0 J
        self.prev_tx_power = 0
        self.prev_velocity = 0
        self.zeta = 1   # Default zeta value = 1


    # TODO: Call compute_velocity here
    # Function to move UAV in 3-D Cartesian Space
    def move(self, delta_pos):
        self.position += delta_pos
        self.velocity = np.linalg.norm(delta_pos)
        # TODO: FIGURE OUT HOW TO CALCULATE ZETA
        #self.velocity = self.compute_velocity(self.zeta)
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
        energy_cons = term1 + term2 + term3 + term4
        return energy_cons

    def compute_zeta(self, dist_to_centroid):
        zeta = 1 - ((self.xmax - dist_to_centroid) / (self.xmax - self.xmin))
        return zeta

    # TODO: Compute zeta either in another function based on the observed state or here
    # Function to compute the velocity of the UAV for any timestep t
    # Zeta must be computed as a variable between 0 and 1 to scale against V_MAX
    def compute_velocity(self, zeta):
        v = zeta * self.V_MAX
        return v

class UAVBaseStation(UAV):
    def __init__(self, *args, coverage_radius=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_radius = coverage_radius
        self.legitimate_users = []

# TODO: Include relayed links between UAVs as a list
class UAVRelay(UAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# TODO: Include eavesdropping GUs being interfered with as a list
class UAVJammer(UAV):
    def __init__(self, *args, noise_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_power = noise_power


# TODO: Program memory experience relay to include the following:
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
        self.num_uavs = 1
        self.num_legit_users = 4

        self.P_MAX = 30
        self.E_MAX = 500e3  # 500kJ in paper
        self.R_MIN = 0.75
        self.V_MAX = 50     # 50 m/s in paper
        self.xmin, self.ymin, self.zmin = 0, 0, 10 
        self.xmax, self.ymax, self.zmax = 1500, 1500, 122
        self.pwr_penalty = self.alt_penalty = self.range_penalty = \
        self.min_rate_penalty = self.energy_penalty = self.velocity_penalty = 10
        
        self.uavs = [
            UAVBaseStation(0, [0, 0, 0], 0, 10, 1000, num_links=4, mass=2000)
        ]

        self.legit_users = [
            LegitimateUser(i, [np.random.uniform(self.xmin, self.xmax), 
                               np.random.uniform(self.ymin, self.ymax), 
                               np.random.uniform(self.zmin, self.zmax)], 
                           cluster_id = 0) 
            for i in range(self.num_legit_users)
        ]

        # 2K+4 Dimensional Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((2 * self.num_legit_users) + 4,), dtype=np.float32
        )

        # 5-D Action Space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
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
        gu_pos = np.concatenate([gu.position[:2] for gu in self.legit_users])
        uav_energy = np.array([self.uavs[0].energy], dtype=np.float32)
        gu_centroid = np.mean([gu.position for gu in self.legit_users], axis=0)
        return np.concatenate([uav_pos, gu_centroid]).astype(np.float32)

    def compute_awgn(self):
        return np.random.normal(0, 1)

    def compute_snr(self, tx_power, noise_power):
        return 10 * np.log10(tx_power / noise_power**2 + 1e-9)

    # TODO: COMPUTE SUM RATE HERE 
    def compute_sum_rate(self, subchan_bw, snr):
        sum_rate_k = subchan_bw * np.log2(1 + snr)
        return sum_rate_k

    def apply_power_allocation(self, scalar):
        return self.P_MAX * np.clip(scalar, 0.1, 1.0)

    def apply_noma_grouping(self, action_scalar):
        # Example: 0.25 → Group 0, 0.75 → Group 3
        group_id = int(np.clip(action_scalar * self.num_legit_users, 0, self.num_legit_users - 1))
        for i, gu in enumerate(self.legit_users):
            gu.cluster_id = group_id

    def _compute_energy_efficiency(self, sum_rate_arr, energy_cons):
        return sum_rate / (energy_cons)

    # TODO: INCLUDE SELF-LINK TOPOLOGY DICTIONARY

    # TODO: STEP FUNCTION
    def step(self, action):
        action = np.clip(action, -1, 1)
        
        for i, uav in enumerate(self.uavs):
            gu_positions = np.array([gu.position for gu in self.legit_users])
            gu_centroid = np.mean(gu_positions, axis=0)
            dist_to_centroid = np.linalg.norm(uav.position - centroid)
            # Only slow down speed when reasonably close to the GU centroid
            if dist_to_centroid <= 25:
                zeta = self.compute_zeta(dist_to_centroid)
            else:
                zeta = 1
            v = uav.compute_velocity(zeta)
            #delta = action[i*3:(i+1)*3] * v
            delta = action[:3] * v
            uav.move(delta)

            # TODO: ADD SUBCHANNEL BWS TO ARRAY HERE FOR UAVs & GUs
            # PASS SUBCHANNEL BWS TO COMPUTE_SUM_RATE
            # PASS RESULTS FROM THIS TO COMPUTE ENERGY EFFICIENCY FUNCTION
            # CALL COMPUTE REWARD FUNCTION HERE TO DO THIS
            # CALCULATE REWARDS BASED ON ENERGY EFFICIENCY SUCH THAT SUM RATE IS MORE THAN THE MINIMUM DESIRABLE SUM RATE (ONLY ALLOCATE IF R_sum > R_min)

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

    # TODO: IMPLEMENT MASR COMPUTATION IN HERE FOR ENERGY EFFICIENCY COMPUTATION
    '''
    def _compute_energy_efficiency(self, masr, energy_cons):
        energy_eff = masr / energy_cons
        return energy_eff 
    '''

    # TODO: REWARD FUNCTION
    # Function is incomplete and cannot work without MASR computation
    # Reward shaping function should factor in the following:
    # Data transmission/secrecy rate
    # Energy efficiency
    # Distance to GU centroid for clustering/grouping of GUs by the UAV-BS
    def _compute_reward(self):
        bs = self.uavs[0]
        reward = 0
        noise = self._compute_awgn()
        snr_legit = self.compute_snr(bs.tx_power, noise)
        gu_positions = np.array([gu.position for gu in self.legit_users])
        centroid = np.mean(gu_positions, axis=0)
        distance_to_centroid = np.linalg.norm(bs.position - centroid)
        energy_consumption = bs.compute_energy_consumption()
        bs.prev_energy_consumption = energy_consumption
        energy_eff = self._compute_energy_efficiency(masr, energy_consumption) 
        reward += energy_eff 

        return reward

    # TODO: CONSTRAINTS VIOLATIONS FUNCTION
    def check_constraints(self):

        violations = {
            "range": False,
            "altitude": False,
            "energy": False,
            "velocity": False
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

        return constraints

    # TODO: RENDER FUNCTION (KEEP AS EMPTY FOR NOW AS RENDERING TO BE ADDED IN LQDRL NOTEBOOK)
    def render(self):
        pass
