# uav_secrecy_env.py
from rician_channel import rician_fading_params, rician_channel_gain # Importing rician_channel.py module
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# === Base UAV Class ===
class UAV:
    # TODO: ADD MAXIMUM ENERGY CONSUMPTION, TRANSMIT POWER CONSUMPTION VALUES, ETC.
    def __init__(self, uav_id, position, velocity, tx_power, energy, links, mass):
        # UAV Characteristics & Parameters
        self.id = uav_id
        self.position = np.array(position, dtype=np.float32)
        self.velocity = velocity
        self.tx_power = tx_power
        self.energy = energy
        self.history = [self.position.copy()]
        # TODO: ADD COMMUNICATION LINKS TO/FROM THE UAV
        self.links = np.array(links)
        # TODO: POSSIBLE NEED TO SPLIT UP UAV FRAME & BATTERY MASS VALUES FOR n_sum IN FUTURE
        self.mass = mass
        # TODO: ADD SUM RATE PARAMETER
        # self.sum_rate = sum_rate

    def move(self, delta_pos):
        self.position += delta_pos
        self.velocity = np.linalg.norm(delta_pos)
        self.history.append(self.position.copy())

    def get_distance_travelled(self):
        if len(self.history) < 2:
            return 0
        diff = self.history[-1] - self.history[-2]
        return np.linalg.norm(diff)

    # TODO: Try different numbers of rotors (4 & 8)
    # NB: Lift-to-Drag ratio k based on rough calculations and can be adjusted if needs be
    # Ideally, it'd be better to compute the L/D ratio in another helper function but for now, use the default "by hand" calculated value
    def compute_energy_consumption(self, g=9.81, k=6.65, num_rotors=4, rho=1.225, theta=0.3, Lambda=5, num_uavs=3, mass=2000):
        c_t = self.get_distance_travelled()
        # n_sum denotes the mass of the UAV frame & battery
        n_sum = 0
        for i in range(num_uavs):
            n_sum += mass
        term1 = (n_sum * g * c_t) / (k * num_rotors)                            # Travelling Energy Consumption
        term2 = ((n_sum * g) ** 1.5) / np.sqrt(2 * num_rotors * rho * theta)    # Hovering Energy Consumption
        term3 = Lambda * c_t / (self.velocity + 1e-6)                           # Avionics Energy Consumption
        # Comms power use (Tx power * placeholder sum rate)
        R_kn = 1.0  # Placeholder sum rate per GU served
        term4 = self.tx_power * R_kn                                            # Communication Energy Consumption
        return term1 + term2 + term3 + term4


# === Subclasses ===
# TODO: ADD JAMMED LINKS (UAV-UAV Links & UAV-GU Links)
class UAVJammer(UAV):
    def __init__(self, *args, noise_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_power = noise_power

# TODO: ADD LINKS WITH OTHER UAVs (UAV-UAV Communications Only)
class UAVRelay(UAV):
    def __init__(self, *args, links=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.links = links or []

# TODO: ADD LINKS BETWEEN GUs & UAVs (UAV-UAV Links & UAV-GU Links)
class UAVBaseStation(UAV):
    def __init__(self, *args, coverage_radius=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_radius = coverage_radius
        self.ground_users = []

    def assign_gu(self, gu):
        self.ground_users.append(gu)

# === Ground User Class ===
class GroundUser:
    def __init__(self, gu_id, position, cluster_id):
        self.id = gu_id
        self.position = np.array(position)
        self.cluster_id = cluster_id
        self.subchannel = None
        self.channel_gain = 1.0  # Placeholder

class LegitGU(GroundUser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Maybe add "is_assigned" binary variable here

class EveGU(GroundUser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# === Custom Gym Environment ===
class UAVSecrecyEnv(gym.Env):
    def __init__(self):
        super(UAVSecrecyEnv, self).__init__()

        self.num_uavs = 3
        self.num_gus = 5

        # Constraints
        self.P_MAX = 30     # 30W maximum wattage
        self.E_MAX = 1000   # 1kJ maximum energy
        self.R_MIN = 0.75   # Minimum secrecy key rate exchange
        self.V_MAX = 20     # Maximum UAV Velocity of 20 m/s
        self.xmin = 0       # 0m minimum range
        self.ymin = 0       # 0m minimum range
        self.zmin = 10      # 10m minimum altitude
        self.xmax = 1500    # 15km radial range (diameter from any UAV position maximum range is 30km)
        self.ymax = 1500    # 15km radial range (diameter from any UAV position maximum range is 30km)
        self.zmax = 122     # 122m maximum altitude

        # Penalty weights
        # Equally-weighted penalty values for now
        self.pwr_penalty = 10
        self.alt_penalty = 10
        self.range_penalty = 10
        self.min_rate_penalty = 10
        self.energy_penalty = 10
        self.velocity_penalty = 10

        # TODO: ADD MORE UAVS HERE FOR DIFFERENT SCENARIOS
        self.uavs = [
            UAVBaseStation(0, [0, 0, 100], 0, 10, 1000, 5, 2000),
            UAVRelay(1, [50, 50, 100], 0, 10, 1000, 2, 2000),
            UAVJammer(2, [100, 100, 100], 0, 10, 1000, 5, 2000, noise_power=2.0)
        ]

        # Number of Randomly Distributed GUs
        #self.gus = [GroundUser(i, [np.random.uniform(0, 200), np.random.uniform(0, 200)], 0) for i in range(self.num_gus)]
        self.gus = [GroundUser(i, [np.random.uniform(self.xmin, self.xmax), np.random.uniform(self.ymin, self.ymax), 0], 0) for i in range(self.num_gus)]

        # Observation: concatenated UAV positions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_uavs * 3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_uavs * 3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        for i, uav in enumerate(self.uavs):
            uav.position = np.random.uniform(0, 200, size=(3,))
            uav.energy = 1000
            uav.history = [uav.position.copy()]

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([uav.position for uav in self.uavs]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        for i, uav in enumerate(self.uavs):
            delta = action[i*3:(i+1)*3] * 5.0
            uav.move(delta)
            energy_used = uav.compute_energy_consumption()
            uav.energy -= energy_used

        reward = self._compute_reward()
        done = any(uav.energy <= 0 for uav in self.uavs)

        violations = self.check_constraints()
        penalty = 0
        # TODO: DETERMINE WHAT SCORES SHOULD BE USED FOR PARTICULAR PENALTIES
        if violations["power"]:
            penalty += self.pwr_penalty
        elif violations["altitude"]:
            penalty += self.alt_penalty
        elif violations["range"]:
            penalty += self.range_penalty
        elif violations["min_rate"]:
            penalty += self.min_rate_penalty
        elif violations["energy"]:
            penalty += self.energy_penalty
        elif violations["velocity"]:
            penalty += self.velocity_penalty

        reward -= penalty

        return self._get_obs(), reward, done, {}

    # TODO: WORK OUT HOW TO DO THIS
    def subcarrier_scheduling(self):
        #psuedocode/template
        if scheduled:
            alpha = 1 
        else:
            alpha = 0

        return self._get_obs(), alpha

    # TODO: COMPUTE AWGN
    def compute_awgn():
        awgn = 0

        return awgn

    # TODO: Get tx_power & noise_power from UAV channel model & AWGN model, respectively
    def compute_snr(self, noise):
        noise_power = noise**2
        snr = 10 * np.log10(self.tx_power / noise_power)
        return snr

    # TODO: COMPUTE MASR
    # Compute R_sec for UAV-GU comms based on Zhang et. al (2024) model
    # Compute worst-case R_sec value and find MASR by subtracting this from calculated R_sec
    # R_sec = alpha * ln(1+SINR) [could also use SNR/SINAD if it's better suited to the environment]
    def compute_masr(self, subcarr_sched_var, snr):
        masr = subcarr_sched_var * np.log2(1 + snr)
        return masr

    def _compute_reward(self):
        # Example: sum of distances to GUs from UAVBaseStation
        # TODO: ADD OTHER REWARD PARAMETERS FOR FINDING MORE OPTIMAL PARAMETERS
        # Must factor in power, secrecy, energy efficiency, etc.
        bs = self.uavs[0]
        energy_efficiency_arr = []
        #masr = self.sum_rate

        # TODO: CALL MASR FUNCTION HERE FOR REWARD COMPUTATION
        noise = compute_awgn()
        snr = compute_snr(noise)
        # TODO: DETERMINE DECISION FOR SUBCARRIER SCHEDULING VARIABLE ALPHA FOR MASR CALCULATION
        masr = self.compute_masr()

        reward = 0
        energy_eff_reward = 0
        for gu in self.gus:
            d = np.linalg.norm(bs.position - gu.position)
            if d < bs.coverage_radius:
                reward += 1.0 / (1 + d)

        i = 0
        for uav in self.uavs:
            # TODO: Compare previous energy efficiency calculations with current one
            # If there's an increase in energy efficiency (i.e., a decrease in energy consumption),
            # then increase the reward factor. 
            energy_efficiency = masr / compute_energy_consumption()
            energy_efficiency_arr.append(energy_efficiency) 
            i += 1
            if i > 1:
                if (energy_efficiency_arr[i] >= energy_efficiency_arr[i-1]):
                    reward += energy_eff_reward
                else:
                    continue
            else:
                continue

            # TODO: Increase reward for increase in the maximum of the minimum average secrecy rate

        return reward

    # Line-of-Sight Signals
    # Provides values for Rician channel fading based on signal parameters over OFDM/NOMA
    def los_signals(C, r, w_c, t, f):
        s = C * (np.cos(w_c * t))
        for n in range(len(r)):
            u += r[n] * np.cos(w_c * t + f[n])
        v = s + u
        return v, s, u

    # Constraints Checker
    # Determines if the constraints have been violated for reward/penalty computation
    def check_constraints(self):
        violations = {
            "power": False,
            "altitude": False,
            "range": False,
            "min_rate": False,
            "energy": False,
            "velocity": False
        }

        # Power constraint
        total_tx_power = sum(uav.tx_power for uav in self.uavs)
        if total_tx_power > self.P_MAX:
            violations["power"] = True

        for uav in self.uavs:
            # Altitude & range constraints
            x, y, h = uav.position
            if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
                violations["range"] = True
            if not (self.zmin <= h <= self.zmax):
                violations["altitude"] = True

            # Energy consumption constraints
            if uav.energy <= 0:
                violations["energy"] = True
            if uav.energy >= self.E_MAX:
                violations["energy"] = True

            # UAV Velocity Constraint 
            if uav.velocity >= self.V_MAX:
                violations["velocity"] = True

        # Minimum rate per GU (placeholder — later use actual SINR)
        # TODO: FIX THIS FUNCTION AS IT'S INCORRECT
        for gu in self.gus:
            if gu.channel_gain < self.R_min:
                violations["min_rate"] = True
                break

        return violations


    # Compute K-factor using varying dominant LoS & non-dominant LoS signals
    # TODO: Make a separate function fo computing the PDF for the Rician channel model from this function
    # USE LoS FUNCTION FOR VALUES PASSED TO THIS FUNCTION (dominant & non-dominant signal power)
    def rician_fading(self, tx, rx, v, sigma):
        d = np.linalg.norm(tx - rx)
        K = (v**2) / (2*(sigma**2))
        Omega = v**2 + (2*sigma**2)
        
        return d, K, Omega

    # TODO: Finish this function
    # Probability Density Function for Rician Channel Model
    # Likely will need to call this function N times in main script for it to work and be able to plot it for a range of values of x
    def rician_pdf(x, K, Omega, I0):
        pdf = ((2 * (K + 1) * x) / (Omega)) * np.exp(-K - (((K + 1) * x**2) / Omega)) * I0 * (2 * np.sqrt((K * (K + 1)) / Omega) * x)
        return pdf

    def render(self, mode='human'):
        # TODO: Visualise the UAV & GU Clustering in 3-D Space
        # Possibly create a separate function for plotting convergence of the parameters to their optimal values over episodes
        pass  # Add matplotlib plot if desired
