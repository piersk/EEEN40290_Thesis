# uav_lqdrl_env.py
# TODO: Merge problem from LQ-DRL problem & Secrecy Rate Problem
# Implementation of the proposed LQ-DRL algorithm presented in Silvirianti et al. (2025)
# All GU instantiation should just be handled by GroundUser parent class for this program
# Possibly remove the Relay & Jammer UAVs for this implementation/version

# TODO: ENSURE THE PENALTIES ARE SCALED SO THAT THEY ACTUALLY MAKE A DENT TO THE REWARDS
# TODO: ENSURE THE UAV CANNOT GO OUTSIDE OF THE BOUNDS SET BY x, y, z MAX AND MIN
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

    # Function to move UAV in 3-D Cartesian Space
    def move(self, delta_pos):
        self.position += delta_pos
        if (self.position[2] <= 0):
            self.position[2] = 0
        self.velocity = np.linalg.norm(delta_pos)
        self.history.append(self.position.copy())

    def get_distance_travelled(self):
        if len(self.history) < 2:
            return 0
        return np.linalg.norm(self.history[-1] - self.history[-2])

    # 100 J/s avionics power from d'Andrea et al (2014)
    #def compute_energy_consumption(self, num_uavs, sum_rate_arr, g=9.81, k=6.65, num_rotors=4, rho=1.225, theta=0.0507, Lambda=100):
    # TODO: NORMALISE THE SUM RATE & CONVERT POWER TO LINEAR FROM dBm
    def compute_energy_consumption(self, tx_power_arr, sum_rate_arr, g=9.81, k=6.65, num_rotors=4, rho=1.225, theta=0.0507, Lambda=100):
        c_t = self.get_distance_travelled()
        c_t = abs(c_t)
        n_sum = self.mass
        travel = (n_sum * g * c_t) / (k * num_rotors)
        hover = ((n_sum * g) ** 1.5) / np.sqrt(2 * num_rotors * rho * theta)
        #term3 = Lambda * c_t / (self.velocity + 1e-6)
        avionics = Lambda * c_t / (self.velocity)
        # TODO: REPLACE R_kn CONSTANT WITH COMPUTED SUM RATE FOR UAV-GU LINKS
        #R_kn = 1.0
        # TODO: MAY REQUIRE A 2-D ARRAY FOR SUM RATES IN FUTURE FOR MULTIPLE UAV-BS AGENTS IN FUTURE
        comms = 0
        for k in range(len(sum_rate_arr)):
            #tx_power = 10**(tx_power_arr[k]/10)/1000
            comms += tx_power * sum_rate_arr[k]
            #comms += tx_power_arr[k] * sum_rate_arr[k]
        #comms = self.tx_power * R_kn
        energy_cons = travel + hover + avionics + comms
        return energy_cons

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
        self.time = 0
        self.delta_t = 10
        self.num_uavs = 1
        self.num_legit_users = 4

        #self.P_MAX = self.dbm_to_watt(30)
        self.P_MAX = 30
        #self.E_MAX = 500e3  # 500kJ in paper
        self.E_MAX = 50e03 # Setting to 50kJ to speed up experiments for now
        #self.R_MIN = 0.75
        self.R_MIN = 1e06
        self.V_MAX = 50     # 50 m/s in paper
        self.xmin, self.ymin, self.zmin = 0, 0, 10 
        self.xmax, self.ymax, self.zmax = 150, 150, 122
        # Penalty values are for scaling the reward based on constraint violations
        self.pwr_penalty = self.alt_penalty = self.range_penalty = \
        self.min_rate_penalty = self.energy_penalty = self.velocity_penalty = 0.15

        # TODO: DETERMINE APPROPRIATE CARRIER FREQUENCY
        # USE 1MHz PLACEHOLDER FOR NOW
        self.f_carr = 1e06
        
        self.uavs = [
            #UAVBaseStation(0, [0, 0, 0], 0, 10, 1000, num_links=4, mass=2000)
            UAVBaseStation(0, [0, 0, 0], 0, self.P_MAX, self.E_MAX, num_links=self.num_legit_users, mass=1.46)
        ]

        self.legit_users = [
            LegitimateUser(i, [np.random.uniform(self.xmin, self.xmax), 
                               np.random.uniform(self.ymin, self.ymax), 
                               np.random.uniform(0, 0)], 
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
            #uav.position = np.random.uniform([0, 0, 10], [200, 200, 100])
            uav.position = np.random.uniform([self.xmin, self.ymin, self.zmin], [self.xmax, self.ymax, self.zmax])
            uav.energy = self.E_MAX
            uav.history = [uav.position.copy()]
            uav.prev_distance_to_centroid = None
        return self._get_obs(), {}

    # Function returns the observation state space of the environment (s)
    def _get_obs(self):
        uav_pos = np.concatenate([uav.position for uav in self.uavs])
        gu_pos = np.concatenate([gu.position[:2] for gu in self.legit_users])
        uav_energy = np.array([self.uavs[0].energy], dtype=np.float32)
        #gu_centroid = np.mean([gu.position for gu in self.legit_users], axis=0)
        return np.concatenate([uav_pos, gu_pos, uav_energy]).astype(np.float32)

    def dbm_to_watt(dbm):
        return (10 ** ((dbm) / 10)) / 1000

    def compute_zeta(self, dist_to_centroid):
        zeta = 1 - ((self.xmax - dist_to_centroid) / (self.xmax - self.xmin))
        return zeta

    # Function to compute the velocity of the UAV for any timestep t
    # Zeta must be computed as a variable between 0 and 1 to scale against V_MAX
    def compute_velocity(self, zeta):
        v = zeta * self.V_MAX
        return v

    # TODO: COMPUTE POLAR ANGLE CONSANTS FOR UAV TO MOVE UP, DOWN, LEFT & RIGHT 
    # Must determine angle between UAV trajectory vector and GU centroid in z-axis and xy-plane 
    def compute_polar_angles(self):
        zeta_p = 0
        zeta_a = 0
        return zeta_p, zeta_a

    def get_uav_position(self):
        for uav in self.uavs:
            position = uav.position
        return position

    def get_remaining_energy(self):
        for uav in self.uavs:
            e_remain = uav.energy
        return e_remain 

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

    # TODO: FIX COMMUNICATION MODEL
    def compute_awgn(self):
        return np.random.normal(0, 1)

    def compute_snr(self, tx_power, channel_gain, noise_power):
        #return 10 * np.log10(tx_power / noise_power**2 + 1e-9)
        snr = (tx_power * abs((channel_gain**2))) / (noise_power**2)
        return snr

    # TODO: COMPUTE SUM RATE HERE 
    def compute_r_k(self, subchan_bw, snr):
        sum_rate_k = subchan_bw * np.log2(1 + snr)
        return sum_rate_k

    # TODO: COMPUTE POWER COEFFICIENT BASED ON CHANNEL GAIN
    # RETURN 0 < DELTA <= 1 SCALED BASED ON NUMBER OF GUs & THEIR CHANNEL GAINS
    def compute_power_coefficients(self, channel_gain, channel_gain_arr):
        # Set minimum channel gain to 0.01
        h_tot = 0
        for n in range(len(channel_gain_arr)):
            h_tot += abs(channel_gain_arr[n])
        #h_tot = np.sum(channel_gain_arr)
        hnorm = abs(channel_gain) / h_tot
        h_prop = 1 - hnorm
        #delta = (1 / (self.num_legit_users - 1)) * h_prop
        delta = (1 / (self.num_legit_users)) * h_prop
        #hnorm = (channel_gain - h_min) / (h_max - h_min)
        # Possible idea for scaling Tx power but for now will use the hnorm above
        #hnorm /= len(channel_gain_arr)
        #h_inv = 1/hnorm
        #delta = 1 - abs(hnorm)
        return delta

    # TODO: UPDATE THIS ONCE POWER COEFFICIENT IS COMPUTED PROPERLY
    # SCALAR MUST BE COMPUTED BASED ON GU CHANNEL GAIN (LARGER SCALAR FOR WEAKER GAIN)
    def compute_power_allocation(self, pwr_coeff):
        # TODO: UPDATE
        power_alloc = self.P_MAX * pwr_coeff
        return power_alloc
        #return self.P_MAX * np.clip(pwr_coeff, 0.1, 1.0)

    def apply_noma_grouping(self, action_scalar):
        # Example: 0.25 → Group 0, 0.75 → Group 3
        group_id = int(np.clip(action_scalar * self.num_legit_users, 0, self.num_legit_users - 1))
        for i, gu in enumerate(self.legit_users):
            gu.cluster_id = group_id

    # TODO: COMPUTE THE FREE SPACE PATHLOSS AS IN PAPER HERE (SECTION V OF SILVIRIANTI)
    # Takes distance between UAV-BS & GU, carrier frequency & speed of light as args
    # CAN EITHER COMPUTE FOR ALL UAVs & GUs IN FUNCTION OR CALL MULTIPLE TIMES IN A LOOP
    # WILL LIKELY GO FOR LATTER AS step() ITERATES OVER UAVs AND _compute_reward() WILL
    def compute_pathloss(self, f_carrier, uav_pos, gu_pos):
        dist = np.linalg.norm(uav_pos - gu_pos) 
        fs_ploss = 20 * np.log10(dist) + 20 * np.log10(f_carrier) + 20 * np.log10((4 * np.pi) / 2.99e08)
        return fs_ploss

    # TODO: COMPUTE CHANNEL GAIN USING PATHLOSS & AWGN
    def compute_channel_gain(self, pathloss, awgn):
        h = pathloss * awgn
        return h

    def compute_subcarrier_allocation(self, f_carrier):
        i = 0
        bw_arr = [0 for i in range(self.num_legit_users)]
        for gus in self.legit_users:
            #bw_arr[i] = (((i + 1) * f_carrier) - (i * f_carrier))
            bw_arr[i] = f_carrier / self.num_legit_users
            i += 1
        return bw_arr

    def compute_sum_rate(self, bw_arr, snr):
        sum_rate_arr = []
        for k in range(self.num_legit_users):
            r_k = bw_arr[k] * np.log2(1 + abs(snr))
            sum_rate_arr.append(r_k)
        return sum_rate_arr

    def _compute_energy_efficiency(self, sum_rate, energy_cons):
        return sum_rate / (energy_cons)

    def _compute_gu_centroid(self, gu_positions):
        #gu_positions = np.array([gu.position for gu in self.legit_users])
        gu_centroid = np.mean(gu_positions, axis=0)
        return gu_centroid 

    # TODO: INCLUDE SELF-LINK TOPOLOGY DICTIONARY

    # TODO: STEP FUNCTION
    # TODO: IMPLEMENT MORE CONTROLLED MOVEMENT (POLAR CO-ORDINATES AS WRITTEN IN PAPER
    # Function to compute the action (a) taken by the UAV agent(s) based on state (s)
    # TODO: COMPUTE THE SIGNAL TRANSMIT POWER PROPERLY
    # ONLY HANDLE COMPUTATION AS NEEDED INSTEAD OF BOTH IN step() AND _compute_reward()
    def step(self, action):
        action = np.clip(action, -1, 1)
        energy_cons_penalty = 0
        reward_boost = 0
        for i, uav in enumerate(self.uavs):
            gu_positions = np.array([gu.position for gu in self.legit_users])
            gu_centroid = self._compute_gu_centroid(gu_positions)
            #gu_centroid = np.mean(gu_positions, axis=0)
            dist_to_centroid = np.linalg.norm(uav.position - gu_centroid)
            # TODO: IMPLEMENT BETTER STEERING IN Z-AXIS & XY-PLANE
            # Only slow down speed when reasonably close to the GU centroid
            if dist_to_centroid <= 25:
                zeta = self.compute_zeta(dist_to_centroid)
            else:
                zeta = 1
            dist = self.compute_velocity(zeta) * self.delta_t
            #delta = action[i*3:(i+1)*3] * v
            delta = action[:3] * dist
            uav.move(delta)

            # TODO: ADD SUBCHANNEL BWS TO ARRAY HERE FOR UAVs & GUs
            # NB: THIS WAS DONE IN _compute_reward()

            # PASS SUBCHANNEL BWS TO COMPUTE_SUM_RATE
            # PASS RESULTS FROM THIS TO COMPUTE ENERGY EFFICIENCY FUNCTION
            # CALL COMPUTE REWARD FUNCTION HERE TO DO THIS
            # CALCULATE REWARDS BASED ON ENERGY EFFICIENCY SUCH THAT SUM RATE IS MORE THAN THE MINIMUM DESIRABLE SUM RATE (ONLY ALLOCATE IF R_sum > R_min)
            #noise = self.compute_awgn()
            channel_gain_arr = []
            pwr_delta_arr = []
            awgn_arr = []
            i = 0
            for gu in self.legit_users:
                uav_pos = uav.position
                gu_pos = gu.position
                dist_from_gu = np.linalg.norm(uav_pos - gu_pos)
                pathloss = self.compute_pathloss(self.f_carr, uav_pos, gu_pos)
                awgn = self.compute_awgn()
                # Ensure the AWGN value is positive
                awgn = np.sqrt(awgn**2)
                awgn_arr.append(awgn)
                channel_gain = abs(self.compute_channel_gain(pathloss, awgn))
                i += 1
                print(f"GU {i} Channel Gain: ", channel_gain)
                channel_gain_arr.append(channel_gain)
                print(f"Distance between UAV & GU {i}: ", dist_from_gu, " m")

            tx_power_arr = []
            snr_arr = []
            bw_arr = self.compute_subcarrier_allocation(self.f_carr)
            sum_rate_arr = []
            # TODO: COMPUTE ALL TX_POWER COEFFICIENTS IN ONE GO AND SORT AS AN ARRAY
            # SORT GAIN ARRAY IN ASCENDING ORDER AND REVERSE THIS LIST TO USE IT AS DELTA_ARR
            for k, gu in enumerate(self.legit_users):
                print("=================================")
                gain = channel_gain_arr[k]
                pwr_delta = self.compute_power_coefficients(gain, channel_gain_arr)
                print(f"Power scaling variable {k}: ", pwr_delta)
                tx_power = self.compute_power_allocation(pwr_delta)
                print(f"Transmit Power {k}: ", tx_power)
                tx_power_arr.append(tx_power)
                noise_kn = awgn_arr[k]
                print(f"AWGN {k}: ", noise_kn)
                snr_legit = self.compute_snr(tx_power, gain, noise_kn)
                print(f"SNR {k}: ", snr_legit)
                print(f"SNR {k}: ", 20 * np.log10(snr_legit), "dB")
                snr_arr.append(snr_legit)
                bw_subchan = bw_arr[k]
                print(f"Subchannel Bandwidth {k}: ", bw_subchan, "Hz")
                r_kn = self.compute_r_k(bw_subchan, snr_legit)
                print(f"Data rate {k}: ", r_kn)
                sum_rate_arr.append(r_kn)

            #sum_rate_arr = self.compute_sum_rate(bw_arr, snr_legit)
        # Energy consumption should only occur once per step for 1 UAV and once per UAV per step if multiple UAV-BSs are to be used
        sum_rate_hz_arr = []
        for i in range(len(sum_rate_arr)):
            sum_rate_hz = (sum_rate_arr[i] / self.f_carr)
            sum_rate_hz_arr.append(sum_rate_hz)

        uav_energy_cons = uav.compute_energy_consumption(tx_power_arr, sum_rate_hz_arr)
        print("UAV Energy Consumption: ", uav_energy_cons)
        uav.energy -= uav_energy_cons

        if uav_energy_cons > uav.prev_energy_consumption:
            energy_cons_penalty += 0.1
        else:
            energy_cons_penalty = 0

        if dist_to_centroid <= 30 and dist_to_centroid >= self.zmax:
            reward_boost += 0.2

        r_sum = np.sum(sum_rate_arr, axis=0)
        reward = self._compute_reward(sum_rate_arr, uav_energy_cons)
        reward += reward_boost * reward
        done = any(uav.energy <= 0 for uav in self.uavs)
        penalties = self.check_constraints()
        total_penalty = sum(v * p for v, p in zip(penalties.values(), [
            self.pwr_penalty, self.alt_penalty, self.range_penalty,
            self.min_rate_penalty, self.energy_penalty, self.velocity_penalty
        ]))
        reward -= reward * (total_penalty + energy_cons_penalty)
        
        return self._get_obs(), reward, done, False, {}

    # TODO: REWARD FUNCTION
    # Function is incomplete and cannot work without MASR computation
    # Reward shaping function should factor in the following:
    # Data transmission/secrecy rate
    # Energy efficiency
    # Distance to GU centroid for clustering/grouping of GUs by the UAV-BS
    # Function to compute the reward allocated based on action a relative to a policy pi for a given state s
    # TODO: REWRITE FUNCTION TO ONLY COMPUTE REWARD AS EVERYTHING ELSE HANDLED IN step()
    # TODO: POSSIBLY ALLOCATE SOME REWARDS FOR PROXIMITY TO GU CENTROID
    def _compute_reward(self, sum_rate_arr, energy_consumption):
        # Compute reward for the UAV agent
        reward = 0
        grant_reward = False
        masr = np.sum(sum_rate_arr, axis=0)
        energy_eff = self._compute_energy_efficiency(masr, energy_consumption)
        print("Energy Efficiency: ", energy_eff)
        j = 0
        for k in range(0, self.num_legit_users):
            sum_rate = sum_rate_arr[k]
            print(f"Sum Rate {k}: ", sum_rate)
            if sum_rate > self.R_MIN:
                j += 1
                if k == (self.num_legit_users - 1) and k == (j - 1):
                    grant_reward = True

        print("Reward Allocated: ", grant_reward)

        if grant_reward == True:
            reward = energy_eff

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

        return violations

    # TODO: RENDER FUNCTION (KEEP AS EMPTY FOR NOW AS RENDERING TO BE ADDED IN LQDRL NOTEBOOK)
    def render(self):
        pass
