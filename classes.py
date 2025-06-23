# ===================== Base Entities =====================

class UAV:
    def __init__(self, id, position, tx_power, energy):
        self.id = id
        self.position = position
        self.tx_power = tx_power
        self.energy = energy
        self.los_metric = None  # update based on scenario

    def move(self, new_position):
        # Update position & energy
        pass

    def compute_path_loss(self, receiver_pos):
        pass


class GroundUser:
    def __init__(self, id, position, cluster_id):
        self.id = id
        self.position = position
        self.cluster_id = cluster_id
        self.subchannel = None
        self.channel_gain = None


# ===================== UAV Roles =====================

class UAVJammer(UAV):
    def __init__(self, *args, noise_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_power = noise_power
        self.jammed_links = set()

    def jam(self, link_id):
        self.jammed_links.add(link_id)


class UAVRelay(UAV):
    def __init__(self, *args, links=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.links = links if links else []

    def forward(self, data):
        # Forward from one UAV or GU to another
        pass


class UAVBaseStation(UAV):
    def __init__(self, *args, coverage_radius=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_radius = coverage_radius
        self.ground_users = []

    def assign_gu(self, gu: GroundUser):
        self.ground_users.append(gu)

    def compute_secrecy_rate(self, gu, eavesdropper):
        # Implement SR = C_main - C_eve
        pass
