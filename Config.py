class Config(object):

    # System setup
    N_UE = 20  # Number of Mobile Devices
    N_EDGE = 2  # Number of Edge Servers
    UE_COMP_CAP = 2.6  # Mobile Device Computation Capacity
    UE_TRAN_CAP = 14  # Mobile Device Transmission Capacity
    EDGE_COMP_CAP = 42  # Edge Servers Computation Capacity

    # Energy consumption settings
    UE_ENERGY_STATE = [
        0.25,
        0.50,
        0.75,
    ]  # Ultra-power-saving mode, Power-saving mode, Performance mode
    UE_COMP_ENERGY = 2  # Computation Power of Mobile Device
    UE_TRAN_ENERGY = 2.3  # Transmission Power of Mobile Device
    UE_IDLE_ENERGY = 0.1  # Standby power of Mobile Device
    EDGE_COMP_ENERGY = 5  # Computation Power of Edge Server

    # Task Requrement
    TASK_COMP_DENS = [0.197, 0.297, 0.397]  # Task Computation Density

    # TASK_COMP_DENS   = 0.297

    TASK_MIN_SIZE = 1
    TASK_MAX_SIZE = 7
    N_COMPONENT = 1  # Number of Task Partitions
    MAX_DELAY = 10

    # Simulation scenario
    N_EPISODE = 1000  # Number of Episodes
    N_TIME_SLOT = 100  # Number of Time Slots
    DURATION = 0.1  # Time Slot Duration
    TASK_ARRIVE_PROB = 0.3  # Task Generation Probability
    N_TIME = N_TIME_SLOT + MAX_DELAY

    # Algorithm settings
    LEARNING_RATE = 0.01
    REWARD_DECAY = 0.9
    E_GREEDY = 0.99
    N_NETWORK_UPDATE = 200  # Networks Parameter Replace
    MEMORY_SIZE = 500  # Replay Buffer Memory Size

    # --------- DDRL / Radio / Lyapunov params (ADD THESE) ---------
    # Subchannels, power discretization, units:
    NUM_SUBCHANNELS = 4  # k (choose small to control action size)
    BANDWIDTH = 1e6  # Hz per subchannel (or adjust to your units)
    NOISE_POWER = 1e-9  # Noise power (Watts). Choose appropriate value.
    P_MAX = 1.5  # Maximum transmit power (Watts)
    POWER_LEVELS = [0.0, 0.1, 0.3, 0.7, 1.5]  # discrete power grid (example)

    # Lyapunov and reward shaping:
    LYAPUNOV_V = 0.1  # V parameter in drift-plus-penalty

    # DQN hyperparams (paper recommended defaults)
    DDRL_LEARNING_RATE = 1e-3
    DDRL_BATCH_SIZE = 10
    DDRL_GAMMA = 0.7
    DDRL_MEMORY_SIZE = 400
    DDRL_TARGET_UPDATE = 10  # sync target every N train steps
    DDRL_EPISODES = 2000
    # ---------------------------------------------------------------
