import math
import queue
import random

import numpy as np

from Config import Config


class MEC:
    def __init__(self, num_ue, num_edge, num_time, num_component, max_delay):
        # Initialize variables
        self.n_ue = num_ue
        self.n_edge = num_edge
        self.n_time = num_time
        self.n_component = num_component
        self.max_delay = max_delay
        self.duration = Config.DURATION
        self.ue_p_comp = Config.UE_COMP_ENERGY
        self.ue_p_tran = Config.UE_TRAN_ENERGY
        self.ue_p_idle = Config.UE_IDLE_ENERGY
        self.edge_p_comp = Config.EDGE_COMP_ENERGY

        self.time_count = 0
        self.task_count_ue = 0
        self.task_count_edge = 0

        # DDRL action-size: 1 (local) + n_edge * k * n_power
        self.k = Config.NUM_SUBCHANNELS
        self.power_levels = Config.POWER_LEVELS
        self.n_power = len(self.power_levels)
        self.n_actions = 1 + self.n_edge * self.k * self.n_power

        self.n_features = 1 + 1 + 1 + 1 + self.n_edge
        self.n_lstm_state = self.n_edge

        # Channels: 3D array [time-slot handled per-step], but we store current channel gains:
        # channel gain magnitude squared per (ue, edge, subchannel)

        self.g = np.zeros(
            (self.n_ue, self.n_edge, self.k)
        )  # will be refreshed each step

        # noise and bandwidth references
        self.noise_power = Config.NOISE_POWER
        self.bw = Config.BANDWIDTH

        # queues (for Lyapunov)
        self.data_queue = np.zeros(self.n_ue)  # bits waiting to be served
        self.energy_queue = np.zeros(self.n_ue)  # virtual energy queue H_{l,m}

        # keep arrays for per-slot bookkeeping (for logging)
        self.ue_bits_transmitted = []  # append per-slot np.array(n_ue)
        self.ue_power_consumed = []  # append per-slot np.array(n_ue)

        self.drop_trans_count = 0
        self.drop_edge_count = 0
        self.drop_ue_count = 0

        # Computation and transmission capacities
        self.comp_cap_ue = Config.UE_COMP_CAP * np.ones(self.n_ue) * self.duration
        self.comp_cap_edge = (
            Config.EDGE_COMP_CAP * np.ones([self.n_edge]) * self.duration
        )
        self.tran_cap_ue = (
            Config.UE_TRAN_CAP * np.ones([self.n_ue, self.n_edge]) * self.duration
        )
        self.n_cycle = 1
        self.task_arrive_prob = Config.TASK_ARRIVE_PROB
        self.max_arrive_size = Config.TASK_MAX_SIZE
        self.min_arrive_size = Config.TASK_MIN_SIZE
        self.arrive_task_size_set = np.arange(
            self.min_arrive_size, self.max_arrive_size, 0.1
        )
        # self.energy_state_set   = np.arange(0.25,1, 0.25)
        self.ue_energy_state = [
            Config.UE_ENERGY_STATE[np.random.randint(0, len(Config.UE_ENERGY_STATE))]
            for ue in range(self.n_ue)
        ]
        self.arrive_task_size = np.zeros([self.n_time, self.n_ue])
        self.arrive_task_dens = np.zeros([self.n_time, self.n_ue])

        # print(self.energy_state_set)

        # print(self.ue_energy_state)

        # self.comp_density=0.297

        self.n_task = int(self.n_time * self.task_arrive_prob)

        # Task delay and energy-related arrays
        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])

        # Queue information initialization
        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])

        # Queue initialization
        self.ue_computation_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.ue_transmission_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.edge_computation_queue = [
            [queue.Queue() for _ in range(self.n_edge)] for _ in range(self.n_ue)
        ]
        self.edge_ue_m = np.zeros(self.n_edge)
        self.edge_ue_m_observe = np.zeros(self.n_edge)

        # Task indicator initialization
        self.local_process_task = [
            {
                "DIV": np.nan,
                "UE_ID": np.nan,
                "TASK_ID": np.nan,
                "SIZE": np.nan,
                "TIME": np.nan,
                "EDGE": np.nan,
                "REMAIN": np.nan,
            }
            for _ in range(self.n_ue)
        ]
        self.local_transmit_task = [
            {
                "DIV": np.nan,
                "UE_ID": np.nan,
                "TASK_ID": np.nan,
                "SIZE": np.nan,
                "TIME": np.nan,
                "EDGE": np.nan,
                "REMAIN": np.nan,
            }
            for _ in range(self.n_ue)
        ]
        self.edge_process_task = [
            [
                {
                    "DIV": np.nan,
                    "UE_ID": np.nan,
                    "TASK_ID": np.nan,
                    "SIZE": np.nan,
                    "TIME": np.nan,
                    "REMAIN": np.nan,
                }
                for _ in range(self.n_edge)
            ]
            for _ in range(self.n_ue)
        ]

        self.task_history = [[] for _ in range(self.n_ue)]

    def _sample_channels(self):
        # Simple Rayleigh power-fading model for each (ue,edge,subch)
        # You can replace with your pathloss + shadowing model if available.
        # We sample exponential (mean=1) for power gain.
        self.g = np.random.exponential(scale=1.0, size=(self.n_ue, self.n_edge, self.k))

    def _make_obs(self):
        # For minimal change: pack per-UE vector: [task_size, data_queue, energy_queue, avg_channel_gain_to_edges...]
        # If your existing state shape is different, append these fields.
        obs = []
        # compute per-UE average channel to each edge (reduce subchannels)
        avg_g = np.mean(self.g, axis=2)  # shape n_ue x n_edge
        for u in range(self.n_ue):
            # example task_size placeholder - adapt if you have arrivals stored elsewhere
            task_size = getattr(self, "current_task_size", np.zeros(self.n_ue))[
                u
            ]  # keep safe
            # flatten avg_g[u,:] to include edge info
            vec = [task_size, self.data_queue[u], self.energy_queue[u]]
            vec.extend(avg_g[u, :].tolist())
            obs.append(vec)
        # return as np.array (n_ue, features)
        return np.array(obs, dtype=np.float32)

    def reset(self, arrive_task_size, arrive_task_dens):

        self.drop_trans_count = 0
        self.drop_edge_count = 0
        self.drop_ue_count = 0

        # Reset variables and queues
        self.task_history = [[] for _ in range(self.n_ue)]
        self.UE_TASK = [-1] * self.n_ue
        self.drop_edge_count = 0

        self.arrive_task_size = arrive_task_size
        self.arrive_task_dens = arrive_task_dens

        self.time_count = 0

        self.local_process_task = []
        self.local_transmit_task = []
        self.edge_process_task = []

        self.ue_computation_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.ue_transmission_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.edge_computation_queue = [
            [queue.Queue() for _ in range(self.n_edge)] for _ in range(self.n_ue)
        ]

        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])

        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])

        self.local_process_task = [
            {
                "DIV": np.nan,
                "UE_ID": np.nan,
                "TASK_ID": np.nan,
                "SIZE": np.nan,
                "TIME": np.nan,
                "EDGE": np.nan,
                "REMAIN": np.nan,
            }
            for _ in range(self.n_ue)
        ]
        self.local_transmit_task = [
            {
                "DIV": np.nan,
                "UE_ID": np.nan,
                "TASK_ID": np.nan,
                "SIZE": np.nan,
                "TIME": np.nan,
                "EDGE": np.nan,
                "REMAIN": np.nan,
            }
            for _ in range(self.n_ue)
        ]
        self.edge_process_task = [
            [
                {
                    "DIV": np.nan,
                    "UE_ID": np.nan,
                    "TASK_ID": np.nan,
                    "SIZE": np.nan,
                    "TIME": np.nan,
                    "REMAIN": np.nan,
                }
                for _ in range(self.n_edge)
            ]
            for _ in range(self.n_ue)
        ]

        # Initialize DDRL queues and channels
        self.data_queue[:] = 0.0
        self.energy_queue[:] = 0.0
        self.ue_bits_transmitted = []
        self.ue_power_consumed = []
        self._sample_channels()

        # Initial observation and LSTM state
        UEs_OBS = np.zeros([self.n_ue, self.n_features])
        for ue_index in range(self.n_ue):
            if self.arrive_task_size[self.time_count, ue_index] != 0:
                UEs_OBS[ue_index, :] = np.hstack(
                    [
                        self.arrive_task_size[self.time_count, ue_index],
                        self.t_ue_comp[ue_index],
                        self.t_ue_tran[ue_index],
                        np.squeeze(self.b_edge_comp[ue_index, :]),
                        self.ue_energy_state[ue_index],
                    ]
                )

        UEs_lstm_state = np.zeros([self.n_ue, self.n_lstm_state])

        return UEs_OBS, UEs_lstm_state

    # perform action, observe state and delay (several steps later)
    def step(self, action_all):
        """
        action_all: array-like of length self.n_ue with integer actions chosen by per-UE DQNs
        returns: obs (n_ue x feat), reward (float or per-UE), done, info
        """
        from action_utils import decode_action

        # 0) sanity cast
        action_all = np.asarray(action_all, dtype=int).ravel()
        assert action_all.shape[0] == self.n_ue

        # 1) sample fresh channels for this slot
        self._sample_channels()

        # 2) decode actions into tuples and collect chosen (edge,subch) pairs
        decoded = [decode_action(a, self.n_edge) for a in action_all]

        # 3) Resolve subchannel conflicts per edge
        # Build mapping (edge,subch) -> list of UE indices choosing it
        usage = dict()
        for u, dec in enumerate(decoded):
            if dec["local"]:
                continue
            key = (dec["edge"], dec["subchannel"])
            usage.setdefault(key, []).append(u)

        # conflict resolution: if multiple UEs pick same (edge,subch), pick the one with highest p (power)
        conflicted = set()
        for key, users in usage.items():
            if len(users) <= 1:
                continue
            # choose best by power, rest are forced local (simple, deterministic)
            best = max(users, key=lambda uu: decoded[uu]["power"])
            for uu in users:
                if uu != best:
                    conflicted.add(uu)
                    decoded[uu] = {
                        "local": True,
                        "edge": None,
                        "subchannel": None,
                        "power": 0.0,
                        "p_idx": 0,
                    }

        # 4) Compute SINR and rate per UE (if offloading) on their chosen edge+subch
        slot_bits = np.zeros(self.n_ue)
        slot_power = np.zeros(self.n_ue)
        duration = getattr(
            self, "slot_duration", 1.0
        )  # seconds, adapt if you have a different variable

        # Precompute interfering power on each (edge,subch)
        # For each (edge, subch) compute total interference from other UEs choosing same subch but different edges might still interfere depending on your model.
        # Here we assume interference across same subchannel across edges may exist; adjust formula if your paper defines differently.
        # We'll compute per-(edge,subch) a list of users transmitting there.
        tx_on = dict()
        for u, dec in enumerate(decoded):
            if dec["local"]:
                continue
            key = (dec["edge"], dec["subchannel"])
            tx_on.setdefault(key, []).append(u)

        # compute SINR for each transmitting UE:
        for u, dec in enumerate(decoded):
            if dec["local"]:
                slot_bits[u] = 0.0
                slot_power[u] = 0.0
                continue
            e = dec["edge"]
            s = dec["subchannel"]
            p = dec["power"]  # in Watts
            # channel gain from UE u to edge e on subchannel s
            g_ues = float(self.g[u, e, s])  # scalar
            # numerator:
            num = p * g_ues
            # denominator: sum of other users on same (edge,subchannel) -> if your interference model is different, adjust.
            denom = self.noise_power
            # Add interference from other UEs that transmit on same subchannel (either on same or other edges) - simple model:
            for (ee, ss), users in tx_on.items():
                if ss != s:
                    continue
                for interferer in users:
                    if interferer == u:
                        continue
                    p_i = decoded[interferer]["power"]
                    # channel gain from interferer -> e (cross-link) ; using g[interferer, e, s] as cross coupling
                    h_iu = float(self.g[interferer, e, s])  # simplistic cross-channel
                    denom += p_i * h_iu

            sinr = num / (denom + 1e-12)
            rate_bps = self.bw * math.log2(1.0 + sinr)  # bits/sec
            bits_served = rate_bps * duration
            slot_bits[u] = bits_served
            slot_power[u] = p

        # 5) Update queues
        # A(t): arrivals for this slot (you already have an arrivals process - keep using it)
        arrivals = getattr(self, "current_arrivals", np.zeros(self.n_ue))
        # Update data queues Q(t+1) = max(Q(t) - b(t),0) + A(t)
        self.data_queue = np.maximum(self.data_queue - slot_bits, 0.0) + arrivals
        # Update energy queue H(t+1) = max(H(t) - P_used + P_T, 0)  <-- P_T any energy harvesting supply; adapt variable name
        P_T = getattr(self, "P_T", 0.0)
        self.energy_queue = np.maximum(self.energy_queue - slot_power + P_T, 0.0)

        # 6) Bookkeeping for logging
        self.ue_bits_transmitted.append(slot_bits.copy())
        self.ue_power_consumed.append(slot_power.copy())
        # optionally update cumulative metrics you already store (transmission energy, etc.)

        # 7) reward: we compute global per-slot DDRL reward inside environment for convenience (or compute in main)
        # compute energy-efficiency numerator and denominator:
        numerator = np.sum(slot_bits)  # bits served this slot
        # Use total instantaneous power consumption across UEs as denominator (plus tiny eps)
        denominator = np.sum(slot_power) + 1e-12
        ee = numerator / denominator if denominator > 0 else 0.0

        # compute Lyapunov drift: simple one-step approx: L(t+1)-L(t)
        L_prev = 0.5 * (
            np.sum((self.data_queue + slot_bits - arrivals) ** 2)
            + np.sum(self.energy_queue**2)
        )  # approximate
        L_curr = 0.5 * (np.sum(self.data_queue**2) + np.sum(self.energy_queue**2))
        drift = L_curr - L_prev

        reward = ee - Config.LYAPUNOV_V * drift

        # 8) Build next observation
        obs = self._make_obs()

        # 9) done flag and info
        done = False  # episodic logic handled externally
        info = {
            "ee": ee,
            "drift": drift,
            "conflicts": list(conflicted),
            "slot_bits": slot_bits,
            "slot_power": slot_power,
        }

        # return in same signature as your current env (adapt if necessary)
        return obs, reward, done, info
