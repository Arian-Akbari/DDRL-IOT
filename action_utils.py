# action_utils.py
from Config import Config


def encode_action(local: bool, edge: int = None, subch: int = None, p_idx: int = None):
    """
    Returns integer action encoding.
    local=True -> 0
    Otherwise: 1 + edge * (k * npw) + subch * npw + p_idx
    """
    if local:
        return 0
    k = Config.NUM_SUBCHANNELS
    npw = len(Config.POWER_LEVELS)
    return 1 + int(edge) * (k * npw) + int(subch) * npw + int(p_idx)


def decode_action(action_int: int, n_edge: int):
    """
    Returns dict: {'local':bool, 'edge':int, 'subchannel':int, 'power':float, 'p_idx':int}
    """
    if int(action_int) == 0:
        return {
            "local": True,
            "edge": None,
            "subchannel": None,
            "power": 0.0,
            "p_idx": 0,
        }
    k = Config.NUM_SUBCHANNELS
    power_levels = Config.POWER_LEVELS
    npw = len(power_levels)
    idx = int(action_int) - 1
    p_idx = idx % npw
    s_idx = (idx // npw) % k
    e_idx = idx // (npw * k)
    return {
        "local": False,
        "edge": int(e_idx),
        "subchannel": int(s_idx),
        "power": float(power_levels[p_idx]),
        "p_idx": int(p_idx),
    }
