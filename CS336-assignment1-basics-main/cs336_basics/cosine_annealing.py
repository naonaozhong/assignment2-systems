import math

def cosine_annealing(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int):
    assert t >= 0, "time should be nonnegative"
    if t < T_w:
        return t / T_w * lr_max
    elif t <= T_c:
        return lr_min + 1/2 * (1 + math.cos((t - T_w)/(T_c - T_w) * math.pi)) * (lr_max - lr_min)
    elif t > T_c:
        return lr_min
    else:
        raise ValueError