import numpy as np

# Workspace limits from Task 9 (meters)
X_MIN, X_MAX = -0.187, 0.253
Y_MIN, Y_MAX = -0.1706, 0.2196
Z_MIN, Z_MAX = 0.1195, 0.2896

# Safe global Z floor (highest observed z_min across corners)
Z_SAFE_MIN = 0.1695


def sample_target(margin: float = 0.02, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample a random target inside a reachable OT-2 work envelope subset.

    Uses a conservative global Z floor (Z_SAFE_MIN) to avoid sampling
    unreachable low-Z targets at high X/Y.

    Args:
        margin: safety distance from the workspace limits (meters)
        rng: optional numpy RNG for reproducibility

    Returns:
        target [x, y, z] as float32
    """
    margin = float(margin)
    if margin < 0.0:
        raise ValueError(f"margin must be >= 0, got {margin}")

    if rng is None:
        rng = np.random.default_rng()

    # Compute valid ranges
    x_low, x_high = float(X_MIN + margin), float(X_MAX - margin)
    y_low, y_high = float(Y_MIN + margin), float(Y_MAX - margin)

    z_low = float(max(Z_SAFE_MIN + margin, Z_MIN + margin))
    z_high = float(Z_MAX - margin)

    # Range checks (fail fast with clear errors)
    if not (x_low < x_high):
        raise ValueError(f"Invalid X range after margin: [{x_low}, {x_high}]")
    if not (y_low < y_high):
        raise ValueError(f"Invalid Y range after margin: [{y_low}, {y_high}]")
    if not (z_low < z_high):
        raise ValueError(f"Invalid Z range after margin/safety floor: [{z_low}, {z_high}]")

    x = rng.uniform(x_low, x_high)
    y = rng.uniform(y_low, y_high)
    z = rng.uniform(z_low, z_high)

    return np.array([x, y, z], dtype=np.float32)
