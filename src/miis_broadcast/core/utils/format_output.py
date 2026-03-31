def convert_idx_to_msecs(idx: int, fps: float) -> float:
    if fps > 0:
        return float(idx) * 1000.0 / fps + 1e-8
    else:
        return 0.0
