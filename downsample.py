import numpy as np

def downsample(u, v, target_length=24):
    # Calculate cumulative distance
    delta_u = np.diff(u)
    delta_v = np.diff(v)
    distances = np.sqrt(delta_u**2 + delta_v**2)
    cum_distances = np.insert(np.cumsum(distances), 0, 0)

    # Create target distances for interpolation
    total_distance = cum_distances[-1]
    target_distances = np.linspace(0, total_distance, target_length)

    # Interpolate the u and v values
    u_new = np.interp(target_distances, cum_distances, u)
    v_new = np.interp(target_distances, cum_distances, v)
    
    return u_new, v_new

