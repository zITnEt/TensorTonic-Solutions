import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.array(points, dtype=float)
    T = np.array(T, dtype=float)

    if points.ndim==1:
        points = points[None, :]
    
    ones = np.ones((points.shape[0], 1))
    h = np.hstack([points, ones])
    r = (T@(h.T)).T
    return r[:, :3] if points.shape[0]>1 else r[0][:3]