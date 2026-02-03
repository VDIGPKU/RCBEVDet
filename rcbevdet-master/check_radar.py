import numpy as np
import os

radar_path = "/mnt/datasets/V2X-Radar-I/V2X-Radar-I/training/radar/000000.bin"
data = np.fromfile(radar_path, dtype=np.float32)
print(f"Total floats: {len(data)}")

# Try common feature sizes
for n in range(5, 10):
    if len(data) % n == 0:
        print(f"Possible shape if {n} features: ({len(data)//n}, {n})")

# Assume 5 if possible (x, y, z, rcs, v)
if len(data) % 5 == 0:
    points = data.reshape(-1, 5)
    print("First 5 points (assuming 5 features):")
    print(points[:5])

# Assume 7 if possible (x, y, z, v_r, v_r_comp, rcs, ...)
if len(data) % 7 == 0:
    points = data.reshape(-1, 7)
    print("First 5 points (assuming 7 features):")
    print(points[:5])
