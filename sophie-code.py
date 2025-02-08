##heatmap
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Whether to use with_gaps folder (True) or without_gaps (False) (from output)
gaps = True

# Start time (in frames) - set to 0 to start from beginning
t_s = 0

# End time (in frames) - set to 't_f = None' if no specific end time
t_f = 300

# Frames per second
fps = 30

# Define the base directory
base_dir = "trajectories\dead_flies_trajectories.csv"

# Load the data
try:
    test_trajectories = np.loadtxt(base_dir, delimiter=",", skiprows=1)
except Exception as e:
    raise Exception(f"Error loading data from {base_dir}: {e}")

t_max = len(test_trajectories)

x_min = np.min(test_trajectories[:, ::2][np.isfinite(test_trajectories[:, ::2])])
x_max = np.max(test_trajectories[:, ::2][np.isfinite(test_trajectories[:, ::2])])
y_max = -np.min(test_trajectories[:, 1::2][np.isfinite(test_trajectories[:, 1::2])])
y_min = -np.max(test_trajectories[:, 1::2][np.isfinite(test_trajectories[:, 1::2])])

border = 0.05

if t_f is None:
    t_f = t_max
elif t_f > t_max:
    raise Exception(f"t_f must be smaller or equal to max no. frames ({t_max})")

# Define grid resolution for the heatmap
grid_resolution = 60  # Increase this for a finer heatmap, decrease for more blurred
x_bins = np.linspace(
    x_min - border * abs(x_max - x_min),
    x_max + border * abs(x_max - x_min),
    grid_resolution,
)
y_bins = np.linspace(
    y_min - border * abs(y_max - y_min),
    y_max + border * abs(y_max - y_min),
    grid_resolution,
)

# Iniialize an empty grid for density accumulation
heatmap, _, _ = np.histogram2d([], [], bins=[x_bins, y_bins])

# Accumulate density in the grid
for i in range(0, len(test_trajectories[0]), 2):
    x = test_trajectories[t_s:t_f, i][np.isfinite(test_trajectories[t_s:t_f, i])]
    y = -test_trajectories[t_s:t_f, i + 1][
        np.isfinite(test_trajectories[t_s:t_f, i + 1])
    ]
    heatmap += np.histogram2d(x, y, bins=[x_bins, y_bins])[0]

# Apply Gaussian filter to smooth the heatmap
heatmap_smooth = gaussian_filter(
    heatmap, sigma=4
)  # Adjust sigma for more/less smoothing, higher = more diffuse

# Plot the smoothed heatmap
plt.imshow(
    heatmap_smooth.T,
    origin="lower",
    cmap="jet",
    extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
    aspect="auto",
)
plt.colorbar(label="Density")

plt.title(
    f"Heatmap of Fly Trajectories between {round(t_s / fps, 2)} and {round(t_f / fps, 2)} secs"
)
plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.show()
