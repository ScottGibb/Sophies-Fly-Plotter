import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging


def plot_heatmap(trajectories: pd.DataFrame) -> None:
    """
    Plot the heatmap of the flies
    Args:
        trajectories (pd.DataFrame): The trajectories to plot
    """
    # Calculate the number of flies based on columns
    num_flys = int(trajectories.shape[1] / 2)
    logging.info(f"Plotting {num_flys} flys")

    # Create lists of all X and Y values
    x_values = []
    y_values = []
    for i in range(1, num_flys + 1):
        x_values.extend(trajectories[f"x{i}"].values)  # Flatten x-values
        y_values.extend(trajectories[f"y{i}"].values)  # Flatten y-values

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Remove NaN and inf values
    valid_mask = np.isfinite(x_values) & np.isfinite(
        y_values
    )  # Valid points are finite
    x_values = x_values[valid_mask]
    y_values = y_values[valid_mask]

    # Log the number of valid points
    logging.info(f"Number of valid points: {len(x_values)}")

    # Create a new figure for the heatmap
    plt.figure(figsize=(8, 6))

    # Create 2D Heat Map
    plt.hist2d(x_values, y_values, bins=(50, 50), cmap=plt.cm.jet)

    # Add color bar for intensity reference
    plt.colorbar(label="Density")

    # Add titles and labels
    plt.title("Heatmap of Fly Data (x, y) Pairs")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.grid(True)

    # Show the plot
    plt.show()


# Example DataFrame with NaN/inf values
data = {
    "x1": [1, 2, 3, np.nan, 5],
    "y1": [1, 2, np.inf, 4, 5],
    "x2": [1, 2, 3, 4, 5],
    "y2": [5, 4, 3, 2, 1],
}
df = pd.DataFrame(data)

# Plot the heatmap
plot_heatmap(df)
