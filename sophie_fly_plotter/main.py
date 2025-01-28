##TRAJECTORY PLOTTING
import logging
import os
import sys
from tkinter import Button, Label, filedialog, Tk
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)

# Global Variables (Very Naughty... but quick)
file_name = None
fps = 30  # Frames per second


def get_file_path_from_gui():
    """
    Opens a file explorer to select the flies Excel file
    Returns:
        str: The file path selected by the user
    """
    global file_name
    file_name = filedialog.askopenfilename(
        title="Select the flies Excel file",
        filetypes=[("Excel files", "*.xlsb *.xlsx *.xls *.csv")],
    )
    logging.info(f"Selected file: {file_name}")


def validate_path() -> None:
    """
    Check if the file path exists
    """
    global file_name
    if os.path.exists(file_name):
        logging.info(f"Good job, the path {file_name} exists!!")
    else:
        logging.error(
            "It looks like the path you gave me doesn't exist.... "
            "You sure you have supplied the right path?"
            "The path you gave me was: " + file_name
        )
        logging.error("Closing Script, try again")
        sys.exit(-1)


def load_csv_file(file_name: str) -> pd.DataFrame:
    """
    Load the CSV file into a pandas DataFrame
    Args:
        file_name (str): The file path of the CSV file
    Returns:
        pd.DataFrame: The CSV file as a pandas DataFrame
    """
    logging.info(f"Loading the CSV file: {file_name}")
    try:
        trajectories = pd.read_csv(file_name, delimiter=",")
    except Exception as e:
        raise Exception(f"Error loading data from {file_name}: {e}")
    return trajectories


def validate_trajectories(trajectories: pd.DataFrame) -> None:
    # Check how many flies are there
    num_columns = trajectories.shape[1]
    num_flys = num_columns / 2
    logging.info(
        f"There are {num_columns} in this spreadsheet, this means there are {num_flys} flys in this sheet"
    )
    if num_flys % 1 != 0:
        logging.error(
            f"Something is wrong with the number of columns in the file: {num_columns}, it should be an even number!!"
        )
        sys.exit(-1)


def remove_nans(trajectories: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any rows that contain NaN values
    Args:
        trajectories (pd.DataFrame): The trajectories to clean
    Returns:
        pd.DataFrame: The cleaned trajectories
    """
    logging.info("Removing any rows with NaN values")
    trajectories = trajectories.notna()
    return trajectories


def plot_cartesian_data(trajectories: pd.DataFrame) -> None:
    """
    Plot the cartesian data of the flies
    Args:
        trajectories (pd.DataFrame): The trajectories to plot
    """
    num_flys = int(trajectories.shape[1] / 2)
    logging.info(f"Plotting {num_flys} flys")
    plot_1 = plt.figure(1)
    for i in range(1, num_flys + 1):  # Loop through x1, y1 to x5, y5
        logging.info(f"Plotting fly {i}")
        plt.plot(
            trajectories[f"x{i}"],
            trajectories[f"y{i}"],
            label=f"Fly {i}",
            marker="o",
        )

    # Add titles and labels
    plt.title("Plot of Fly Data (x, y) Pairs")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.grid(True)
    plot_1.show()


def process_sheet() -> None:
    """
    Process the selected file
    Check for issues with the format of the file
    Plot the values on a graph ignore nan values
    """
    global file_name
    if file_name is None:
        logging.error("You need to select a file first!")
        sys.exit(-1)

    sheet = load_csv_file(file_name)
    validate_trajectories(sheet)
    plot_cartesian_data(sheet)


def main_gui() -> None:
    """
    Main GUI for the Fly Swapper
    """
    global threshold_input, threshold
    root = Tk()
    root.title("Fly Swapper")
    text = Label(root, text="Fly Swapper")
    text.pack()

    # File Selection
    text = Label(root, text="Select the flies Excel file")
    button = Button(root, text="Select File", command=get_file_path_from_gui)
    text.pack()
    button.pack()

    # Process Button
    button = Button(root, text="Process", command=process_sheet)
    button.pack()

    # Run the GUI
    root.mainloop()


main_gui()
