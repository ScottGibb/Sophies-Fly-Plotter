##TRAJECTORY PLOTTING
import logging
import os
import sys
from tkinter import Button, Image, Label, filedialog, Tk
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tkinter import messagebox
from PIL import Image, ImageTk


logging.basicConfig(level=logging.INFO)

# Global Variables (Very Naughty... but quick)
csv_file_name = None
video_file_name = None
fps = 30  # Frames per second

# Dish Variables Defaults
dish_center_x = 550
dish_center_y = 550
## Dish is a circle with diameter of 55mm
dish_radius = 550


def get_csv_file_path():
    """
    Opens a file explorer to select the flies Excel file
    Returns:
        str: The file path selected by the user
    """
    global csv_file_name
    file_types = [("Excel files", "*.xlsb *.xlsx *.xls *.csv")]
    csv_file_name = get_file_path(file_types)
    logging.info(f"Selected file: {csv_file_name}")

def get_video_file_path():
    """
    Opens a file explorer to select the video file
    Returns:
        str: The file path selected by the user
    """
    global video_file_name
    file_types = [("Video files", "*.mp4 *.avi *.mov")]
    video_file = get_file_path(file_types)
    logging.info(f"Selected file: {video_file_name}")
    
def get_file_path(file_types) -> str:
    file_name = filedialog.askopenfilename(
        title="Select the flies Excel file",
        filetypes=file_types,
    )
    return file_name

def validate_path() -> None:
    """
    Check if the file path exists
    """
    global csv_file_name
    if os.path.exists(csv_file_name):
        logging.info(f"Good job, the path {csv_file_name} exists!!")
    else:
        logging.error(
            "It looks like the path you gave me doesn't exist.... "
            "You sure you have supplied the right path?"
            "The path you gave me was: " + csv_file_name
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
    draw_dish_and_set_limits(plt)
    # Add titles and labels
    plt.title("Plot of Fly Data (x, y) Pairs")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.grid(True)
    plot_1.show()


def plot_heatmap(trajectories: pd.DataFrame) -> None:
    """
    Plot the heatmap of the flies
    Args:
        trajectories (pd.DataFrame): The trajectories to plot
    """
    global fps
    plt_2 = plt.figure()

    num_flys = int(trajectories.shape[1] / 2)
    logging.info(f"Plotting {num_flys} flys")

    # Gather all X and Y values
    x_values = []
    y_values = []
    for i in range(1, num_flys + 1):
        x_values.extend(trajectories[f"x{i}"].values)
        y_values.extend(trajectories[f"y{i}"].values)

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Remove NaNs
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid_mask]
    y_values = y_values[valid_mask]


    t_s = 0
    t_f = x_values.size if x_values.size > 0 else 0
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()
    border = 0.05
    grid_resolution = 60

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

    heatmap, _, _ = np.histogram2d(x_values, y_values, bins=[x_bins, y_bins])
    heatmap_smooth = gaussian_filter(heatmap, sigma=3)

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
    draw_dish_and_set_limits(plt)

    plt_2.show()


def draw_dish_and_set_limits(plt) -> tuple:
    """
    Draw a dish and return the center of the dish and radius
    """
    global dish_center_x, dish_center_y, dish_radius
    circle = plt.Circle(
        (dish_center_x, dish_center_y), dish_radius, color="r", fill=False
    )
    plt.gca().add_artist(circle)
    plt.xlim(dish_center_x - dish_radius, dish_center_x + dish_radius)
    plt.ylim(dish_center_y - dish_radius, dish_center_y + dish_radius)
    plt.gca().set_aspect("equal", adjustable="box")


def process_sheet() -> None:
    """
    Process the selected file
    Check for issues with the format of the file
    Plot the values on a graph ignore nan values
    """
    global csv_file_name, video_file_name
    
    # if video_file_name is None or csv_file_name is None:
    #     logging.error("You need to select a Video file first!")
    #     messagebox.showwarning("Missing Video File", "You need to select a file first!")
    #     return    
    if csv_file_name is None:
        logging.error("You need to select a  CSV file first!")
        messagebox.showwarning("Missing CSV File", "You need to select a file first!")
        return
    calculate_center_and_radius()
    sheet = load_csv_file(csv_file_name)
    validate_trajectories(sheet)
    plot_cartesian_data(sheet)
    plot_heatmap(sheet)

def calculate_center_and_radius():
    global dish_center_x, dish_center_y, dish_radius, video_file_name
    # OpenCV code to calculate the center and radius of the dish
    # Load video
    cap = cv2.VideoCapture(video_file_name)
    if not cap.isOpened():
        logging.error("Error opening video file")
        messagebox.showwarning("Error Opening Video", "Error opening video file")
        return      
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        logging.error("Error reading video file")
        messagebox.showwarning("Error Reading Video", "Error reading video file")
        return
    # Perform hough circle transform
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is None:
        logging.error("No circles found in the video")
        messagebox.showwarning("No Circles Found", "No circles found in the video")
        return
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Find the Biggest Circle
        if i[2] > dish_radius:
            dish_center_x = i[0]
            dish_center_y = i[1]
            dish_radius = i[2]
    
    logging.info(f"Center of the dish: ({dish_center_x}, {dish_center_y})")
    logging.info(f"Radius of the dish: {dish_radius}")
    cap.release()
    
def main_gui() -> None:
    """
    Main GUI for the Fly Swapper
    """
    root = Tk()
    root.title("Fly Swapper 4000")
    text = Label(root, text="Fly Swapper 4000")
    text.pack()

    # Add Image to Main GUI
    img = Image.open("./docs/Fly.webp")
    img = img.resize((100, 100), Image.Resampling.LANCZOS)
    fly_img = ImageTk.PhotoImage(img)
    panel = Label(root, image=fly_img)
    panel.image = fly_img
    panel.pack()
    
    # File Selection
    button = Button(root, text="Select File", command=get_csv_file_path)
    text.pack()
    button.pack()
    
    # Video Selection
    button = Button(root, text="Select Video", command=get_video_file_path)
    text.pack()
    button.pack()

    # Process Button
    button = Button(root, text="Process", command=process_sheet)
    button.pack()

    # Run the GUI
    root.mainloop()


if __name__ == "__main__":
    main_gui()


    
