"""Plots a recorded mouse trace"""

from PIL import Image, ImageDraw
import numpy as np

class TracePlotter:
    """Class to Plot Trace b/c of global variables"""

    def __init__(self):
        # init fields
        self.size = (1600, 900)
        self.input_file = "mouse_data.txt"
        self.output_file = "trace.jpg"
        self.mouse_data = np.array([], dtype=int)
        self.mouse_samples = 0
        self.image = Image.new("RGB", size=self.size)
        self.draw = ImageDraw.Draw(self.image)

        # load input data from text file
        self.load_data()

        self.plot_trace()

        self.image.save(self.output_file)

    def load_data(self):
        """Loads data from numpy txt file"""
        self.mouse_data = np.loadtxt(self.input_file, dtype=int)

        # transpose mouse data, to let first index be x or y
        self.mouse_data = np.transpose(self.mouse_data)
        self.mouse_samples = np.size(self.mouse_data[0])
        print("Processing %d samples" % self.mouse_samples)

    def plot_trace(self):
        """Iterates over Mouse Samples and Plots the Trace"""
        for i in range(0, self.mouse_samples - 1):
            # determine initial and final coordinates
            x_i = self.mouse_data[0][i]
            y_i = self.mouse_data[1][i]
            x_f = self.mouse_data[0][i+1]
            y_f = self.mouse_data[1][i+1]
            line_parameters = [(x_i, y_i), (x_f, y_f)]

            self.draw.line(line_parameters, fill=(0, 0, 255))


if __name__ == "__main__":
    plotter = TracePlotter()
