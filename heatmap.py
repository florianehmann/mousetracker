"""Generates a Heatmap based on Mouse Data"""

from PIL import Image, ImageDraw
import numpy as np

class Heatmap:
    """Class to generate a Heatmap"""

    def __init__(self):
        # init fields
        self.screen_size = (1600, 900)
        self.oversize = 2
        self.canvas_size = (self.oversize * self.screen_size[0], self.oversize
                            * self.screen_size[1])
        self.input_file = "mouse_data.txt"
        self.output_file_histogram = "histogram.jpg"
        self.output_file_heatmap = "heatmap.jpg"
        self.mouse_data = np.array([], dtype=int)
        self.histogram = np.zeros(self.screen_size, dtype=int)
        self.mouse_samples = 0
        self.image_histogram = Image.new("RGB", size=self.screen_size)
        self.draw_histogram = ImageDraw.Draw(self.image_histogram)
        self.image_heatmap = Image.new("RGB", size=self.screen_size)
        self.draw_heatmap = ImageDraw.Draw(self.image_heatmap)

        # parameters and fields for Gaussian blur
        self.padx = int(self.screen_size[0] / 2)
        self.pady = int(self.screen_size[1] / 2)
        self.peak_size = self.screen_size[1] / 35

        # load input data from text file
        print("Loading Data")
        self.load_data()

        print("Generating Histogram")
        self.generate_histogram()
        self.normalize_histogram()
        self.plot_histogram()
        self.image_histogram.save(self.output_file_histogram)

        print("Generating Heatmap")
        self.generate_heatmap()
        self.normalize_heatmap()
        print("Plotting Heatmap")
        self.plot_heatmap()
        self.image_heatmap.save(self.output_file_heatmap)

    def load_data(self):
        """Loads data from numpy txt file"""
        self.mouse_data = np.loadtxt(self.input_file, dtype=int)

        # transpose mouse data, to let first index be x or y
        self.mouse_data = np.transpose(self.mouse_data)
        self.mouse_samples = np.size(self.mouse_data[0])
        print("\tLoaded %d samples" % self.mouse_samples)

    def generate_histogram(self):
        """Generates a Histogram from the Mouse Data"""
        for i in range(0, self.mouse_samples):
            # increase pixel counter
            mousex = self.mouse_data[0][i]
            mousey = self.mouse_data[1][i]
            if i > 0:
                if mousex != self.mouse_data[0][i - 1] \
                        or mousey != self.mouse_data[1][i - 1]:

                    self.histogram[mousex, mousey] += 1
            else:
                self.histogram[mousex, mousey] += 1

    def normalize_histogram(self):
        """Normalizes the Histogram to values between 0.0 and 1.0"""
        self.histogram = self.histogram / self.histogram.max()

    def plot_histogram(self):
        """Plots the normalized Histogram"""
        for xcoord in range(0, self.screen_size[0]):
            for ycoord in range(0, self.screen_size[1]):
                blue = min(255, int(self.histogram[xcoord, ycoord] * 255 * 128))
                color = (0, blue, blue)
                self.draw_histogram.point((xcoord, ycoord), fill=color)

    def canvas_coordinates(self, screenx, screeny):
        """Returns (x,y) of the Coordinates on the oversized Canvas"""
        return (screenx * self.oversize, screeny * self.oversize)

    def peak_dist(self, x, y):
        """Gaussian Peak centered on the Screen"""
        return np.exp((- (x - self.screen_size[0] / 2)**2 \
            - (y - self.screen_size[1] / 2)**2) \
            /(2 * self.peak_size**2))

    def pad_array(self, arr):
        """Pads an Array to prevent circular Artifacts in Convolution"""
        return np.pad(arr, ((self.padx, self.padx), (self.pady, self.pady)), 
                      mode='constant')

    def generate_heatmap(self):
        """Generated a Heatmap by blurring a normalized Histogram"""
        xaxis = np.linspace(0, self.screen_size[0], self.screen_size[0])
        yaxis = np.linspace(0, self.screen_size[1], self.screen_size[1])

        # generate Gaussian peak
        peak = self.peak_dist(xaxis[:,None], yaxis[None,:])
        peak = self.pad_array(peak)
        peak = np.roll(peak, int(self.screen_size[0] / 2 + self.padx), axis=0)
        peak = np.roll(peak, int(self.screen_size[1] / 2 + self.pady), axis=1)

        # transform peak for convolution theorem
        peak_fft = np.fft.fft2(peak)

        # prepare histogram for convolution
        conv = self.pad_array(self.histogram)
        conv = np.fft.fft2(conv)

        # convolve
        conv = peak_fft * conv
        conv = np.fft.ifft2(conv)
        conv = conv[self.padx:-self.padx,self.pady:-self.pady]

        self.heatmap = np.abs(conv)

    def normalize_heatmap(self):
        """Normalizes the Heatmap to Values from 0 to 1"""
        self.heatmap = self.heatmap / np.max(self.heatmap)

    def color_gradient(self, l, lmin, lmax, ci, cf):
        """A General Part of a Color Gradient"""
        ci = np.array(ci)
        cf = np.array(cf)
        v = (cf - ci)
        lconstrained = l
        if (l > lmax):
            lconstrained = lmax
        elif (l < lmin):
            lconstrained = 0
        c = ci + (lconstrained - lmin) / (lmax - lmin) * v
        return c

    def color_function(self, l):
        """Returns a Color of a Gradient"""
        c = np.array([0,0,0])
        if l <= 0.3:
        	c = self.color_gradient(l, 0, 0.3, [0, 0, 0], [204, 34, 0])
        elif l > 0.3 and l <= 0.6:
             c = self.color_gradient(l, 0.3, 0.6, [204, 34, 0], [255, 183, 0])
        elif l > 0.6:
             c = self.color_gradient(l, 0.6, 1.0, [255, 183, 0], [255, 255, 255])

        c = c.astype(int)
        return tuple(c)

    def plot_heatmap(self):
        """Plots the Heatmap"""
        print("\tStart Plotting")

        for xcoord in range(0, self.screen_size[0]):
            CURSOR_UP_ONE = '\x1b[1A'
            ERASE_LINE = '\x1b[2K'
            print(CURSOR_UP_ONE + ERASE_LINE, end='')
            print("\tFinished %.1f percent of the Heatmap" %
                  (xcoord * 100 / self.screen_size[0]))

            for ycoord in range(0, self.screen_size[1]):
                color = self.color_function(self.heatmap[xcoord, ycoord])
                #blue = int(self.heatmap[xcoord][ycoord] * 255)
                #color = (0, blue, blue)
                self.draw_heatmap.point((xcoord, ycoord), fill=color)


if __name__ == "__main__":
    HEATMAP = Heatmap()
