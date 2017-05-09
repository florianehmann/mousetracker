"""Mouse Tracking Script"""

from time import sleep
import signal
import numpy as np
from pymouse import PyMouse

MOUSE = PyMouse()

CAPTURING = True
DELTAT = 1/120

MOUSE_DATA = np.array([], dtype=int)

def capture_position():
    """Loop for mouse capture"""
    global MOUSE_DATA
    while CAPTURING:
        sleep(DELTAT)
        mouse_x, mouse_y = MOUSE.position()
        MOUSE_DATA = np.append(MOUSE_DATA, np.array([mouse_x, mouse_y]))
        #print("Mouse is at (%d,%d)" % (mouse_x, mouse_y))

def finish_capture(sig, frame):
    """Finishes the capture of the movement and saves the data"""
    global CAPTURING, MOUSE_DATA
    CAPTURING = False
    print("Finished capture")

    # save data
    MOUSE_DATA = np.reshape(MOUSE_DATA, (-1, 2))
    #print(MOUSE_DATA)
    headerstring = "Mouse Data\nTime Spacing: %f" % (DELTAT)
    np.savetxt("mouse_data.txt", MOUSE_DATA, fmt="%d", header=headerstring)

if __name__ == "__main__":
    print("Script initialized")

    # register ctrl c handler
    signal.signal(signal.SIGINT, finish_capture)

    # go to capture loop
    capture_position()
