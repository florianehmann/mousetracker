"""Mouse Tracking Script"""

from time import sleep
import signal
from pymouse import PyMouse

MOUSE = PyMouse()

CAPTURING = True
DELTAT = 0.1

def capture_position():
    """Loop for mouse capture"""
    while CAPTURING:
        sleep(DELTAT)
        print("Mouse is at (%d,\t%d)" % MOUSE.position())

def finish_capture(sig, frame):
    """Finishes the capture of the movement and saves the data"""
    global CAPTURING
    CAPTURING = False
    print("Finished capture")

if __name__ == "__main__":
    print("Script initialized")

    # register ctrl c handler
    signal.signal(signal.SIGINT, finish_capture)

    # go to capture loop
    capture_position()
