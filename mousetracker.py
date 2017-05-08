from pymouse import PyMouse
from time import sleep
import signal
import sys

mouse = PyMouse()

capturing = True
deltat = 0.1

def capture_position():
    """Loop for mouse capture"""
    while capturing:
        sleep(0.1)
        print("Mouse is at (%d,\t%d)" % mouse.position())

def finish_capture(signal, frame):
    """Finishes the capture of the movement and saves the data"""
    global capturing
    capturing = False
    print("Finished capture")

if __name__ == "__main__":
    print("Script initialized")

    # register ctrl c handler
    signal.signal(signal.SIGINT, finish_capture);

    # go to capture loop
    capture_position()
