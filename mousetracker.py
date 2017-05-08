from pymouse import PyMouse, PyMouseEvent

m = PyMouse()

if __name__ == "__main__":
    print("Script initialized")

    print("%d, %d" % m.position());
