import threading
import time
a = 0
def increase_a():
    global a
    while True:
        a += 1
        time.sleep(1)

if __name__ == "__main__":
    a = 0
    thread = threading.Thread(target=increase_a)
    thread.start()
