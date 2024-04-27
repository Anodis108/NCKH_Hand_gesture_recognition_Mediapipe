import threading
import f1
import f2

if __name__ == "__main__":
    thread1 = threading.Thread(target=f1.increase_a)
    thread2 = threading.Thread(target=f2.print_a)
    thread1.start()
    thread2.start()
