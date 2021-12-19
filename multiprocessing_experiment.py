import signal
import time
# import multiprocessing as mp
import torch.multiprocessing as mp

import numpy as np


class GracefulExit(Exception):
    pass


def signal_handler(signum, frame):
    print("Received exit signal")
    raise GracefulExit()


def subprocess_function():
    try:
        sem = mp.Semaphore()
        print("Acquiring semaphore")
        sem.acquire()
        print("Semaphore acquired")

        print("Blocking on semaphore - waiting for SIGTERM")
        sem.acquire()
    except GracefulExit:
        print("Subprocess exiting gracefully")


def subprocess2(idx):
    while True:
        print(f"Subprocess {idx}")
        time.sleep(10.0)


def process_return_val(idx, input_val):
    wait_time = np.random.uniform(input_val)
    print(f"Subprocess {idx} waiting for {wait_time:.3f}")
    time.sleep(wait_time)
    return_val = input_val ** 2
    print(f"Subprocess {idx} returning {return_val}")
    return return_val


def return_process_wrapper(idx, input_val, return_queue):
    return_queue.put(process_return_val(idx, input_val))


def main():
    num_processes = 10

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # # Start a subprocess and wait for it to terminate.
    # p = mp.Process(target=subprocess2, args=())
    # p.start()
    #
    # print(f"Subprocess pid: {p.pid}")
    # p.join()

    # for idx in range(num_processes):
    #     p = mp.Process(target=subprocess2, args=(idx,))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    processes = []
    return_queue = mp.Queue()
    inputs = [np.random.randint(50) for _ in range(100)]

    proc_idx = 0
    # TODO implement
    for proc_idx in range(num_processes):
        p = mp.Process(target=return_process_wrapper, args=(proc_idx, input_val, return_queue))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    pass


if __name__ == "__main__":
    main()
