import time


def show_elapsed_time(start_time: float, name: str = "None"):
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    print(
        f"{name}, Elapsed time: {elapsed_time:.3f} seconds, frame_rate: {1 / elapsed_time:.3f}"
    )

    return elapsed_time
