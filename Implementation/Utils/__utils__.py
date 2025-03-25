import time
from contextlib import contextmanager

@contextmanager
def timer(task_name=""):
    start_time = time.time()
    yield  # This is where the `with` block executes
    end_time = time.time()
    print(f"{task_name} took {end_time - start_time:.2f} seconds.")