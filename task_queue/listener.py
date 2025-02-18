import threading
import queue
from task_queue.executor import execute_task

task_queue = queue.Queue()

def listen_for_tasks():
    while True:
        task = task_queue.get()
        if task is None:
            break
        execute_task(task)
        task_queue.task_done()

listener_thread = threading.Thread(target=listen_for_tasks)
listener_thread.start()