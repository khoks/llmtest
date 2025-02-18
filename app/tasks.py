from models.task import Task

tasks = {}
task_id_counter = 1

def submit_task(task_data):
    global task_id_counter
    task = Task(task_id_counter, task_data)
    tasks[task_id_counter] = task
    task_id_counter += 1
    return task.id

def get_all_tasks():
    return {task_id: task.to_dict() for task_id, task in tasks.items()}

def get_task(task_id):
    task = tasks.get(task_id)
    return task.to_dict() if task else None

def update_task(task_id, update_data):
    task = tasks.get(task_id)
    if task:
        task.data.update(update_data)