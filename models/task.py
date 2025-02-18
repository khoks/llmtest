class Task:
    def __init__(self, task_id, data):
        self.id = task_id
        self.data = data

    def to_dict(self):
        return {
            'id': self.id,
            'system_prompt': self.data.get('system_prompt'),
            'user_prompt': self.data.get('user_prompt'),
            'past_messages': self.data.get('past_messages', []),
            'response': self.data.get('response')
        }