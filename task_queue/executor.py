import openai
from langchain import LangChain
from app.config import Config

openai.api_key = Config.OPENAI_API_KEY

def execute_task(task):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": task['system_prompt']},
                {"role": "user", "content": task['user_prompt']}
            ]
        )
        result = response.choices[0].message['content'].strip()
        # Save result to vector DB or perform further processing
        # Example: vector_db.save(result)
        print(f"Task {task['id']} completed with result: {result}")
    except Exception as e:
        print(f"Error executing task {task['id']}: {str(e)}")