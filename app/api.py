from app.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.tasks import submit_task, get_all_tasks, get_task, update_task
from openai import OpenAI, AsyncOpenAI
import os
import asyncio
from typing import Optional

app = FastAPI()

# instantiate OpenAI client using the API key
# Load OpenAI API key from environment variables
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

async_client = AsyncOpenAI(
    api_key=Config.OPENAI_API_KEY
)

class Message(BaseModel):
    role: str
    content: str

class TaskData(BaseModel):
    task_id: Optional[int] = None
    system_prompt: str
    user_prompt: str
    past_messages: list[Message] = []

async def fetch_openai_response(task_id: int, system_prompt: str, user_prompt: str, past_messages: list[Message]):
    try:
        messages = [{"role": "system", "content": system_prompt}] + [message.dict() for message in past_messages] + [{"role": "user", "content": user_prompt}]
        
        response = await async_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=1,
        )
        result = response.choices[0].message.content.strip()
        
        # Update the task with the response and past messages
        new_message = {"role": "assistant", "content": result}
        past_messages.append({"role": "system", "content": system_prompt})
        past_messages.append({"role": "user", "content": user_prompt})
        past_messages.append(new_message)
        
        update_task(task_id, {
            "response": result,
            "past_messages": past_messages
        })
    except Exception as e:
        update_task(task_id, {"response": f"OpenAI API error: {str(e)}"})


@app.post("/genai-tasks")
async def submit_genai_task(task_data: TaskData):
    if task_data.task_id:
        # Retrieve the existing task
        existing_task = get_task(task_data.task_id)
        if not existing_task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Append new input to past messages
        past_messages = existing_task['past_messages']
        past_messages.append({"role": "system", "content": task_data.system_prompt})
        past_messages.append({"role": "user", "content": task_data.user_prompt})
        
        # Update the task with the new input
        update_task(task_data.task_id, {
            "system_prompt": task_data.system_prompt,
            "user_prompt": task_data.user_prompt,
            "past_messages": past_messages,
            "response": None
        })
        
        # Call OpenAI asynchronously
        asyncio.create_task(fetch_openai_response(task_data.task_id, task_data.system_prompt, task_data.user_prompt, past_messages))
        
        return {"task_id": task_data.task_id}
    else:
        # Store the task without the OpenAI response
        task_id = submit_task({
            "system_prompt": task_data.system_prompt,
            "user_prompt": task_data.user_prompt,
            "past_messages": [message.dict() for message in task_data.past_messages],
            "response": None
        })
        
        # Call OpenAI asynchronously
        asyncio.create_task(fetch_openai_response(task_id, task_data.system_prompt, task_data.user_prompt, task_data.past_messages))
        
        return {"task_id": task_id}

@app.get("/genai-tasks")
async def get_genai_tasks():
    tasks = get_all_tasks()
    return tasks

@app.get("/genai-tasks/{task_id}")
async def get_genai_task(task_id: int):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task