from fastapi import FastAPI

api = FastAPI()

all_todos = [
    {"todo_id": 1, "todo_name": "task1", "todo_descritpion": "do task 1"},
    {"todo_id": 2, "todo_name": "task2", "todo_descritpion": "do task 2"},
    {"todo_id": 3, "todo_name": "task3", "todo_descritpion": "do task 3"},
    {"todo_id": 4, "todo_name": "task4", "todo_descritpion": "do task 4"},
    {"todo_id": 5, "todo_name": "task5", "todo_descritpion": "do task 5"},
]


@api.get("/")
def index():
    return {"message": "Hello World"}


@api.get("/todos/{todo_id}")
def get_todo(todo_id: int):
    for todo in all_todos:
        if todo["todo_id"] == todo_id:
            return {"result": todo}


@api.get("/todos")
def get_items(first_n: int = None):
    if first_n:
        return all_todos[:first_n]
    else:
        return all_todos
