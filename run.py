""" from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True) """

import uvicorn
from app.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)