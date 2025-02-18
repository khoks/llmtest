# Project Name

## Description
A brief description of what this project does and who it's for.

## Creation of this Repo
Below commands were used to create this application for the first time on windows laptop


```bash
git init
Add-Content .gitignore ".vscode`r`nvenv`r`n.venv`r`n"
git add .
git commit -m "first commit with readme and gitignore"
git remote add origin https://github.com/khoks/llmtest
git push -u origin master
pip install poetry
```
Added poetry path in path environment variable
Initialize Poetry
```bash
poetry init
poetry add fastapi
poetry add uvicorn
poetry add openai
poetry install
OPENAI_API_KEY=your_openai_api_key
```


## Installation
Instructions on how to install and set up the project.

```bash
# Example:
git clone https://github.com/yourusername/your-repo.git
cd your-repo



```


## Running the Application

```bash
poetry run python run.py
```

## Contributing
Guidelines for contributing to the project.

## License
Information about the project's license.

## Contact
Contact information for the project maintainers.