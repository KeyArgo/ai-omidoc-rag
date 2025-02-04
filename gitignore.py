def create_gitignore():
    gitignore_content = """# First, ignore everything
*
*/*

# Then, explicitly allow only our Python files
!check_ollama.py
!gitignore.py
!query_rag.py
!rag_api.py
!rag_cli.py
!rag_process.py
!rag_query.py
!rag_system.py
!test_all.py
!test_ollama.py
!test_rag.py
!test_search.py

# Still ignore these even if they match above patterns
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.py[cod]
*$py.class

# Virtual Environment - multiple formats
rag-env/
rag-env/*
/rag-env/
.rag-env/
*rag-env/
venv/
env/
.env/
.venv/
ENV/
virtual/
virtualenv/
*env*/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
pip-log.txt
pip-delete-this-directory.txt

# Project specific
*.txt
*.md
chroma_db/
/chroma_db
chroma_db/*
data/
/data
data/*
*.db
*.sqlite
*.sqlite3

# IDE
.idea/
.vscode/
*.swp
*.swo
.vs/
*.sublime-workspace
*.sublime-project

# OS specific
.DS_Store
Thumbs.db
*.bak
*.tmp
*.temp
"""
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)

if __name__ == "__main__":
    create_gitignore()
    print(".gitignore has been created successfully!") 