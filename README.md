

Creating a virtual environment to install requirements is recommended:

```
cd /path/to/repo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_git.txt
```

Then, create a jupyter kernel of the virtual environment and launch jupyter:

```
python -m ipykernel install --user --name=venv_meta
jupyter notebook
```

Once openned the notebook, change the kernel to venv_meta.
