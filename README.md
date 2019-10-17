# Meta Analysis Notebook

In neuroimaging, meta-analysis is an essential tool to increase data power and to face the recurrent issue of the reproductibility of studies which are often conducted on a small number of subjects.
This notebook gathers and explains some well known meta-analysis techniques, discusses their limitations and applies them to real fMRI data.

## Installation
To be able to run the notebook, the following steps are recommended:
1. Create a virtual environment and install requirements.

```
cd /path/to/repo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_git.txt
```

2. Create a jupyter kernel of the virtual environment and launch the notebook.

```
python -m ipykernel install --user --name=venv_meta
jupyter notebook notebook.ipynb
```

3. Once openned the notebook, **change the kernel to venv_meta**.
