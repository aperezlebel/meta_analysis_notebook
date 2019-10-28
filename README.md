# Meta Analysis Notebook

In neuroimaging, meta-analysis is an essential tool to increase data power and to face the recurrent issue of the reproductibility of studies which are often conducted on a small number of subjects.
This notebook gathers and explains some well known meta-analysis techniques, discusses their limitations and applies them to real fMRI data.

Two choices are available to you whether you want to be able to rerun the notebook.

## Viewing only
Just view the notebook without the possibility of running it here:
https://nbviewer.jupyter.org/github/alexprz/meta_analysis_notebook/blob/master/notebook.ipynb

## Installation (view and run)
To be able to run the notebook, the following steps are recommended:
1. Create a virtual environment and install requirements.

```
cd /path/to/repo
python3 -m venv venv
source venv/bin/activate
xargs -L 1 pip install < requirements.txt
```

**NB:** Some packages (e.g nipy) have `setup_requires` type of requirements, which requires to have other packages already installed before being able to install dependencies. We thus need to install packages in a specific order, but the habitual `pip install -r requirements.txt` does not preserve order, hence the use of the workaround `xargs -L 1 pip install < requirements.txt`.

2. Create a jupyter kernel of the virtual environment and launch the notebook.

```
python -m ipykernel install --user --name=venv_meta
jupyter notebook notebook.ipynb
```

3. Once openned the notebook, **change the kernel to venv_meta**.
