# Meta Analysis Notebook

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alexprz/meta_analysis_notebook/master?filepath=notebook.ipynb)

In neuroimaging, meta-analysis is an essential tool to increase data power and to face the recurrent issue of the reproductibility of studies which are often conducted on a small number of subjects.
This notebook gathers and explains some well known meta-analysis techniques, discusses their limitations and applies them to real fMRI data.

Two choices are available to you to run interactively the notebook: remotely and locally.

## 1. Remotely
You can run the notebook interactively on Binder:
https://mybinder.org/v2/gh/alexprz/meta_analysis_notebook/master?filepath=notebook.ipynb

## 2. Locally
To be able to run the notebook locally, two choices are available.

### 2.1 Using Docker (recommended)

1. Get this repository: `$ git clone https://github.com/alexprz/meta_analysis_notebook`
2. Go to the repo: `$ cd meta_analysis_notebook`
3. Build the docker image: `$ docker build -t notebook .`
4. Start the docker container: `$ docker run -it --rm -p 8888:8888 notebook`
5. Connect to the notebook using a browser and the link displayed in your terminal (should look like http://localhost:8888/?token=sometoken).

### 2.2 Manual install (requires FSL on your computer)

0. Install FSL (optional)

Some functions (e.g Multi Level GLM) implemented in NiMARE need FSL installed on your computer. Since you still can run the other parts without FSL, this step is optional.
You can find the installation procedure [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).

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


