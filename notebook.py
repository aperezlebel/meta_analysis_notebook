# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A guide to neuroimaging meta-analysis

# %% [markdown]
# ## Alexandre Perez 
# Neuro-Data Science - ORIGAMI lab

# %% [markdown]
# ## Foreword

# %% [markdown]
# In neuroimaging, meta-analysis is an essential tool to better synthesize literature results and to answer the recurrent issue of the lack of reproductibility of studies conducted on a small number of subjects.  
# This notebook gathers and explains some well known meta-analysis techniques, discusses their limitations and applies them to real fMRI data. Neuroimaging meta-analysis are divided into two types: coordinate-based meta-analysis (CBMA) and image-based meta-analysis (IBMA); so is this notebook.
#
# **Warning**:  
# In order to illustrate the techniques, operations have been coded in an **inefficient** way in this notebook for the sake of clarity. Do not reuse the provided code for real analysis. These techniques have been coded in a more efficient and complete way in the python package NiMARE. The use of this package is strongly recommanded and is illustrated at the end of the notebook. 

# %% [markdown]
# ### Requirements and installation
#
# If you run this notebook on Binder, then everything is all set. However, if you run it manually, please refer to the Readme instructions in the repository to install the required dependencies.

# %% [markdown]
# ## Data

# %% [markdown]
# Before getting to the heart of the matter, let's load the data that will be used to illustrate the techniques. On first reading, one can jump to the next section: **Coordinate-based meta-analysis**.

# %%
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from os import path as osp
import multiprocessing
from joblib import Parallel, delayed
import nilearn
import numpy as np
from nilearn import masking, plotting
from nipy.labs.statistical_mapping import get_3d_peaks
import nibabel as nib
import scipy
from matplotlib import pyplot as plt
import ntpath
from nimare.dataset import Dataset
import nimare
import copy
import shutil

# Some global variables used in the notebook
input_dir = 'data/sim/'  # Studies' directory at the git repository root
study_filename = 'study_'  # Template of a study image: f'{study_filename}_{i}.nii.gz'
template = nilearn.datasets.load_mni152_template()
Ni, Nj, Nk = template.shape
affine = template.affine
gray_mask = masking.compute_gray_matter_mask(template)


# %% [markdown]
# ### Required data structure

# %% [markdown]
# The dataset is a set of 10 studies that have been entirely simulated from a real base image, from the study's map to each of the subjects' maps (not reported here). The simulation was performed by adding normal noise, previously smoothed with a gaussian kernel, to the base image. From the simulated subjects (a total of 10 for each study), the standard deviation map was derived.
#
# The data, located at the root of the repo, follow the following structure:
# ```
# root_of_repo/
#     notebook.ipynb
#     data/
#         ...
#         sim/
#             study_0.nii.gz
#             study_0_con.nii.gz
#             study_0_se.nii.gz
#             study_1.nii.gz
#             study_1_con.nii.gz
#             study_1_se.nii.gz
#             ...
# ```
# Where 
# - `study_i.nii.gz` is the z map file of the i-th study.
# - `study_i_con.nii.gz`is the contrast map file of the i-th study.
# - `study_i_se.nii.gz`is the standard deviation map of the subjects of the i-th study.
#
# Note that you can change both the data directory and the template names at the beginning of this notebook. However, your custom filenames still would have to meet the naming requirements `prefix_i_type.nii.gz` where:
# - `prefix` is custom.
# - `i` is the index of the study.
# - `type` is the map's type (z: empty string, contrast: `con`, standard deviation: `se`)

# %% [markdown]
# ### Construct list of images paths

# %%
# retrieve path of images in the data directory
img_paths = []
i = 0

while True:
    path = os.path.abspath(f'{input_dir}{study_filename}{i}.nii.gz')
    
    if not os.path.isfile(path):
        break
    
    img_paths.append(path)
    i += 1
    
N_img = len(img_paths)


# %% [markdown]
# ### Extract peaks

# %% [markdown]
# The provided data contains the full brain images (with same dimension as the MNI template image). This is perfect for image-based meta-analysis methods but not for coordinate-based ones. To demonstrate the coordinate-based meta-analysis (CBMA) techniques we need to extract from these images the peaks' coordinates. Then we will also be able to run the CBMA methods on the provided data.
#
# To do so, we use the `get_3D_peaks` function of the `nipy.labs.statistical_mapping` module of the `nipy` package.

# %%
def get_activations(path, threshold, space='ijk'):
    """
    Retrieve the activation coordinates from an image.

    Args:
        path (string or Nifti1Image): Path to or object of a
            nibabel.Nifti1Image from which to extract coordinates.
        threshold (float): Peaks under this threshold will not be detected.
        space (string): Space of coordinates. Available: 'ijk' and 'pos'.

    Returns:
        (tuple): Size 3 tuple of np.array storing respectively the X, Y and
            Z coordinates

    """
    I, J, K = [], [], []
    try:
        img = nilearn.image.load_img(path)
    except ValueError:  # File path not found
        print(f'File {path} not found. Ignored.')
        return None

    if np.isnan(img.get_fdata()).any():
        print(f'Img {path} contains Nan. Ignored.')
        return None

    img = nilearn.image.resample_to_img(img, template)

    peaks = get_3d_peaks(img, mask=gray_mask, threshold=threshold)

    if not peaks:
        return I, J, K

    for peak in peaks:
        I.append(peak[space][0])
        J.append(peak[space][1])
        K.append(peak[space][2])

    del peaks
    
    I, J, K = np.array(I), np.array(J), np.array(K)
    if space == 'ijk':
        I, J, K = I.astype(int), J.astype(int), K.astype(int)
        
    return I, J, K


# %%
activation_thresh = 1.96 
activation_peaks = [get_activations(path, activation_thresh, space='ijk') 
                    for path in img_paths]

# %% [markdown]
# We turn peaks' coordinates into binary Nifti images for convenience and create a list of nifti images:

# %%
binary_imgs = []
for I, J, K in activation_peaks:
    arr = np.zeros(template.shape)
    arr[I, J, K] = 1
    img = nib.Nifti1Image(arr, affine)
    binary_imgs.append(img)

# %% [markdown]
# Now that the data is properly loaded, let's talk about meta-analysis techniques.

# %% [markdown]
# ## Coordinate-based meta-analysis (CBMA)

# %% [markdown]
# The first type of neuroimaging meta-analysis technique is coordinate-based meta-analysis. These techniques only use a list of activation coordinates (which can be seen as binary maps) as input.  
# This part has been written using the following references: [1] [2] [5].

# %% [markdown]
# **Context**:  
# Coordinate-based meta-analysis techniques are used in the following context:
#
# One has $N$ experiments that deal with the same or related task/topic. For each of these experiments, one has a list of coordinates of activation peaks in the brain. One wants to combine these results to build a common activation map in the hope of having a stronger prior on the activation areas of a given task than each experiment can give individually.

# %% [markdown]
# ### ALE

# %% [markdown]
# The key idea of the Activation Likelihood Estimation (ALE) algorithm is to consider the activation foci not in a discrete way, but rather as a center of probability distribution.
#
# #### Notations:
# - $N$: Number of experiments/studies.
# - $S_n$: Number of subjects in the experiment $n$.
# - $F_n$: Number of foci in the experiment $n$.
#

# %% [markdown]
# #### Pipeline
# The ALE pipeline is summarized in figure [1](#ALE_fig), and can be described as follows:
#
# 1. Gather the reported activation peaks in $N$ binary maps, with one map per study.
# 2. For each of these maps, apply the following pipeline:  
#     a. Consider each peak individually and convolve each individual peak with a gaussian kernel, resulting in one map per peak. Note that the width of the gaussian kernel depends on the number of subjects in the considered experiment.  
#     b. Merge the obtained probability maps within the given study into a single map, called a Modeled Activation (MA) map.  
# 3. Merge all the MA maps across studies into a single map; this ALE map is the result of the meta-analysis.  

# %% [markdown]
# <a id='ALE_fig'></a>
# ![ALE](resource/ALE.png)

# %% [markdown]
# **Figure 1**: The ALE pipeline. The input are $N$ binary maps, one for each of the studies.

# %% [markdown]
# 3 points remain unclear for now:
# - How the gaussian kernel depends on the number of subjects.
# - How to merge probability maps into MA maps (labeled by 1 in figure 1).
# - How to merge MA maps into an ALE map (labeled by 2 in figure 1).
#
# Let's clarify these points.

# %% [markdown]
# #### ALE Kernel

# %% [markdown]
# The idea of having a kernel depend on the number of subjects of an experiment is justified by the fact that an experiment with a low number of subjects has more uncertainty than an experiment with a high number of subjects. Therefore, the lower the number of subjects in an experiment, the greater is the FWHM of the gaussian kernel.

# %% [markdown]
# #### Merging probability maps (within a study)

# %% [markdown]
# Why do we split the binary map into as many maps as foci, leaving only one focus in each map, to finally merge them together? The convolution is linear right, why don't we simply convolve the full binary map?  
# Well, the answer is we don't apply a linear operation to merge the probability maps. Basically, the MA map is derived from the probability maps by taking the maximum of all the probability maps at each point in space.

# %% [markdown]
# #### Merging MA maps (across studies)

# %% [markdown]
# To understand how the MA maps are merged together into the ALE map, we first have to remember the meaning of the MA maps and what we want to express in the ALE map.

# %% [markdown]
# Remember that the key point of ALE is to see the discrete foci as the centers of probability distributions. Hence, the MA map of an experiment can be thought as a rational modelisation of the distribution of the experiment's peaks. More specifically, the value associated with a given coordinate (i, j, k) represents the likelihood that an activation peak lies on this coordinate.
# If we denote $\textit{MA}_n$ the MA map associated with the experiment $n$, $$\textit{MA}_n(i, j, k) = P(\mbox{ijk activated in the experiment $n$})$$

# %% [markdown]
# A rational expectation for a meta-analysis map such as ALE is to give us an approximation of the likelihood that a given coordinate $ijk$ is truly activated in at least one experiment. Hence, if we denote $\textit{ALE}$ the ALE map:
# $$
# \begin{equation} \label{eq1}
#     \begin{split}
#     \textit{ALE}(i,j,k) & = P(\mbox{$ijk$ activated by at least one experiment}) \\
#      & = 1 - P(\mbox{$ijk$ not activated by any experiment}) \\
#      & = 1 - P(\mbox{For each experiment $n$, $ijk$ not activated in $\textit{MA}_n$}) \\
#      & = 1 - \prod_{n=1}^{N}P(\mbox{$ijk$ not activated in $\textit{MA}_n$}) \\
#      & = 1 - \prod_{n=1}^{N}(1-P(\mbox{$ijk$ activated in $\textit{MA}_n$})) \\
#      & = 1 - \prod_{n=1}^{N}(1 - \textit{MA}_n(i,j,k))
#     \end{split}
# \end{equation}
# $$

# %% [markdown]
# #### Mathematical recap

# %% [markdown]
# For the sake of simplicity and to be closer to the implementation, we will consider the mathematical objects directly in the discrete 3D brain space rather than the continuous case.

# %% [markdown]
# We recall the previous notations:
# - $N$: Number of experiments/studies.
# - $S_n$: Number of subjects in experiment $n$.
# - $F_n$: Number of foci in experiment $n$.

# %% [markdown]
# And add new ones:
# - Let $(N_i, N_j, N_k)$ be the shape of the brain space. Typically $(N_i, N_j, N_k) = (91, 109, 91)$ in the MNI space.
# - For all $n \in [1 .. N],$
#     * let $B_n \in \{0, 1\}^{N_iN_jN_k}$ be the binary map associated with experiment $n$.
#     * let $D_{p, n} \in \{0, 1\}^{N_iN_jN_k}$ be the binary dirac map associated with experiment $n$ and focus $p$ for all $p \in [1 .. F_n]$. In other words, a null map with only the position of the focus $p$ set to 1 (called dirac map on figure [1](#ALE)).
#     * let $\textit{MA}_n \in \mathbb{R}^{N_iN_jN_k}$ be the Modeled Activation map of experiment $n$.
#     * let $G_{S_n}$ be the gaussian kernel associated with experiment $n$.
# - Let $\textit{ALE} \in \mathbb{R}^{N_iN_jN_k}$ be the desired ALE map.

# %% [markdown]
# Given these notations, we have the following: 

# %% [markdown]
# $$
# \begin{equation}
#     B_n(i, j, k) = \sum_{p=1}^{F_n}D_{p, n}(i, j, k)
# \end{equation}
# $$

# %% [markdown]
# $$
# \begin{equation}
#     \textit{MA}_n(i, j, k) = \underset{n \in [1..N]}{\text{max}} \Big[(D_{p, n}*G_{S_n})(i, j, k)\Big]
# \end{equation}
# $$

# %% [markdown]
# $$
# \begin{equation}
#     \textit{ALE}(i, j, k) = 1 - \prod_{n=1}^{N}(1 - \textit{MA}_n(i,j,k))
# \end{equation}
# $$

# %% [markdown]
# $$
# \begin{equation}
#     \textit{ALE} = 1 - \prod_{n=1}^{N}(1 - \textit{MA}_n)
# \end{equation}
# $$

# %% [markdown]
# #### Inefficient python implementation

# %% [markdown]
# Let's see what output is obtained with this method on the loaded studies.

# %% [markdown]
# | Math variables      | Python variables          |
# |:-------------------:|:-------------------------:|
# |$B_n$                | `binary_arr`              |
# |$D_{p, n}*G_{S_n}$   | `prob_arrs[p, :, :, :]`   |
# |$\textit{MA}_n$      | `ma_maps[n, :, :, :]` |
# |$\textit{ALE}$       | `ale_arr`                 |
#
# <center>Link between math and python variables.</center>

# %%
# sigma is the width of the gaussian kernel used in the convolution of the 
# dirac maps to build the probability maps. Dimension is in voxel.

sigma = 2.

def compute_ale_ma_maps():
    """
    Use the following global variables
    N_img, Ni, Nj, Nk
    binary_imgs
    sigma
    """
    ma_maps = np.zeros((N_img, Ni, Nj, Nk))

    for n in range(N_img):
        print(f'{n+1} out of {N_img}', end='\r')
        binary_img = binary_imgs[n]
        
        # Create probability maps
        binary_arr = binary_img.get_fdata()
        nz_i, nz_j, nz_k = np.nonzero(binary_arr)
        prob_arrs = np.zeros((len(nz_i), Ni, Nj, Nk))
        
        for p in range(len(nz_i)):
            prob_arrs[p, nz_i[p], nz_j[p], nz_k[p]] = 1
            # Gaussian convolve
            prob_arrs[p, :, :, :] = scipy.ndimage.gaussian_filter(prob_arrs[p, :, :, :],
                                                                  sigma=sigma)

        # Merge probability maps
        ma_maps[n, :, :, :] = np.max(prob_arrs, axis=0)
        
    return ma_maps


def merge_ale_ma_maps(ma_maps):
    return 1 - np.prod(1-ma_maps, axis=0)

ale_ma_maps = compute_ale_ma_maps()
ale_arr = merge_ale_ma_maps(ale_ma_maps)
ale_img = nib.Nifti1Image(ale_arr, affine)

# %%
plotting.view_img(ale_img, threshold=0.01)

# %% [markdown]
# ### KDA

# %% [markdown]
# The Kernel Density Analysis (KDA) technique differs from ALE in the way peaks are considered. Whereas a probability map was derived from the peaks in ALE, in KDA, the value of interest is the peak density (namely the number of peaks lying in a sphere of a given radius).

# %% [markdown]
# #### Pipeline
# The KDA pipeline is quite similar to the ALE one, the few differences being the kernel and the way maps are merged together.
#
# The pipeline is summarized in figure [2](#KDA_fig) and can be described as follows:
#
# 1. First of all, gather the reported activation peaks in $N$ binary maps (one map per study).
# 2. Convolve each of these maps with a spherical uniform kernel to build the Modeled Activity (MA) maps.
# 3. Derive the KDA map by summing the MA maps together.

# %% [markdown]
# <a id='KDA_fig'></a>
# ![KDA](resource/KDA.png)

# %% [markdown]
# **Figure 2**: The KDA pipeline.

# %% [markdown]
# The first thing we see is that the KDA process is simpler than the ALE one. Indeed, in KDA, the merge operation number 1 of ALE (the non linear maximum element-wise) is replaced by a sum operation element-wise, which is linear. So, in KDA we can mix all these steps into a single convolution of the binary map.
#
# The second merge operation (#2 in figure [2](#KDA)) is also replaced by a sum element-wise.

# %% [markdown]
# Let's give a bit of interpretation of the different parts in KDA.

# %% [markdown]
# #### KDA Kernel

# %% [markdown]
# As previously stated, the KDA kernel is a sphere with a chosen radius $r$ of ones and zero else-where. Therefore, the result of a convolution of a dirac map (i.e. a zero-valued map with a single 1) with this kernel is simply a sphere of ones of same radius.  
# The derived map expresses a density. For each peak, the value stored in this map gives the number of peaks lying at most at a distance of $r$ from the chosen peak. In the particular case of a dirac map, the derived density is 0 when we look further than $r$ of the unique peak and 1 when we look closer than a $r$ distance from the unique peak.

# %% [markdown]
# #### MA maps

# %% [markdown]
# Following from the previous point, the result of a convolution of a binary map (i.e. a sum of dirac maps) with the KDA Kernel of radius $r$, expresses simply the density of the binary map with a radius $r$.

# %% [markdown]
# #### KDA map

# %% [markdown]
# Here again, since the merge operation #2 is just a sum, the KDA map expresses a density with a radius $r$ accross studies. That is to say the number of peaks lying at most at a distance of $r$.

# %% [markdown]
# #### Mathematical recap

# %% [markdown]
# - Let $(N_i, N_j, N_k)$ be the shape of the brain space. Typically $(N_i, N_j, N_k) = (91, 109, 91)$ in the MNI space.
# - For all $n \in [1 .. N],$
#     * let $B_n \in \{0, 1\}^{N_iN_jN_k}$ be the binary map associated with experiment $n$.
#     * let $\textit{MA}_n \in \mathbb{R}^{N_iN_jN_k}$ be the Modeled Activation map of experiment $n$.
#     * let $K_r$ be the uniform kernel of size $r$ associated with experiment $n$.
# - Let $\textit{KDA} \in \mathbb{R}^{N_iN_jN_k}$ be the desired KDA map.

# %% [markdown]
# $$
# \begin{equation}
#     \textit{MA}_n = B_n*K_r
# \end{equation}
# $$

# %% [markdown]
# $$
# \begin{equation}
#     \textit{KDA}(i, j, k) = \sum_{n=1}^N\textit{MA}_n(i, j, k)
# \end{equation}
# $$

# %% [markdown]
# #### Inefficient python implementation

# %% [markdown]
# Let's see what output is obtained with this method on the loaded studies.

# %% [markdown]
# | Math variables      | Python variables                                         |
# |:-------------------:|:--------------------------------------------------------:|
# |$B_n$                | `binary_arr`                                             |
# |$\textit{MA}_n$      | `prob_arrs[:, :, :]` <br> and `ma_maps[n, :, :, :]`      |
# |$\textit{KDA}$       | `kda_arr`                                                |
#
# <center>Link between math and python variables.</center>

# %%
r = 15. # mm

def uniform_kernel(r, affine):
    A, B, C = r, r, r
    
    a = int(A/abs(affine[0, 0]))
    b = int(B/abs(affine[1, 1]))
    c = int(C/abs(affine[2, 2]))
    
    kernel = np.zeros((2*a+1, 2*b+1, 2*c+1))
    
    i0, j0, k0 = a, b, c
    
    for i in range(2*a+1):
        for j in range(2*b+1):
            for k in range(2*c+1):
                if ((i-i0)/a)**2 + ((j-j0)/b)**2 + ((k-k0)/c)**2 <= 1:
                    kernel[i, j, k] = 1
                
    return kernel


def compute_kda_ma_maps():
    ma_maps = np.zeros((N_img, Ni, Nj, Nk))

    for n in range(N_img):
        print(f'{n+1} out of {N_img}', end='\r')
        binary_img = binary_imgs[n]
        
        # Create probability maps
        binary_arr = binary_img.get_fdata()
        
        prob_arrs = scipy.ndimage.filters.convolve(binary_arr, uniform_kernel(r, affine), mode='constant')

        # Merge probability maps
        ma_maps[n, :, :, :] = prob_arrs
        
    return ma_maps


def merge_kda_ma_maps(ma_maps):
    return np.sum(ma_maps, axis=0)


kda_ma_maps = compute_kda_ma_maps()
kda_arr = merge_kda_ma_maps(kda_ma_maps)
kda_img = nib.Nifti1Image(kda_arr, affine)

# %%
plotting.view_img(kda_img, threshold=25)

# %% [markdown]
# ### ALE and KDA limitations [1]

# %% [markdown]
# Both ALE and KDA treat each focus independently. As a consequence, if one study has a lot more peaks than the others, this study will have an immense impact on the results, even if the increased number of foci is not relevant and is not indicative of the quality of a study (e.g due to different thresholding). To prevent studies with a lot of peaks from getting the upper hand, MKDA technique does not treat all foci equally; instead, it considers study as the unit of analysis. 
# A more fundamental issue is that studies with no activations cannot be included in the meta analyses.  

# %% [markdown]
# ### MKDA

# %% [markdown]
# The Multilevel Kernel Density Analysis (MKDA) technique, as the name suggests, is closely related to the KDA one. Actually, it is an improved version of KDA in which the unit of analysis is no longer peak location but study. This allows it to avoid the drawbacks mentioned above.

# %% [markdown]
# #### Pipeline

# %% [markdown]
# The MKDA pipeline is  basically the KDA one with extra steps.
#
# The pipeline is summarized in figure [3](#MKDA_fig) and can be described as follows:
#
# 1. First of all, gather the reported activation peaks in $N$ binary maps.
# 2. Convolve each of these maps with a spherical uniform kernel.
# 3. Binarize these maps by setting all strictly positive values to 1 to build the Comparison Indicator Maps (CIM).
# 3. Derive the MKDA map by performing a weighted average on the CIM (weights are described later).

# %% [markdown]
# <a id='MKDA_fig'></a>
# ![MKDA](resource/MKDA.png)

# %% [markdown]
# **Figure 3**: The MKDA pipeline.

# %% [markdown]
# Let's describe these extra steps.

# %% [markdown]
# #### MKDA Kernel

# %% [markdown]
# The MKDA kernel is the same as the KDA kernel, namely, a sphere of ones with a chosen radius $r$ and zero elsewhere.

# %% [markdown]
# #### Binarization

# %% [markdown]
# In comparison to KDA, a new step is added between the convolution and the merge operations: binarization. The purpose of this operation is to prevent studies that have a lot of peaks from getting the upper hand over studies having only a few peaks, which is, as said before, a problem with ALE and KDA. 

# %% [markdown]
# #### Weighted average

# %% [markdown]
# The purpose of the average by itself is intuitive, but what about the weighting?  
#
# 1. We want the larger studies to carry more weight in the meta-analysis because it is assumed that the larger the sample, the more precise the results.  
# 2. Moreover, two main kinds of studies exist in the literature: those that treat subjects as fixed effects (FFX, mostly older studies), and those subjecs as random effects (RFX, which uses the appropriate degrees of freedom). Thus, the idea is to downweight the first kind of studies because their results may be less generalizable beyond their sample. There is no agreed-upon value for this downweighting, but some papers propose 0.75 ([1]).

# %% [markdown]
# In order to meet these needs, a reasonable weight for an experiment $n$ is the following:
#
# $$
#     \alpha_n = \delta_n\sqrt{S_n}
# $$
#
# With
# - $S_n$ is the number of subjects in the experiment $n$.
# - $\delta_n$ the RFX/FFX weight given to experiment $n$ :
#
# $$\delta_n =  \left\{ 
#     \begin{array}[cc]\\
#         1 & \mbox{if RFX}\\
#         .75 & \mbox{if FFX}
#     \end{array}
#     \right.
# $$

# %% [markdown]
# #### Mathematical recap

# %% [markdown]
# - Let $N \in \mathbb{N}^*$ be the number of experiments/studies.
# - Let $S_n \in \mathbb{N}$ be the number of subjects in the experiment $n$.
# - Let $(N_i, N_j, N_k)$ be be the shape of the brain space. Typically $(N_i, N_j, N_k) = (91, 109, 91)$ in MNI space.
# - For all $n \in [1 .. N],$
#     * let $B_n \in \{0, 1\}^{N_iN_jN_k}$ be the binary map associated with experiment $n$.
#     * let $\textit{MA}_n \in \mathbb{R}^{N_iN_jN_k}$ be the Modeled Activation map of experiment $n$.
#     * let $K_r$ be the uniform kernel of size $r$ associated with experiment $n$.
# - Let $\textit{MKDA} \in \mathbb{R}^{N_iN_jN_k}$ be the desired MKDA map.

# %% [markdown]
# $$
# \begin{equation}
#     \textit{MA}_n(i, j, k) = \min((B_n*K_r)(i, j, k),~1)
# \end{equation}
# $$

# %% [markdown]
# $$
# \begin{equation}
#     \textit{MKDA}(i, j, k) = \frac{1}{\sum_{n=1}^N\alpha_n}\sum_{n=1}^N\alpha_n\textit{MA}_n(i, j, k)
# \end{equation}
# $$

# %% [markdown]
# With:
# $$
#     \alpha_n = \delta_n\sqrt{S_n}
# $$

# %% [markdown]
# $$\delta_n =  \left\{ 
#     \begin{array}[cc]\\
#         1 & \mbox{if RFX}\\
#         .75 & \mbox{if FFX}
#     \end{array}
#     \right.
# $$

# %% [markdown]
# #### Inefficient python implementation

# %% [markdown]
# Let's see what output is obtained with this method on the loaded studies.

# %% [markdown]
# | Math variables      | Python variables                                         |
# |:-------------------:|:--------------------------------------------------------:|
# |$B_n$                | `binary_arr`                                             |
# |$\textit{MA}_n$      | `prob_arrs[:, :, :]` <br> and `ma_maps[n, :, :, :]`      |
# |$\textit{MKDA}$      |  `mkda_arr`                                              |
#
# <center>Link between math and python variables.</center>

# %%
r = 10. # radius of the sphere, mm

def compute_mkda_ma_maps():
    ma_maps = np.zeros((N_img, Ni, Nj, Nk))

    for n in range(N_img):
        print(f'{n+1} out of {N_img}', end='\r')
        binary_img = binary_imgs[n]
        
        # Create probability maps
        binary_arr = binary_img.get_fdata()
        
        prob_arrs = scipy.ndimage.filters.convolve(binary_arr, uniform_kernel(r, affine), mode='constant')
        
        prob_arrs = prob_arrs > 0 

        # Merge probability maps
        ma_maps[n, :, :, :] = prob_arrs
        
    return ma_maps


def merge_mkda_ma_maps(ma_maps):
    return np.sum(ma_maps, axis=0)


mkda_ma_maps = compute_mkda_ma_maps()
mkda_arr = merge_mkda_ma_maps(mkda_ma_maps)
mkda_img = nib.Nifti1Image(mkda_arr, affine)

# %%
plotting.view_img(mkda_img, threshold=5)


# %% [markdown]
# ## Coordinate-based meta-analysis limitations

# %% [markdown]
# The main limitation of coordinate-based meta-analysis is the loss of information between the full image of the brain (which is acquired during the experiment) and the reported coordinates in the published paper. Another limitation of this representation is the high sensitivity of coordinates to methods used in the study, from thresholding to reporting.  
# That's why, whenever the full images of the brain is available, one should use image-based meta-analysis to obtain results based on more information. 

# %% [markdown]
# ## Image-based meta-analysis (IBMA)

# %% [markdown]
# In image-based meta-analysis, as the name suggests, we have at our disposal at least the full group-level activation image from each study, and often the full activation image for each subject within each study. The idea of image-based meta-analysis is to mix these maps together to create a meta-analysis map.
#
# We will go through three methods: Fisher, Stouffers and Multilevel GLM. These methods differ in the way they combine images. Fisher and Stouffers are general methods that are used to combine a number of independant tests, whereas Multilevel GLM is more neuroimaging-specific in the way the operations are conceived.
#
# This part has been written using the following references: [3] [4] [5].

# %% [markdown]
# **Context**  
# What precisely are the inputs of image-based meta-analysis?  
# The type of map (contrast, z, t or standard error maps) depends on the method used, and will be detailed in each section.
#
# However, whatever the type of map, one should be aware about the following caveat.  
# Since the input data is now the full brain image instead of coordinates (that are assumed to be in a common space), one should check that all images are in a common space (e.g. the [MNI152](http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009)).

# %% [markdown]
# ### Fisher

# %% [markdown]
# Fisher's combination method, as Stouffers' method, aims to combine a number of independant tests into one.

# %% [markdown]
# **Input data**  
# Only the group-level z maps from each of the studies are required.
#
# **Ouput data**  
# The ouput of the method is a single p-value map. A z map can be computed from this map, seen as the meta-analysis result.

# %% [markdown]
# #### Mathematic foreword
# Before applying the Fisher's method to neuroimaging, let's consider the general case.
#
# Let $T_1, ... T_N$ be $N$ independant tests on $H_0$.  
# Let $p_1, ..., p_N$ be their corresponding p-values.

# %% [markdown]
# The Fisher's method builds the test statistics $T_f$ as follows:
# $$
#     T_f = -2\sum_{k=1}^N\ln(p_k)
# $$

# %% [markdown]
# An important result is that, under the null hypothesis $H_0$, $T_f$ follows a $\chi^2$ distribution with $2N$ degrees of freedom. Thus, the only remaining thing to do is to compare the obtained value of $T_f$ to the null $\chi^2_{2N}$ distribution. To control for a risk of error $\alpha$, one would chose a threshold $\chi^2_{2N\alpha}$ value such that the propability to find a $\chi^2_{2N}$ greater or equal to $\chi^2_{2N\alpha}$ is $\alpha$.
# Large values of $T_f$, less likely according to the $\chi^2_{2N}$ distribution, lead to the rejection of $H_0$.
#
# The p-value $p_f$ of the combined probability test can then be computed by looking to the tail of the $\chi^2_{2N}$ distribution.
#
# $$
#     p_f = \mathbb{P}(X \geq T_f)
# $$
# With
# $$
#     X \sim \chi^2_{2N}
# $$

# %% [markdown]
# <img src="resource/chi2.png" width="400"> \
# **Figure 4:** The $\chi^2$ distribution.

# %% [markdown]
# #### Neuorimaging application
#
# The key idea is to understand how to transform z maps into p-value maps and vice versa.
#
# **Z map to p-value map**  
# Z maps store z-scores. For a voxel $v$, the null hypothesis is "the voxel is not activated." When $H_0$ is true, the z-score of this voxel follows a standard normal distribution. Since we are interested in activation rather than deactivation, given a z-score $Z$, its associated p-value $p_Z$ is the probability to observe a value greater than the test critical value.

# %% [markdown]
# $$
# p_Z = \mathbb{P}(X \geq Z)
# $$
# With
# $$X \sim \mathcal{N}(0, 1)$$
#
# Which can be rewritten using the normal cumulative distribution function $\Phi$, and gives the link between z score and its p-value:
# $$
#     p_Z = 1 - \Phi(Z)
# $$

# %% [markdown]
# <img src="resource/normal.jpg" width="300">
#
# <a id='Normal'></a>**Figure 5:** The normal distribution of the z map.

# %% [markdown]
# Considering only activation (the right side of the distribution) and not deactivation is a one-sided test. Considering both of them (answering a more general question of excessive positive or negative values) is referred to as a two-sided test.

# %% [markdown]
# **p-value map to Z map**  
# Using the inverse normal cumulative distribution function, it is straightforward to invert the previous process. Given a p-value $p$, the associated critical value is:

# %% [markdown]
# $$
#    Z = \Phi^{-1}(1-p_Z)
# $$

# %% [markdown]
# **Application**  
# Let's consider only one brain voxel $v$. In each study $k$, the question of whether $v$ is activated, i.e., whether we reject $H_0$, is represented by the hypothesis test $T_k$. The studies are supposedly independant; so are the tests $T_k$. For each study $k$, we can derive the p-value $p_k$ from the z-score associated to voxel $v$ as described previously. We then combine all these p-values together to build a test $T_f$ using the Fisher's formula. We compare $T_f$ to the $\chi^2_{2N}$ distribution and derive the combined p-value $p_f$. Finally, we derive the meta-analysis z-score from the combined p-value $p_f$ as described above.  
# By performing this task voxel-wise, we can derive the meta-analysis z map.

# %% [markdown]
# #### Inefficient python implementation

# %% [markdown]
# Let's see what output is obtained with this method on the loaded studies.

# %% [markdown]
# | Math variables      | Python variables|
# |---------------------|-------------|
# |$T_f$                | `T_f`       |
# |$p_f$                | `p_f`       |
# |$\Phi^{-1}(1-p_f)$   | `fisher_arr`|
#
# <center>Link between math and python variables.</center>

# %%
def z_to_p(z_arr):
    return 1-scipy.stats.norm.cdf(z_arr)

def p_to_z(p_arr):
    """
    scipy.special.ndtri returns the argument x for which the area under the Gaussian 
    probability density function (integrated from minus infinity to x) is equal to y.
    """
    return scipy.special.ndtri(1-p_arr)

p_values = np.zeros((N_img, Ni, Nj, Nk))

for n in range(N_img):
    img = nilearn.image.load_img(img_paths[n])
    arr = img.get_fdata()
    p_values[n, :, :, :] = z_to_p(arr)
    
T_f = np.nan_to_num(-2*np.sum(np.log(p_values), axis=0))

# chi2 survival funtion: 
p_f = scipy.stats.chi2.sf(T_f, 2 * N_img)

fisher_arr = p_to_z(p_f)
maxnan = fisher_arr[fisher_arr != np.inf].max()
minnan = fisher_arr[fisher_arr != -np.inf].min()
np.nan_to_num(fisher_arr, copy=False, posinf=maxnan, neginf=minnan) # Truncate inf values
fisher_img = nib.Nifti1Image(np.array(fisher_arr), affine)

# %%
plotting.plot_stat_map(fisher_img)

# %% [markdown]
# ### Stouffers

# %% [markdown]
# Stouffers' method is very close to Fisher's one. Both have z maps as inputs and outputs and both aim to mix p-value maps into one.

# %% [markdown]
# **Input data**  
# Only the z maps of each studies are required.
#
# **Ouput data**  
# The ouput of the method is a single p-value map. From this map can be computed a z map, seen as the meta-analysis result.

# %% [markdown]
# **Mathematic foreword**
#   
# Let $Z_1, ..., Z_N$ be $N$ Z-scores. The Stouffers' method aims to build a test $T_s$ as follow:

# %% [markdown]
# $$
# T_s = \sum_{k=1}^N \frac{Z_k}{\sqrt{N}}
# $$

# %% [markdown]
# Note that since $Z_k = \Phi^{-1}(1-p_k)$ with the previous notations, it can be seen as a p-value combination method, as Fisher's one.

# %% [markdown]
# Under $H_0$, since $Z_k \sim \mathcal{N}(0, 1)$, $T_s$ follows a standard normal distribution
# $$
#     T_s \sim \mathcal{N}(0, 1)
# $$

# %% [markdown]
# From there, the process is similar to the one explained in the Fisher's part, replacing the $\chi^2_{2N}$ distribution with the $\mathcal{N}(0, 1)$ one. 

# %% [markdown]
# The p-value $p_s$ of the combined probability test is similarly given by
#
# $$
#     p_s = \mathbb{P}(X \geq T_s)
# $$
# With
# $$
#     X \sim \mathcal{N}(0,1)
# $$

# %% [markdown]
# #### Application
# As with Fisher's method, let's consider only one brain voxel $v$. For each study $k$, we combine the Z-scores together to build a test $T_s$. The p-value $p_s$ and its associated Z-score are then derived by comparing $T_s$ to the $\mathcal{N}(0, 1)$ distribution. By performing this task voxel-wise, we can derive the meta-analysis z map.

# %% [markdown]
# #### Inefficient python implementation

# %% [markdown]
# Let's see what output is obtained with this method on the loaded studies.

# %% [markdown]
# | Math variables      | Python variables|
# |---------------------|-----------------|
# |$T_s$                | `T_s`           |
# |$p_s$                | `p_s`           |
# |$\Phi^{-1}(1-p_s)$   | `stouffers_arr` |
#
# <center>Link between math and python variables.</center>

# %%
T_s = np.zeros((Ni, Nj, Nk))

for n in range(N_img):
    img = nilearn.image.load_img(img_paths[n])
    T_s += img.get_fdata()
    
T_s /= np.sqrt(N_img)
p_s = scipy.stats.norm.sf(T_s, loc=0, scale=1)

stouffers_arr = p_to_z(p_s)
maxnan = stouffers_arr[stouffers_arr != np.inf].max()
minnan = stouffers_arr[stouffers_arr != -np.inf].min()
np.nan_to_num(stouffers_arr, copy=False, posinf=maxnan, neginf=minnan) # Truncate inf values
stouffers_img = nib.Nifti1Image(stouffers_arr, affine)

# %%
plotting.view_img(stouffers_img)


# %% [markdown]
# ### Fisher's or Stouffers'?

# %% [markdown]
# Here are some comparisons found in the litterature:
#
# Stouffer’s method is commonly preferred in the presence of weights, as the weighted test statistic
# $\sum^n_{i=1} w_i\phi^{−1}(p_i)$ retains a closed-form distribution, $\mathcal{N}(0,\sum_{i=1}^n w_i)$, under the null hypothesis [6]. This straightforward way to introduce weights can make Stouffer's method more powerful than Fisher’s method when the p-values are from studies of different size [7] [8].

# %% [markdown]
# ### Multilevel GLM

# %% [markdown]
# The last technique to be approached is also the most complex: the Multilevel General Linear Level. Below is the general idea of a simplified version of the method, where a strong hypothesis has been made.  
#
# The key idea of this technique is to add a level of GLM on top of the other usual levels, that is, adding an inter-study level GLM on top of subject level and within-study level GLM.
#
# This part strongly references the paper of Beckmann et al [4].

# %% [markdown]
# <a id="GLM_Model"></a>
# #### GLM Model
# GLM intempts to explain a variable $Y$ as a combination of $\beta$-weighted predictors $X$ and noise $e$. The $\beta$ weights are the unknown values that are estimated. The simplest modelisation of GLM is the following:
#
# $$
#     Y = X\beta + e
# $$
#
# $$e \sim \mathcal{N}(0, \sigma^2 I)$$
#
# Where $Y$, $X$, $e$ are vectors and $X$ is a matrix.  
# - $X$ is called the design matrix and is fixed by the experimenter (assumed full rank).  
# - $Y$ are observations.
# - $\beta$ is unknown and will be estimated in the GLM process.
# - $e$ is unknown, but assumptions are made on its covariance structure (e.g independant and identically distributed...)

# %% [markdown]
# **1. Estimation of $\beta$**  
# Under the more general assumption $e \sim \mathcal{N}(0, V)$, $\beta$ and its covariance can be estimated using the Generalized Least Square (GLS) technique through the following formulae:
#
# $$
#     \hat{\beta} = (X^TV^{-1}X)^{-1}X^TV^{-1}Y
# $$
# $$
#     \mbox{Cov}(\hat{\beta}) = (X^TV^{-1}X)^{-1}
# $$
#
# Under the particular assumption $V = \sigma^2I$, i.e $e \sim \mathcal{N}(0, \sigma^2I)$, the above formulae become:
# $$
#     \hat{\beta} = (X^TX)^{-1}X^TY
# $$
# $$
#     \mbox{Cov}(\hat{\beta}) = \sigma^2(X^TX)^{-1}
# $$
#
# This case of GLS is known as Ordinary Least Squares (OLS)

# %% [markdown]
# **2. Estimation of $V$**  
# Remember that $V$ is unknown in the above process, hence one needs to estimate $V$ before applying these formulae.
#
# Under the simplest assumption $e \sim \mathcal{N}(0, \sigma^2 I)$, the covariance structure can be estimated as follow:
#
# $$
#     \hat{\sigma} = \frac{1}{N}(y-X\hat{\beta})^T(y-X\hat{\beta})
# $$
#
# With $N$ the length of $y$.
#
# Note that this is possible since $\hat{\beta}$ does not depend on $V$ under the assumption $V = \sigma^2I$. Once $\sigma$ estimated, can be estimated the covariance of the estimates: $\mbox{Cov}(\hat{\beta}) = \hat{\sigma}^2(X^TX)^{-1}$.
#
# Under the more general assumption $e \sim \mathcal{N}(0, V)$, $\beta$ does depend on $V$ and therefore, more complex method must be used. Beckmann et al. [4] present some of them.

# %% [markdown]
# #### 2-level GLM model
#
# The principle of Multilevel GLM is to stack the previous process for each level (e.g subject, within study, inter-study...).  
# As a simplified version, we consider only the upper two levels (within study and inter study levels, which are now called 1st and 2nd level) and we take the subject level for ground truth instead of estimation. See figure [6](#MGLM_simplified_fig) for diagram.

# %% [markdown]
# <a id='MGLM_simplified_fig'></a>
# ![MGLM](resource/Hierarchical_model_simplified.png)

# %% [markdown]
# **Figure 6:** Simplified 2-level multilevel GLM diagramm where subjects' maps are taken as ground truth (i.e not as estimates from time series).

# %% [markdown]
# Without loss of generality, we take ony one voxel $v$ into consideration. Let's formalize the simplified Multilevel GLM on each level.

# %% [markdown]
# **1st level: Within study**  
#
# For a study $k$,
# * Let $n_k$ be the number of subjects who participated in study $k$.
# * Let $Y_k$ be the vector of the values taken by the voxel $v$ for each subjects of experiment $k$. Hence $Y_k$ has same length as the number of subjects in study $k$.
# * Let $X_k$ be the design matrix of experiment $k$.
# * Let denote $\epsilon_k$ the residuals of the fit.

# %% [markdown]
# According to the first part [GLM Model](#GLM_Model), the GLM for experiment $k$ consists of estimating $\beta_k^{(1)}$ in the following equation, with one equation for each voxel:
#
# $$
#     Y_k = X_k\beta_k^{(1)} + \epsilon_k
# $$
#
# In our simple model, we assume that the noise is independant between subjects and has the following covariance structure:
#
# $$
#     \epsilon_k \sim \mathcal{N}(0, \sigma_k^2I_{n_k})
# $$
#
# As showed above, the ordinary least square technique can be used to derive the estimates of the first level $\hat{\beta}^{(1)}_k$. We have for each study $k$:
#
# $$
#     \hat{\beta}^{(1)}_k = (X_k^TX_k)^{-1}X_k^TY
# $$
# $$
#     \mbox{Cov}(\hat{\beta}^{(1)}_k) = \sigma_k^2(X_k^TX_k)^{-1}
# $$

# %% [markdown]
# Let's stack these equations together.  
# We define
# $
# Y = \begin{bmatrix}
# Y_1 \\
# ... \\
# Y_N
# \end{bmatrix}$
# $
# X = \begin{bmatrix}
# X_1 & 0 & 0\\
# 0 & ... & 0\\
# 0 & 0 & X_N
# \end{bmatrix}$
# $
# \beta^{(1)} = \begin{bmatrix}
# \beta_1^{(1)} \\
# ... \\
# \beta_N^{(1)}
# \end{bmatrix}$
# $
# \epsilon = \begin{bmatrix}
# \epsilon_1 \\
# ... \\
# \epsilon_N
# \end{bmatrix}$
# $
# V = \begin{bmatrix}
# \sigma_1I_{n_1} & 0 & 0\\
# 0 & ... & 0\\
# 0 & 0 & \sigma_NI_{n_N}
# \end{bmatrix}$
#
# Hence the previous equation becomes:
# $$
#     \hat{\beta}^{(1)} = (X^TX)^{-1}X^TY
# $$
# $$
#     \mbox{Cov}(\hat{\beta}^{(1)}) = (X^TV^{-1}X)^{-1}
# $$
#
# Which are the estimates of the first level and their covariance (Respectively the studies' red/blue maps and their yellow variance maps on figure [6](#MGLM_fig)).

# %% [markdown]
# **2nd level: Inter-study**

# %% [markdown]
# As said before, multilevel GLM is the stacking of single GLMs. Hence for the 2nd level we just stack another level of GLM on the previous results. More formally,
#
# $$
#     \beta^{(1)} = X_G\beta^{(2)} + \eta
# $$
# Where
# * $X_G$ is the group level design matrix.
# * $\eta$ is the vector of residuals of the fit.
#
# With, here again, the assumption is made that $\eta$ has a diagonal covariance structure $V_G = \mbox{Cov}(\eta) = \sigma^2I$.
#
#

# %% [markdown]
# The problem here is that $\beta^{(1)}$ is unknown, we only know its estimation $\hat{\beta}^{(1)}$. Hence, we fit a GLM on the estimates $\hat{\beta}^{(1)}$ instead. The previous equation hence becomes
#
# $$
#     \hat{\beta}^{(1)} = X_G\beta^{(2)} + \eta'
# $$
#
# Keepping the same notation for the quantity we want to estimate $\beta^{(2)}$. We denote $V_{G2} := \mbox{Cov}(\eta')$. Note that we cannot make any assumption about the noise $\eta'$ since we are now fitting GLM on estimates. However, we can derive a formula between the covariance of $\eta'$ and other covariances of the problem (see [4] for details).
#
# $$
#     \mbox{Cov}(\eta') = \mbox{Cov}(\eta) + \mbox{Cov}(\hat{\beta}^{(1)})
# $$
# Hence
# $$
#     V_{G2} := \mbox{Cov}(\eta') = \sigma^2I_N + (X^TV^{-1}X)^{-1}
# $$

# %% [markdown]
# As showed in the [GLM Model](#GLM_Model) part, applying the general least squares technique leads to
#
# $$
#     \hat{\beta}^{(2)} = (X_G^TV_{G2}^{-1}X_G)^{-1}X_G^TV_{G2}^{-1}Y
# $$
# $$
#     \mbox{Cov}(\hat{\beta}^{(2)}) = (X_G^TV_{G2}^{-1}X_G)^{-1}
# $$

# %% [markdown]
# **Covariance estimation**  
# Note that all the previous equations use the real covariance matrices $V_{G2}, V_G, V$. These matrices are unkown and need to be estimated to be able to use the previous equations. These estimations are complex and several techniques, listed in [4], can be used.

# %% [markdown]
# #### 3-level GLM model
#
# A 3-level GLM use the exact same idea as 2-level GLM except that the subjects' maps are now estimated from time series instead. The process is summarized in figure [7](#MGLM_fig).

# %% [markdown]
# <a id='MGLM_fig'></a>
# ![MGLM](resource/Hierarchical_model.png)

# %% [markdown]
# **Figure 7:** 3-level multilevel GLM diagram, where subjects' maps are estimated from time series in a voxel-wise manner. The yellow variance maps represent the variance of the estimates at each level of the GLM.

# %% [markdown]
#
# To see what results are obtained from multilevel GLM, please see the [NiMARE](#NIMARE) use.

# %% [markdown]
# ## Hypothesis testing

# %% [markdown]
# Although more interesting than individual maps, the meta-analysis maps derived from the previous methods don't show their full potential. It remains to differentiate the true signal of the noise given a tolerance $\alpha$ using hypothesis testing. One way of doing this is to use the permutation testing method (note that other methods exist such as Monte Carlo simulations).

# %% [markdown]
# #### Permutation testing
# The key idea of permutation testing is to permute the labels of the experiments to generate a null distribution. From the former can be derived an appropriate threshold at the desired tolerance rate.
#
# **Which labels are we talking about?**  
#
# Let's say you have a number $N_a$ of studies related to the same topic (e.g language) and a number $N_b$ of studies dealing with varied other topics. To generate a null distribution, permute randomly the labels of the studies (e.g language related and non language related studies) and apply the same meta-analysis method on the newly created set of $N_a$ studies labelled as your initial topic (e.g language). By doing so a large number of times, the null distribution is derived.
#
# **What if I don't have any studies unrelated to my topic of interest?**
#
# A way of using the permutation testing method even with studies on a single topic is to generate false unrelated studies by taking the opposite of the contrasts (since contrasts follows a normal distribution under the null hypothesis, which is sign invariant). Instead of permuting labels with another set of unrelated studies that we don't have, we randomly flip the sign of studies in our original set. Then, we apply the same meta-analysis method on the newly created set of studies. By doing so a large number of times, the null distribution is derived.
#
# **What do we do after obtaining the null distribution?**  
#
# Once the null distribution is obtained, two well-known error controls are the False Discovery Rate (FDR) and Family Wise Error Rate (FWER). Whether to prefer the FDR or FWER control rates depends on the meta-analysis method used and will be discussed later.
#
# **FDR or FWER?**
# Here are some recommandations found in the litterature [1].
#
# | Method | Error rate  |
# |:--------:|:-------------:|
# |ALE     | FDR         |
# |KDA     | FWER        |
# |MKDA    | FWER (recommended) or FDR|
#

# %% [markdown]
# <a id='NIMARE'></a>
# # Meta-analysis using NiMARE [5]

# %% [markdown]
# The purpose of this section is to show the use of NiMARE to perform the meta-analysis techniques we just presented in the notebook.

# %% [markdown]
# ## 1. Build the NiMARE.Dataset object

# %% [markdown]
# In order to be able to use the NiMARE package, we need a little more work done on the data to be accepted as input by NiMARE. The following functions build the dictionnary input used buy NiMARE. 

# %%
# The following functions are used only to convert the input images to a NiMARE input format for IBMA analysis. 
# These functions also extract peaks coordinates from full images for CBMA analysis.
# The understanding of these functions is not crucial.
def get_sub_dict(XYZ, path_dict, sample_size):
    """
    Build sub dictionnary of a study using the nimare structure.

    Args:
        XYZ (tuple): Size 3 tuple of list storing the X Y Z coordinates.
        path_dict (dict): Dict which has map name ('t', 'z', 'con', 'se')
            as keys and absolute path to the image as values.
        sample_size (int): Number of subjects.

    Returns:
        (dict): Dictionary storing the coordinates for a
            single study using the Nimare structure.

    """
    d = {
        'contrasts': {
            '0': {
                'metadata': {'sample_sizes': 119}
            }
        }
    }

    if XYZ is not None:
        d['contrasts']['0']['coords'] = {
                    'x': list(XYZ[0]),
                    'y': list(XYZ[1]),
                    'z': list(XYZ[2]),
                    'space': 'MNI'
                    }
        d['contrasts']['0']['sample_sizes'] = sample_size

    if path_dict is not None:
        d['contrasts']['0']['images'] = path_dict

    return d


def extract_from_paths(paths, sample_size, data=['coord', 'path'],
                       threshold=1.96):
    """
    Extract data (coordinates, paths...) from the data and put it in a
        dictionnary using Nimare structure.

    Args:
        path_dict (dict): Dict which keys are study names and values
            absolute paths (string).
        data (list): Data to extract. 'coord' and 'path' available.
        sample_size (int): Number of subjects in the experiment. 
        threshold (float): value below threshold are ignored. Used for
            peak detection.

    Returns:
        (dict): Dictionnary storing the coordinates using the Nimare
            structure.

    """

    # Computing a new dataset dictionary
    def extract_pool(path):
        """Extract activation for multiprocessing."""
        print(f'Extracting {path}...')

        XYZ = None
        if 'coord' in data:
            XYZ = get_activations(path, threshold, space='pos')
            if XYZ is None:
                return

        if 'path' in data:
            base, filename = ntpath.split(path)
            file, ext = filename.split('.', 1)

            path_dict = {'z': path}
            for map_type in ['t', 'con', 'se']:
                file_path = f'{base}/{file}_{map_type}.{ext}'
                if os.path.isfile(file_path):
                    path_dict[map_type] = file_path

            return get_sub_dict(XYZ, path_dict, sample_size)

        if XYZ is not None:
            return get_sub_dict(XYZ, None, sample_size)

        return

    n_jobs = multiprocessing.cpu_count()
    res = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(extract_pool)(path) for path in paths)

    # Removing potential None values
    res = list(filter(None, res))
    
    # Merging all dictionaries
    return {k: v for k, v in enumerate(res)}


# %%
ds_dict = extract_from_paths(img_paths, data=['path', 'coord'], sample_size=119, threshold=1.96)

# %%
print(ds_dict)


# %% [markdown]
# ## Run meta-analysis on data

# %%
def run_ALE(ds_dict):
    """Run ALE on given data."""
    ds = Dataset(ds_dict)
    ma = nimare.meta.cbma.ale.ALE()
    res = ma.fit(ds)

    img_ale = res.get_map('ale')
    img_p = res.get_map('p')
    img_z = res.get_map('z')

    return img_ale, img_p, img_z


def run_KDA(ds_dict):
    """Run KDA on given data."""
    ds = Dataset(ds_dict)
    ma = nimare.meta.cbma.mkda.KDA()
    res = ma.fit(ds)

    return res.get_map('of')


def run_MKDA(ds_dict):
    """Run KDA on given data."""
    ds = Dataset(ds_dict)
    ma = nimare.meta.cbma.mkda.MKDADensity()
    res = ma.fit(ds)

    return res.get_map('of')


def run_MFX_GLM(ds_dict):
    """Run MFX_GLM on given data."""
    # Check if FSL is installed
    if os.environ.get('FSLDIR', None) is None:
        warnings.warn('FSL not installed, skipping MFX_GLM.')
        return None
    
    # Delete NiMare's temporary mfx_glm folder if it is in the current directory.
    # NiMare makes this folder while running the MFX_GLM() function, and it 
    # deletes the folder afterwards. But if you interrupt the process, the folder
    # will not be deleted, and it will not run again until it is deleted.   
    folder = './mfx_glm'
    if osp.exists(folder) and osp.isdir(folder):
        shutil.rmtree(folder)
    
    ds = Dataset(ds_dict)
    ma = nimare.meta.ibma.MFX_GLM()
    res = ma.fit(ds)

    return res.get_map('t')


def run_Fishers(ds_dict):
    """Run Fishers on given data."""
    ds = Dataset(ds_dict)
    ma = nimare.meta.ibma.Fishers()
    res = ma.fit(ds)

    return res.get_map('z')


def run_Stouffers(ds_dict):
    """Run Stouffers on given data."""
    ds = Dataset(ds_dict)
    ma = nimare.meta.ibma.Stouffers()
    res = ma.fit(ds)

    return res.get_map('z')


def fdr_threshold(img_list, img_p, q=0.05):
    """Compute FDR and threshold same-sized images."""
    arr_list = [copy.copy(img.get_fdata()) for img in img_list]
    arr_p = img_p.get_fdata()
    aff = img_p.affine

    fdr = nimare.stats.fdr(arr_p.ravel(), q=q)

    for arr in arr_list:
        arr[arr_p > fdr] = 0

    res_list = [nib.Nifti1Image(arr, aff) for arr in arr_list]

    return res_list


# %%
img_ale, img_p, img_z = run_ALE(ds_dict)
img_kda = run_KDA(ds_dict)
img_mkda = run_MKDA(ds_dict)
img_z_F = run_Fishers(ds_dict)
img_z_S = run_Stouffers(ds_dict)
img_t_MFX = run_MFX_GLM(ds_dict)

# %%
img_ale_thr, img_p_thr, img_z_thr = fdr_threshold([img_ale, img_p, img_z], img_p)

# %% [markdown]
# ## Print results

# %%
meta_analysis = { 
    'ALE': img_ale,
    'ALE thresholded': img_ale_thr,
    'KDA': img_kda,
    'MKDA': img_mkda,
    'z Fishers': img_z_F,
    'z Stouffers': img_z_S,
    't MFX': img_t_MFX,
}

# %%
for name, img in meta_analysis.items():
    if img is not None:
        plotting.plot_stat_map(img, title=name, figure=plt.figure(figsize=(10,5)))

# %% [markdown]
# # References

# %% [markdown]
# [1] Meta-analysis of functional neuroimaging data: current and future directions  
# Tor D. Wager, Martin Lindquist, and Lauren Kaplan
#
# [2] Meta-analysis of neuroimaging data: A comparison of image-based and coordinate-based pooling of studies  
#     Gholamreza Salimi-Khorshidi, Stephen M. Smith, John R. Keltner, Tor D. Wager, Thomas E. Nichols 
#     
# [3] Combining Brains: A Survey of Methods for Statistical Pooling of Information  
#     Nicole A. Lazar, Beatriz Luna, John A. Sweeney, and William F. Eddy
#     
# [4] General multilevel linear modeling for group analysis in FMRI  
# Christian F. Beckmann, Mark Jenkinson, and Stephen M. Smith
#
# [5] NiMARE: https://github.com/neurostuff/NiMARE
#
# [6] Choosing between methods of combining p-values, N. A. Heard, P. Rubin-Delanchy
#
# [7] Whitlock, M. C. “Combining probability from independent tests: the weighted Z-method is superior to Fisher’s approach.” Journal of Evolutionary Biology 18, no. 5 (2005): 1368-1373.
#
# [8] Zaykin, Dmitri V. “Optimally weighted Z-test is a powerful method for combining probabilities in meta-analysis.” Journal of Evolutionary Biology 24, no. 8 (2011): 1836-1841.
