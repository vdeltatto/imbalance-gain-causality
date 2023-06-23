# imbalance-gain-causality

This repository contains the supporting codes for the article **Robust inference of causality in high-dimensional dynamical processes from the Information Imbalance of distance ranks** ([arXiv](https://doi.org/10.48550/arXiv.2305.10817)).

The functions employed to compute the Imbalance Gain in the paper are in the Python modules _utilities.py_ and _imbalance_gain.py_, and their use is illustrated in the notebook _tutorial.ipynb_. The Information Imbalance for the causality tests can also be computed using the [DADApy package](https://github.com/sissa-data-science/DADApy), as shown in the same tutorial.

The subdirectory _dynamical-systems_ contains the scripts to generate the trajectories of the dynamical systems analyzed in the paper.

The codes require installing the packages NumPy (1.24.2), Matplotlib (3.7.0), SciPy (1.10.1), scikit-learn (1.2.1), Joblib (1.2.0) and DADApy (0.2.0). 
