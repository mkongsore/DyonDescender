# DyonDescender
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXX%20-green.svg)](https://arxiv.org/abs/2506.14867)

This GitHub page is the home of DyonDescender; the simple field relaxation code described in [2506.14867](https://arxiv.org/abs/2506.14867) capable of minimizing the action associated with dyon field configurations in the $SU(2)$ Georgi-Glashow model.

![RingFlux](/dyon_cross_section_R=0.4.png "Cross section of a relaxed Euclidean dyon loop field configuration, showing the associated E and B fields.")

If this pipeline is used in published work, please cite [2506.XXXXX](https://arxiv.org/abs/2505.14867).

## Contents

The two relevant files in this repository are:
1) `relaxation_jax.py`
2) `relaxation_notebook.ipynb`

The first is a `.py` script containing the full relaxation algorithm, implemented in jax. This script can be run locally or on a cluster, provided the file paths are appropriately updated.

The second is a `.ipynb` notebook spelling out the individual steps in our numerical procedure, starting from constructing the ansatz and grid, and ending with the relaxation procedure output. This notebook contains both a numpy and a (faster) jax implementation of the main numerical relaxation loop. Note, however, that the procedure is too slow to be run in full in a Jupyter notebook, which serves as an illustrative example only.

## Authors

- Isabel Garcia Garcia
- Marius Kongsore
- Ken Van Tilburg
