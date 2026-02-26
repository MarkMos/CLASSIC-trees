# CLASSIC-trees

#### !!!CLASSIC-trees is still undergoing development and tests. It is not yet ready for science use!!!

The code is developed primarily by [Markus Braun](https://github.com/MarkusBraunPhys) under the supervision of Markus R. Mosbech.

CLASSIC-trees is new public code to generate dark matter halo merger trees directly from running CLASS, for use with semi-analytical galaxy modeling codes, relying on a Monte-Carlo prescription rather than N-body simulations.

CLASSIC-trees is written to be fast and user-friendly, with an easy-to-use Python interface, and integration with [CLASS](https://github.com/lesgourg/class_public) for computation of the background and linear cosmology.

The code relies on the Monte Carlo method of producing halo merger trees presented by Parkinson, Cole, Helly, and the GALFORM Team, [2008 MNRAS (383,557)](https://academic.oup.com/mnras/article/383/2/557/993299) ([arXiv:0708.1382](https://arxiv.org/abs/0708.1382)), and extends it to capture more halo parameters and provide modern output format.

Features of CLASSIC-trees:
* Easy to use Python interface
* Merger tree output matching the Gadget-4 HDF5 Format
* Computes all halo parameters, including subhalos, for use with modern semi-analytical galaxy model codes, such as [SAGE](https://github.com/sage-home/sage-model)
* Support for custom linear matter power spectrum, as well as automatic computation with CLASS.

## Installation

### Downloading
CLASSIC-trees can be downloaded by cloning the GitHub repository:

```bash
git clone https://github.com/MarkMos/CLASSIC-trees.git
cd CLASSIC-trees
```
and installing from he local source code using `pip`:

```bash
pip install .
```

## Working with CLASSIC-trees
We provide several examples for running the code in the examples folder.
