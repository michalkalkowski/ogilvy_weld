MINA
==============================

An implementation of the geometrical Ogilvy's model for predicting grain orientation in austenitic stainless steel multipass welds

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Exploratory Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── output             <- Reports, results, outputs
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── mina               <- Source code for use in this project.
    │   ├── __init__.py    <- __init__.py
    │   │
    │   └── mina_model.py  <- Main file containing the MINA_weld class and related functions.

--------
Setup
--------
`ogilvy_weld` is used in the same environment as `mina_weld`. To keep things
simple, `mina_weld` environment is used for both. 

1. Install the virtual environment:
```bash
$ conda env create -f environment.yml
```
2. Activate the environment:
```bash
$ conda activate mina_weld
```
or
```bash
$ source activate mina_weld
```
3. Install `mina` package inside the virtual environment
```bash
$ pip install --editable .
```
(no reinstallation is requres after modification thanks to the `--editable` flag)

4. Enjoy!

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/hgrif/example-project/">GoDataDriven project template</a>.</small></p>
