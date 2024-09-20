---
title: AI in the Industry (Anomaly Detection in Higher Dimensional Spaces)
author: michele.lombardi2@unibo.it
---

# Anomaly Detection in Higher Dimensional Spaces #

This a lecture block of the 2023/2023 edition of the "AI in the Industry" course, from [University of Bologna](https://www.unibo.it). Each lecture consists of a tutorial that tackles a simplified industrial problem and tackles it using AI techniques, from Machine Learning to Combinatorial Optimization (and later on their combination).

While the whole course looks like cookbook, the real goal is using examples to teach how industrial problem can be methodically approaches, analyzed, and tackles using a combination of techniques.

This tutorial in particular tackles anomaly detection in task in (relatively) high-dimensional spaces, via density estimation techniques and autoencoders. The focus is on showing how density estimation can be tackled with a broader range of techniques, and also to show how some methods that are not presented from a probabilistic perspective have indeed a probabilistic interpretation.

# Accessing the Lecture #

## Local Execution (Preferred) ##

Students are strongly encouraged to _run all lectures locally_. Doing this will require to:

* Download or clone this repository, e.g. via the command:
```sh
git clone https://github.com/a3i-2024-2025/02-ad-hd.git
```
* Open a terminal in the main directory of the downloaded/cloned repository
* Install the required dependencies, _by using a virtual environment_. Each lectures already contains configuration files to manage dependencies via [poetry](https://python-poetry.org). If you are using poetry, you can install dependencies via:
```sh
poetry install
```
* Start the notebook server in a virtual environment
```sh
poetry run jupyter notebook
```

If you do not wish to use poetry, a `requirements.txt` file with frozen dependency versions is also provided. However, **you are strongly advised to still use virtual environments**. The course will eventually make use of multiple libraries and package versions between lectures will sometimes be inconsistent.

## Read-only Access and PDF Notes ##

You can inspect the individual notebooks in by just clicking on any `*.ipynb` file in the `notebooks` directory: github provides a notebook viewer that mostly works, though this access method may occasionally have issue when displaying plots.

The repository contains PDF notes for all the notebooks. They can be used for read-only access (with more consistent results compared to the github notebook viewer), but more importantly they can be useful to add annotations. Just keep in mind that in case of updates, cloning the repository will replace the PDF files.
