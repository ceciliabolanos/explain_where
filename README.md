# Time-Localized Explanations for Audio Models

This repository contains the complete codebase for generating **time-localized explanations for audio models**.

## 🧠 General Pipeline

The general process to generate an explanation for a given audio signal is:

1. **Generate perturbed instances** near the original signal and process them through the **black-box model** we want to explain.
2. Use the resulting data (i.e., what was perturbed and the corresponding model outputs) to train a **surrogate model**.
3. The surrogate model is designed to be **interpretable**, so we use it to derive explanations for the black-box model.

---

## 📂 Repository Structure

### `data_generator/`

Contains code to define and apply different perturbation strategies to input signals.  
You can modify existing perturbation types or add your own.

---

### `explainers/`

Includes implementations of surrogate models used to approximate and explain the black-box model.  
You are welcome to implement and add your own surrogate models here.

---

### `datasets/` & `models/`

These folders contain the datasets and corresponding black-box models used for each dataset.  
Each model defines the required **forward pass** used during the explanation process.

---

### `evaluation/`, `experiments/`, `plots/`

These folders organize the evaluations and experiments conducted in the paper, as well as visualization of results.

---

### `run_explanation.py`

This script **runs the full pipeline**:  
Given an input instance, it generates perturbed samples, queries the black-box model, trains the surrogate model, and produces an explanation.

---

## 📌 Notes

- This repository is part of an ongoing effort to provide **benchmark tools** for explaining audio models.
- Feel free to open an issue or pull request if you want to contribute new perturbation methods, surrogate models, or experiments.

