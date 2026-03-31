# Assignment 1 - Adversarial Attacks

This repository contains my implementation for Assignment #1 of **Reliable and Trustworthy Artificial Intelligence**.

The goal of this assignment is to implement adversarial attacks on image classifiers for:

- MNIST
- CIFAR-10

Implemented attacks:

- Targeted FGSM
- Untargeted FGSM
- Targeted PGD
- Untargeted PGD

The implementation follows the assignment requirements, including:
- training/evaluating models on clean data
- measuring attack success rates
- saving adversarial visualization results

---

## Project Structure

- `models.py`  
  Defines classification models for MNIST and CIFAR-10.

- `train.py`  
  Contains training and evaluation functions.

- `attacks.py`  
  Implements FGSM and PGD attacks (targeted / untargeted).

- `utils.py`  
  Utility functions such as seed fixing, prediction, attack evaluation, and visualization saving.

- `test.py`  
  Main script that loads or trains models, runs attacks, prints attack success rates, saves results images and exports CSV results.

- `results/`  
  Stores saved adversarial example images, attack success rate CSV file, graphs generated from the CSV results

- `report.pdf`  
  Final short report for the assignment.


---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:


```bash
python test.py'
```

This will:
1. load or train the MNIST model
2. load or train the CIFAR-10 model
3. evaluate clean accuracy
4. run FGSM and PGD attacks
5. measure attack success rates
6. save adversarial example visualizations
7. save attack success rates as a CSV file

---

## Requirements

The main dependencies are:
- torch
- torchvision
- numpy
- matplotlib
- pandas

See requirements.txt for details.

---

## Notes
- For targeted attacks, the target label is chosen as (y + 1) % 10.
- Attack success rate is measured on 256 test samples.
- Visualization images show:
	- original image
	- adversarial image
	- magnified perturbation

---

## Reproducibility

To improve reproducibility:
- random seeds are fixed
- dependencies are listed in requirements.txt
- the code is separated into modular files