# Robust-PMC
Code for Paper *"Robust Chance-constrained Policy Synthesis and Verification via Moment-Based Distributional Analysis"*

This repository contains implementations and experimental code for robust chance-constrained policy synthesis using moment-based distributional analysis.

## Running the algorithm

- **`mdp.py`**: the environment implementation adapted from a PRISM case study. This file defines the finite MDP used in our experiments.

- **`policy_synthesis.ipynb`**: the notebook demo comparing (1) **risk-neutral value iteration** (baseline) and (2) **robust chance-constrained policy synthesis** using moment-based distributional analysis.

The notebook provides an end-to-end, runnable example on the betting-game environment adapted from a PRISM case study.