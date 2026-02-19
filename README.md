# Truck Braking System Optimization Using FFNN & GA

## Overview
This project implements a **genetic algorithm (GA)** to optimize a **feedforward neural network (FFNN)** controlling the braking of heavy-duty trucks on variable downhill slopes. The network decides **brake pedal pressure** and **gear changes** while ensuring brake temperatures stay safe.

The system models:
- Foundation brakes and engine brakes  
- Brake temperature dynamics  
- Truck acceleration and downhill slope

## Features
- FFNN with a single hidden layer, optimized via GA  
- Simulation of truck dynamics with constraints on **speed** and **brake temperature**  
- Evaluation on training, validation, and test slope sets  
- Automatic test program with plots of slope angle, speed, brake pressure, gear, and temperature  

## Project Structure
- `run_encoding_decoding_test.py` – Encode/decode FFNN to/from chromosome  
- `slopes.py` – Generates training, validation, and test slopes  
- `run_ffnn_optimization.py` – GA for FFNN optimization  
- `run_test.py` – Test program to evaluate the best FFNN on any slope  
- `best_chromosome.py` – Stores the optimized FFNN chromosome  

## Usage
1. Run GA optimization:  
```bash
python run_ffnn_optimization.py
