# Anomaly Detector with Hidden Markov Models

## Project Overview
This project implements a machine sensor anomaly detection system using a Gaussian Hidden Markov Model (HMM).

The system learns normal machine operating patterns and identifies unusual sensor readings using a sliding window approach.

## Technologies Used
- Python 3.14
- NumPy
- Pandas
- Matplotlib
- Gaussian Hidden Markov Model
- Seaborn

## Project Structure
- data/: Normal and test sensor datasets
- results/: Detection results and visualization
- anomaly_detector.py: Main implementation
- requirements.txt: Required libraries

## Implementation
1. Generated simulated normal machine sensor data.
2. Trained a two-state Gaussian HMM.
3. Calculated training sequence log-likelihoods.
4. Established an anomaly threshold using the 5th percentile.
5. Applied sliding window anomaly detection.
6. Visualized detected anomalies.
7. Exported detection results to CSV.

## Model Selection
A two-state HMM was selected to represent two normal operating conditions of the simulated machine. These states capture different sensor reading patterns.

## Results
The system identifies unusual sensor patterns and highlights detected anomaly regions in a visualization.

Generated outputs:
- anomaly_detection_plot.png
- anomaly_detection_results.csv

## Execution
Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python anomaly_detector.py
```

## Note
The current implementation uses a custom Gaussian HMM built with NumPy. It does not use the hmmlearn library specified in the original assignment requirements.
