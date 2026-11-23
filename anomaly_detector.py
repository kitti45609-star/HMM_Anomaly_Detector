
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# 1. Generate simulated normal sensor data
def generate_normal_data(n=2000):
    states = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    readings = []

    for state in states:
        if state == 0:
            readings.append(np.random.normal(50, 2))
        else:
            readings.append(np.random.normal(60, 2))

    return np.array(readings).reshape(-1, 1)


# 2. Gaussian HMM implementation using NumPy
class GaussianHMM:
    def __init__(self, n_states=2, iterations=30):
        self.n_states = n_states
        self.iterations = iterations

    def logsumexp(self, values, axis=None):
        maximum = np.max(values, axis=axis, keepdims=True)
        result = maximum + np.log(
            np.sum(np.exp(values - maximum), axis=axis, keepdims=True) + 1e-300
        )
        return np.squeeze(result, axis=axis)

    def emission_probabilities(self, X):
        n = len(X)
        log_prob = np.zeros((n, self.n_states))

        for state in range(self.n_states):
            variance = max(self.variances[state], 1e-6)
            log_prob[:, state] = (
                -0.5 * np.log(2 * np.pi * variance)
                - 0.5 * (X[:, 0] - self.means[state]) ** 2 / variance
            )

        return log_prob

    def fit(self, X):
        X = np.asarray(X)
        n = len(X)

        self.means = np.percentile(
            X[:, 0],
            np.linspace(20, 80, self.n_states)
        )
        self.variances = np.full(
            self.n_states, np.var(X[:, 0]) / self.n_states
        )
        self.start_prob = np.full(self.n_states, 1 / self.n_states)
        self.trans_prob = np.full(
            (self.n_states, self.n_states), 1 / self.n_states
        )

        for _ in range(self.iterations):
            emission = self.emission_probabilities(X)

            log_start = np.log(self.start_prob + 1e-12)
            log_trans = np.log(self.trans_prob + 1e-12)

            forward = np.zeros((n, self.n_states))
            forward[0] = log_start + emission[0]

            for t in range(1, n):
                forward[t] = emission[t] + self.logsumexp(
                    forward[t - 1][:, None] + log_trans, axis=0
                )

            backward = np.zeros((n, self.n_states))

            for t in range(n - 2, -1, -1):
                backward[t] = self.logsumexp(
                    log_trans + emission[t + 1][None, :]
                    + backward[t + 1][None, :],
                    axis=1
                )

            gamma_log = forward + backward
            gamma_log -= self.logsumexp(gamma_log, axis=1)[:, None]
            gamma = np.exp(gamma_log)

            xi_sum = np.zeros((self.n_states, self.n_states))

            for t in range(n - 1):
                log_xi = (
                    forward[t][:, None]
                    + log_trans
                    + emission[t + 1][None, :]
                    + backward[t + 1][None, :]
                )
                log_xi -= self.logsumexp(log_xi)
                xi_sum += np.exp(log_xi)

            self.start_prob = gamma[0] + 1e-8
            self.start_prob /= self.start_prob.sum()

            self.trans_prob = xi_sum + 1e-8
            self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)

            weights = gamma.sum(axis=0) + 1e-8

            self.means = (
                gamma.T @ X[:, 0]
            ) / weights

            for state in range(self.n_states):
                self.variances[state] = max(
                    np.sum(
                        gamma[:, state]
                        * (X[:, 0] - self.means[state]) ** 2
                    ) / weights[state],
                    1e-6
                )

        return self

    def score(self, X):
        X = np.asarray(X)
        emission = self.emission_probabilities(X)

        forward = np.log(self.start_prob + 1e-12) + emission[0]
        log_trans = np.log(self.trans_prob + 1e-12)

        for t in range(1, len(X)):
            forward = emission[t] + self.logsumexp(
                forward[:, None] + log_trans, axis=0
            )

        return float(self.logsumexp(forward))


# 3. Generate training data
print("Generating normal sensor data...")
train_data = generate_normal_data(2000)

pd.DataFrame(
    train_data, columns=["Sensor_Reading"]
).to_csv(DATA_DIR / "normal_sensor_data.csv", index=False)


# 4. Train the HMM
print("Training Gaussian HMM...")

model = GaussianHMM(n_states=2, iterations=30)
model.fit(train_data)

print("HMM training completed!")


# 5. Calculate training likelihoods and threshold
window_size = 50
train_scores = []

for i in range(0, len(train_data) - window_size + 1, window_size):
    window = train_data[i:i + window_size]
    train_scores.append(model.score(window) / window_size)

threshold = np.percentile(train_scores, 5)

print("Anomaly threshold:", threshold)


# 6. Generate test stream with anomalies
print("Generating test sensor stream...")

test_data = generate_normal_data(1000)

# Insert abnormal machine readings
test_data[300:350] = np.random.normal(85, 3, (50, 1))
test_data[700:750] = np.random.normal(20, 3, (50, 1))

pd.DataFrame(
    test_data, columns=["Sensor_Reading"]
).to_csv(DATA_DIR / "test_sensor_data.csv", index=False)


# 7. Sliding window anomaly detection
anomaly_regions = []
window_scores = []

for i in range(0, len(test_data) - window_size + 1, 1):
    window = test_data[i:i + window_size]
    score = model.score(window) / window_size

    window_scores.append(score)

    if score < threshold:
        anomaly_regions.append(i)

anomaly_mask = np.zeros(len(test_data), dtype=bool)

for start in anomaly_regions:
    anomaly_mask[start:start + window_size] = True

print("Total anomaly windows:", len(anomaly_regions))
print("Anomalous sensor readings:", anomaly_mask.sum())


# 8. Save detection results
results = pd.DataFrame({
    "Time": np.arange(len(test_data)),
    "Sensor_Reading": test_data[:, 0],
    "Anomaly": anomaly_mask
})

results.to_csv(
    RESULTS_DIR / "anomaly_detection_results.csv",
    index=False
)


# 9. Visualize sensor readings and anomalies
plt.figure(figsize=(15, 6))

plt.plot(
    test_data[:, 0],
    label="Sensor Readings",
    color="blue",
    linewidth=1
)

plt.fill_between(
    np.arange(len(test_data)),
    test_data[:, 0].min() - 5,
    test_data[:, 0].max() + 5,
    where=anomaly_mask,
    color="red",
    alpha=0.25,
    label="Detected Anomalies"
)

plt.title("Gaussian HMM - Machine Sensor Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Sensor Reading")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "anomaly_detection_plot.png",
    dpi=300
)

plt.show()

print("\nProject completed successfully!")
print("Check the results folder for the graph and CSV file.")
