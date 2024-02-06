import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import eigvals

np.random.seed(42)

csv_file_path = "winequality-white.csv"

samples = []

with open(csv_file_path, "r") as data_file:
    for line in data_file:
        try:
            samples.append([float(val) for val in line.strip().split(";")])
        except:
            # Skip unparsable lines
            continue

samples = np.array(samples)
# Dictionaries for distribution properties per-class
means_dict = dict()
covariances_dict = dict()
distributions_dict = dict()
priors_dict = dict()

REGULARIZATION_ALPHA = 0.00005

def remove_label(sample):
    return sample[:, :-1]

NUM_SAMPLES = 4898
labeled_samples = dict()
for i in range(0, 11):
    labeled_samples[i] = samples[np.where(samples[:, -1] == i)]
    print(f"Samples {i} have count {len(labeled_samples[i])}")
    if len(labeled_samples[i]) > 0:
        priors_dict[i] = len(labeled_samples[i]) / NUM_SAMPLES
        means_dict[i] = np.mean(remove_label(labeled_samples[i]), axis=0)
        centered = remove_label(labeled_samples[i]) - means_dict[i]

        initial_covariance = np.cov(centered, rowvar=False)
        reg_param = (
            REGULARIZATION_ALPHA
            * np.trace(initial_covariance)
            / np.linalg.matrix_rank(initial_covariance)
        )
        print(reg_param)
        covariances_dict[i] = initial_covariance + reg_param * np.identity(11)
        distributions_dict[i] = multivariate_normal(means_dict[i], covariances_dict[i])

ordered_keys = sorted(distributions_dict.keys())
min_key = ordered_keys[0]


def classify_point(point):
    dist_probs = np.array(
        [distributions_dict[i].pdf(point[:-1]) * priors_dict[i] for i in ordered_keys]
    )
    best_dist = np.argmax(dist_probs) + min_key
    return best_dist


error_count = 0

confusion_matrix = dict()
for key in ordered_keys:
    confusion_matrix[key] = dict()
    for key2 in ordered_keys:
        confusion_matrix[key][key2] = 0


for i in ordered_keys:
    for point in labeled_samples[i]:
        classification = classify_point(point)
        if int(point[-1]) != classification:
            error_count += 1
        confusion_matrix[int(point[-1])][classification] += 1

error_rate = error_count / NUM_SAMPLES

print("Error rate", error_rate)

print("Printing confusion matrix: ")
for key in confusion_matrix:
    for value in confusion_matrix[key].values():
        print(value, end=",")
    print()

colors = ["red", "orange", "yellow", "green", "blue", "purple", "forestgreen"]
    
plt.figure()
for i, key in enumerate(ordered_keys):
    plt.scatter(labeled_samples[key][:, 3], labeled_samples[key][:, 10], label=f'Quality {key}', alpha=0.7,c=colors[i])

plt.xlabel("Residual Sugar")
plt.ylabel("Alcohol")
plt.title("Wine Quality Distribution")
plt.show()
