import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import eigvals


train_features = "X_train.txt"
train_labels = "y_train.txt"

samples = []

with open(train_features, "r") as features_file:
    with open(train_labels, "r") as labels_file:
        for features_line, labels_line in zip(features_file, labels_file):
            try:
                sample = []
                for element in features_line.strip().split(" "):
                    try:
                        sample.append(float(element))
                    except:
                        # Skip any unparasable characters for whatever reason
                        continue
                # Add a label to each element so they can be tracked
                sample = sample + [int(labels_line.strip())]
                samples.append(sample)
            except:
                # Skip unparsable lines
                continue
samples = np.array(samples)
means_dict = dict()
covariances_dict = dict()
distributions_dict = dict()
priors_dict = dict()

REGULARIZATION_ALPHA = 0.5


def remove_label(sample):
    return sample[:, :-1]

NUM_SAMPLES = 7352
labeled_samples = dict()
for i in range(1, 7):
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
        covariances_dict[i] = initial_covariance + reg_param * np.identity(561)
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

print("Printing confusion matrix: ")
for key in confusion_matrix:
    for value in confusion_matrix[key].values():
        print(value, end=",")
    print()

print("Error rate on training data: ", error_count / NUM_SAMPLES)

colors = ["red", "orange", "yellow", "green", "blue", "purple", "forestgreen"]
labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

plt.figure()
for i, key in enumerate(ordered_keys):
    plt.scatter(labeled_samples[key][:, 0], labeled_samples[key][:, 1], label=labels[i], alpha=0.7,c=colors[i])
plt.xlabel("tBodyAcc-mean()-X")
plt.ylabel("tBodyAcc-mean()-Y")
plt.title("Activity Distribution")
plt.show()


###
### Testing using test data below this section
###

test_features = "X_test.txt"
test_labels = "y_test.txt"

test_samples = []

NUM_TEST_SAMPLES = 2947
with open(test_features, "r") as features_file:
    with open(test_labels, "r") as labels_file:
        for features_line, labels_line in zip(features_file, labels_file):
            try:
                sample = []
                for element in features_line.strip().split(" "):
                    try:
                        sample.append(float(element))
                    except:
                        # Skip any unparasable characters for whatever reason
                        continue
                sample = sample + [int(labels_line.strip())]
                test_samples.append(sample)
            except:
                # Skip unparsable lines
                continue
test_samples = np.array(test_samples)


labeled_test_samples = dict()
for i in range(1, 7):
    labeled_test_samples[i] = test_samples[np.where(test_samples[:, -1] == i)]

test_error_count = 0

for i in ordered_keys:
    for point in labeled_test_samples[i]:
        classification = classify_point(point)
        if int(point[-1]) != classification:
            test_error_count += 1

error_rate_test = test_error_count / NUM_TEST_SAMPLES

print("Error rate on test data: ", error_rate_test)



