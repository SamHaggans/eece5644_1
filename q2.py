import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
from numpy.linalg import eigvals

# Set random seed for reproducibility
np.random.seed(42)

# Consider that a value is "positive" if 1 and "negative" if 0
POSITIVE = 1
NEGATIVE = 0

P_POSITIVE = 0.65
P_NEGATIVE = 0.35
SAMPLE_COUNT = 10000

def expected_error_integration(rv0, rv1):
    def min_point(w, x, y, z):
        val = min(
            rv1.pdf([w, x, y, z]) * P_POSITIVE, rv0.pdf([w, x, y, z]) * P_NEGATIVE
        )
        return val

    # Suggested from Piazza
    opts = {"epsabs": 1.0e-2}
    expected_error, calculation_error = nquad(
        min_point, ranges=[[-5, 5], [-5, 5], [-5, 5], [-5, 5]], opts=opts
    )
    print("Expected Minimum Error:", expected_error)
    return expected_error


def run_q2(
    mean0,
    cov0,
    classification_cov0,
    mean1,
    cov1,
    classification_cov1,
    estimate_covariance=False,
):
    # Generate two multivariate Gaussian random variable objects

    e = eigvals(cov0)
    print("Eigenvalues of cov0 are:", e)
    rv0 = multivariate_normal(mean0, cov0)

    e = eigvals(cov1)
    print("Eigenvalues of cov1 are:", e)
    rv1 = multivariate_normal(mean1, cov1)

    def pick_distribution(choice):
        return rv1 if choice == POSITIVE else rv0

    choices = np.random.choice(
        [NEGATIVE, POSITIVE], size=SAMPLE_COUNT, p=[P_NEGATIVE, P_POSITIVE]
    )
    distribution_choices = np.array([pick_distribution(choice) for choice in choices])
    samples = np.array([choice.rvs() for choice in distribution_choices])
    print(samples)

    if estimate_covariance:

        def pick_mean(i):
            return mean0 if choices[i] == NEGATIVE else mean1

        covariance_entries = []
        for i, sample in enumerate(samples):
            mean = pick_mean(i)
            covariance_entries.append(
                (sample - mean).reshape(4, 1) @ (sample - mean).reshape(1, 4)
            )
        covariance_entries = np.array(covariance_entries)
        classification_cov0 = (1 / SAMPLE_COUNT) * np.sum(covariance_entries, axis=0)
        classification_cov1 = classification_cov0
        print("Generated covariance: \n", classification_cov0)
        e = eigvals(classification_cov0)
        print("Eigenvalues of generated covariance are: ", e)

    classification_rv0 = multivariate_normal(mean0, classification_cov0)
    classification_rv1 = multivariate_normal(mean1, classification_cov1)
    positive_count = len(np.where(choices == POSITIVE)[0])
    negative_count = len(np.where(choices == NEGATIVE)[0])

    print(
        f"Number of positives: {positive_count}, {positive_count / SAMPLE_COUNT * 100}%"
    )
    print(
        f"Number of negatives: {negative_count}, {negative_count / SAMPLE_COUNT * 100}%"
    )

    expected_error_integration(rv0, rv1)

    # Classification with decision rule
    def classify_point(point, gamma):
        if (
            classification_rv1.pdf(point) / classification_rv0.pdf(point)
        ) > gamma:
            return POSITIVE
        else:
            return NEGATIVE

    gammas = []
    error_rates = []
    tprs = []
    fprs = []

    def run_classification(gamma):
        i = 0
        classified_array = []
        for point in samples:
            classification = classify_point(point, gamma)
            new_point = [
                point[0],  # 4 values from the distribution
                point[1],
                point[2],
                point[3],
                classification,
                (
                    1 if classification == choices[i] else 0
                ),  # Correctness of classification
                (
                    1 if classification == POSITIVE and choices[i] == POSITIVE else 0
                ),  # True positive
                (
                    1 if classification == POSITIVE and choices[i] == NEGATIVE else 0
                ),  # False positive
            ]
            classified_array.append(np.array(new_point))
            i += 1

        classified = np.array(classified_array)

        correct_choice_count = len(np.where(classified[:, 5] == 1)[0])
        errors = SAMPLE_COUNT - correct_choice_count
        error_fraction = errors / SAMPLE_COUNT
        true_positives = len(np.where(classified[:, 6] == 1)[0])
        false_positives = len(np.where(classified[:, 7] == 1)[0])

        true_positive_rate = true_positives / positive_count
        false_positive_rate = false_positives / negative_count

        gammas.append(gamma)
        error_rates.append(error_fraction)
        tprs.append(true_positive_rate)
        fprs.append(false_positive_rate)

    GAMMA_COUNT = 10

    for gamma in np.concatenate(
        [
            np.linspace(0, 1, GAMMA_COUNT),
            np.linspace(1, 7, GAMMA_COUNT),
            np.linspace(7, 100, GAMMA_COUNT),
            np.linspace(100, 1000, GAMMA_COUNT),
            np.linspace(1000, 10000, GAMMA_COUNT),
            np.linspace(10000, 10000000, GAMMA_COUNT),
        ]
    ):
        run_classification(gamma)

    min_error_index = np.argmin(np.array(error_rates))

    min_error_fpr = fprs[min_error_index]
    min_error_tpr = tprs[min_error_index]
    min_error_fnr = 1 - min_error_tpr
    print(
        "Min error is ",
        error_rates[min_error_index],
        " at value gamma = ",
        gammas[min_error_index],
        " true positive rate ",
        min_error_tpr,
        " false positive rate ",
        min_error_fpr,
    )

    plt.figure(figsize=(8, 8))
    plt.plot(fprs, tprs, color="darkorange", lw=2, label="ROC Curve")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.scatter(min_error_fpr, min_error_tpr, marker="o", color="blue")
    plt.xlim([-0.0001, 1.0])
    plt.ylim([0.0, 1.0001])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    plt.figure(figsize=(8, 8))
    plt.plot(gammas, error_rates, color="darkorange", lw=2, label="Gamma")
    plt.ylabel("Empirical Error")
    plt.xlabel("Gamma")
    plt.xscale("log")
    plt.ylim([0.0, 1.0])
    plt.title("Empirical Error vs Gamma")
    plt.legend(loc="lower right")

    def k(beta):
        mean_0 = np.array(mean0)
        mean_1 = np.array(mean1)
        cov_0 = np.array(cov0)
        cov_1 = np.array(cov1)

        return (0.5 * beta * (1 - beta)) * np.dot(
            np.transpose(mean_1 - mean_0),
            np.dot(np.linalg.inv(beta * cov_0 + (1 - beta) * cov_1), (mean_1 - mean_0)),
        ) + (0.5) * np.log(
            np.linalg.det(beta * cov_0 + (1 - beta) * cov_1)
            / ((np.linalg.det(cov_0) ** beta) * (np.linalg.det(cov_1) ** (1 - beta)))
        )

    def chernoff_calc(beta):
        return P_POSITIVE**beta * P_NEGATIVE ** (1 - beta) * np.exp(-1 * k(beta))

    beta_tests = np.arange(0, 1.0, 0.001)
    chernoff = [chernoff_calc(beta) for beta in beta_tests]
    bhattacharyya = chernoff_calc(0.5)

    min_chernoff_index = np.argmin(np.array(chernoff))
    print(
        "Min chernoff error is ",
        chernoff[min_chernoff_index],
        " at value beta = ",
        beta_tests[min_chernoff_index],
    )

    print("Bhattacharyya bound is ", bhattacharyya)

    plt.figure(figsize=(8, 8))
    plt.plot(beta_tests, chernoff, color="darkorange", lw=2, label="Chernoff Bound")
    plt.ylabel("Chernoff Error Bound")
    plt.xlabel("Beta")
    plt.title("Chernoff Error Bound vs Beta")
    plt.legend(loc="lower right")
    plt.show()

    return (gammas[min_error_index], min_error_fnr, min_error_fpr)


# For problem 2A

mean0, cov0 = [-1, -1, -1, -1], [
    [5, 3, 1, -1],
    [3, 5, -2, -2],
    [1, -2, 6, 3],
    [-1, -2, 3, 4],
]

wrong_cov_0 = [
    [5, 0, 0, 0],
    [0, 5, 0, 0],
    [0, 0, 6, 0],
    [0, 0, 0, 4],
]


mean1, cov1 = [1, 1, 1, 1], [
    [1.6, -0.5, -1.5, -1.2],
    [-0.5, 8, 6, -1.7],
    [-1.5, 6, 6, 0],
    [-1.2, -1.7, 0, 1.8],
]

wrong_cov1 = [
    [1.6, 0, 0, 0],
    [0, 8, 0, 0],
    [0, 0, 6, 0],
    [0, 0, 0, 1.8],
]

# 2A
run_q2(mean0, cov0, cov0, mean1, cov1, cov1)


# 2C
# run_q2(mean0, cov0, wrong_cov_0, mean1, cov1, wrong_cov1)


# 2D
# run_q2(mean0, cov0, cov0, mean1, cov1, cov1, True)
