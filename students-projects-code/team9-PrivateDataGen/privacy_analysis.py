from prv_accountant import PRVAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism, GaussianMechanism
import numpy as np

dataset = "OA"
assert dataset in ["summarize", "HH", "SHP", "OA"]
eps_pca_list = []
sigma_sgd_list = []
sigma_histogram_list = []

if dataset == "OA":
    full_train_num = 14167
    n_cluster = 10
    cluster_size = int(full_train_num/(n_cluster+3))
    n = 4
    running_steps_dpsgd = int(4/(n/cluster_size))

    eps_pca_list = [0.5, 1]
    sigma_sgd_list = [0.57, 0.47]
    sigma_histogram_list = [20, 20]

elif dataset == "HH":
    full_train_num = 160800
    n_cluster = 10
    cluster_size = int(full_train_num/(n_cluster+3))
    n = 4
    running_steps_dpsgd = int(4/(n/cluster_size))

    eps_pca_list = [0.5, 1]
    sigma_sgd_list = [0.49, 0.41]
    sigma_histogram_list = [20, 20]

elif dataset == "summarize":
    full_train_num = 92858
    n_cluster = 10
    cluster_size = int(full_train_num/(n_cluster+3))
    n = 4
    running_steps_dpsgd = int(4/(n/cluster_size))

    eps_pca_list = [0.5, 1]
    sigma_sgd_list = [0.50, 0.42]
    sigma_histogram_list = [20, 20]

sample_rate = n / cluster_size
assert len(eps_pca_list) == len(sigma_sgd_list) == len(sigma_histogram_list)
delta = 1 / full_train_num

def get_privacy_spent(sampling_prob_dpsgd, running_steps_dpsgd,
                      noise_multiplier_dpsgd, noise_multiplier_histogram, delta):

    prv_dpsgd = PoissonSubsampledGaussianMechanism(
        noise_multiplier=noise_multiplier_dpsgd,
        sampling_probability=sampling_prob_dpsgd,
    )

    prv_histogram = GaussianMechanism(
        noise_multiplier=noise_multiplier_histogram,
        l2_sensitivity = np.sqrt(2),
    )

    accountant = PRVAccountant(
        prvs=[prv_dpsgd, prv_histogram],
        max_self_compositions=[running_steps_dpsgd, 1],
        eps_error=0.01,
        delta_error=delta/10,
    )

    # Compute ε for a given δ
    eps_lower, eps_estimate, eps_upper = accountant.compute_epsilon(
        delta=delta,
        num_self_compositions=[running_steps_dpsgd, 1],
    )

    return eps_upper

for i in range(len(eps_pca_list)):
    eps_pca = eps_pca_list[i]
    sigma_sgd = sigma_sgd_list[i]
    sigma_hist = sigma_histogram_list[i]

    eps_dpsgd_hist = get_privacy_spent(
        sampling_prob_dpsgd=sample_rate,
        running_steps_dpsgd=running_steps_dpsgd,
        noise_multiplier_dpsgd=sigma_sgd,
        noise_multiplier_histogram=sigma_hist,
        delta=delta
    )

    total_eps = eps_pca + eps_dpsgd_hist

    print(f"total ε = {total_eps:.3f}")



