import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from scipy.special import gammaln, erfc

##################### DATA SIMULATION #####################
def simulate_data(n, p=[0.38, 0.35, 0.27]):
    p = np.array(p) / np.sum(p)
    return np.random.choice(len(p), size=n, p=p)


##################### CLT WITH CONTINUITY CORRECTION AND BERRY-ESSEN #####################
def clt_correction_individual(n, p_c = 0.38, p_jstar = 0.35):
    sigma2 = p_c * (1-p_c) + p_jstar * (1-p_jstar) + 2 * p_c * p_jstar # second order moment
    z_cc = np.sqrt(n) * (p_c - p_jstar) / np.sqrt(sigma2) - 0.5 / np.sqrt(n * sigma2)
    correction_term = 0.5 * erfc(z_cc/np.sqrt(2))

    rho = p_c - p_jstar + 2 * (p_c - p_jstar)**3 - 3 * (p_c - p_jstar) * (p_c + p_jstar) # cubic moment
    berry_essen_term = 0.56 * rho / (sigma2**(3/2) * np.sqrt(n))
    return correction_term + berry_essen_term

def clt_correction(n, p_0 = 0.38, p_1 = 0.35, p_2 = 0.27):
   return clt_correction_individual(n, p_c = p_0, p_jstar = p_1) + clt_correction_individual(n, p_c = p_0, p_jstar = p_2)

##################### HOEFFDING BOUND ####################
def hoeffding(n, p_0 = 0.38, p_1 = 0.35, p_2= 0.27):
    term_1 = np.exp(-0.5 * n * (p_0 - p_1)**2)
    term_2 = np.exp(-0.5 * n * (p_0 - p_2)**2)
    return term_1 + term_2


##################### BERNSTEIN BOUND #####################
def bernstein_individual(n, p_0 = 0.38, p_1 = 0.35):
    sigma2 = p_0 * (1-p_0) + p_1 * (1-p_1) + 2 * p_0 * p_1
    denominator = 2 * sigma2 + 2/3 * (p_0 - p_1) + 2/3 * (p_0 - p_1)**2
    return np.exp(- n * (p_0 - p_1)**2/denominator)

def bernstein(n, p_0 = 0.38, p_1 = 0.35, p_2 = 0.27):
    return bernstein_individual(n, p_0 = p_0, p_1 = p_1) + bernstein_individual(n, p_0 = p_0, p_1 = p_2)


##################### CHERNOFF-MARKOV BOUND #####################
def chernoff_markov_individual(n, p_0 = 0.38, p_1 = 0.35):
    return np.exp(n * np.log(2 * np.sqrt(p_0 * p_1) + 1 - p_0 - p_1))

def chernoff_markov(n, p_0 = 0.38, p_1 = 0.35, p_2 = 0.27):
    return chernoff_markov_individual(n, p_0 = p_0, p_1 = p_1) + chernoff_markov_individual(n, p_0 = p_0, p_1 = p_2)


##################### CHERNOFF-MARKOV W/ BAHADUR-RAO LATTICE PREFACTOR #####################
def chernoff_markov_individual_correction(n, p_0 = 0.38, p_1 = 0.35):
    sigma = 2 * np.sqrt(p_0 * p_1) / (1-(np.sqrt(p_0)-np.sqrt(p_1))**2)
    prefactor = np.sqrt(2 * np.pi * n * sigma) * (1-np.sqrt(p_1/p_0)) 
    return np.exp(n * np.log(2 * np.sqrt(p_0 * p_1) + 1 - p_0 - p_1)) / prefactor

def chernoff_markov_correction(n, p_0 = 0.38, p_1 = 0.35, p_2 = 0.27):
    return chernoff_markov_individual_correction(n, p_0 = p_0, p_1 = p_1)


##################### EXACT MULTINOMIAL ERROR #####################
def multinomial_error(n, p0 = 0.38, p1 = 0.35, p2 = 0.27, c_star=0):
    p = [p0, p1, p2]
    k = 3

    # DP state: (t, m, s) -> weight, where
    #   t = votes for true class (c_star)
    #   m = max rival votes so far
    #   s = total votes allocated so far
    #   weight = product_j p_j^{x_j} / x_j!

    # Precompute logs
    log_p = [math.log(pi) if pi > 0 else float('-inf') for pi in p]
    log_fact = [math.lgamma(x+1) for x in range(n+1)]  # log(x!)

    # DP state: (t, m, s)
    DP = defaultdict(list)
    DP[(0, 0, 0)] = [0.0]  # log(1) = 0

    for j in range(k):
        newDP = defaultdict(list)
        for (t, m, s), logs in DP.items():
            max_x = n - s
            for x in range(max_x + 1):
                new_t = t + x if j == c_star else t
                new_m = m if j == c_star else max(m, x)
                new_s = s + x

                # contribution in log form
                if log_p[j] == float('-inf') and x > 0:
                    continue  # pass not possible events
                contrib = x * log_p[j] - log_fact[x]

                for lw in logs:
                    newDP[(new_t, new_m, new_s)].append(lw + contrib)
        DP = newDP

    # add log n!
    log_factn = log_fact[n]

    # Collect all error states
    logs_to_sum = []
    for (t, m, s), logs in DP.items():
        if s == n and m >= t:
            for lw in logs:
                logs_to_sum.append(log_factn + lw)

    # Stable log-sum-exp
    if not logs_to_sum:
        return 0.0
    mval = max(logs_to_sum)
    prob = math.exp(mval) * sum(math.exp(lw - mval) for lw in logs_to_sum)
    return prob


##################### EMPIRICAL ERROR #####################

def empirical_error_fun(n, p = [0.38, 0.35, 0.27], montecarlo_n = 10**6):
    data = np.random.choice(len(p), size=(montecarlo_n,n), p=p)
    counts = np.stack([(data==i).sum(axis=1) for i in range(3)], axis=1)
    sorted_indices = np.argsort(-counts, axis=1)
    incorrect = np.sum(sorted_indices[:,0]!=0)
    return incorrect / montecarlo_n

##################### MAIN FUNCTION TO PLOT #####################

def main():
    n_list = np.array([20,30,40,50,60,80,100,500,1000]) 

    empirical_error_list = np.array([empirical_error_fun(int(n)) for n in n_list])
    hoeffding_list = np.array([hoeffding(int(n)) for n in n_list])
    bernstein_list = np.array([bernstein(int(n)) for n in n_list])
    chernoff_markov_list = np.array([chernoff_markov(int(n)) for n in n_list])
    chernoff_markov_correction_list = np.array([chernoff_markov_correction(int(n)) for n in n_list])
    multinomial_error_list = np.array([multinomial_error(int(n)) for n in n_list])
    corrections_clt_list = np.array([clt_correction(int(n)) for n in n_list])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(n_list, empirical_error_list, marker='o', label='Empirical')
    plt.plot(n_list, hoeffding_list, marker='s', label='Hoeffding')
    plt.plot(n_list, bernstein_list, marker='^', label='Bernstein')
    plt.plot(n_list, chernoff_markov_list, marker='x', label='Chernoff-Markov')
    plt.plot(n_list, chernoff_markov_correction_list, marker='v', label='Chernoff-Markov + BR')
    plt.plot(n_list, multinomial_error_list, marker='d', label='Multinomial Exact')
    plt.plot(n_list, corrections_clt_list, marker='+', label='CLT + CC + BE')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Panel size n')
    plt.ylabel('Error probability (log-log scale)')
    # plt.title('Empirical vs. theoretical Condorcet bounds')
    plt.legend()
    plt.grid(True, which="both", ls='--', lw=0.5)
    plt.tight_layout()
    #plt.savefig("empirical_vs_theoreticalbounds.png", dpi=300)  # High resolution
    plt.savefig("empirical_vs_theoreticalbounds.pdf") 

    return 
        

if __name__ == "__main__":
    main()

