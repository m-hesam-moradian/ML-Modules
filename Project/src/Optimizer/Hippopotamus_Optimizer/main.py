import matplotlib.pyplot as plt
import numpy as np
from HO import HO

def F2(x):
    return np.sum(x**2)

def fun_info(fun_name):
    if fun_name == "F2":
        return -100, 100, 30, F2
    raise ValueError("Function not supported.")

if __name__ == "__main__":
    Fun_name = "F2"
    SearchAgents = 16
    Max_iterations = 500
    lowerbound, upperbound, dimension, fitness = fun_info(Fun_name)

    Best_score, Best_pos, HO_curve = HO(
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness
    )

    print(f"Best solution for {Fun_name}: {Best_pos}")
    print(f"Best value for {Fun_name}: {Best_score}")

    plt.figure()
    plt.semilogy(HO_curve, color="#b28d90", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best score obtained so far")
    plt.title("Hippopotamus Optimization Convergence")
    plt.grid(True)
    plt.legend(["HO"])
    plt.show()
