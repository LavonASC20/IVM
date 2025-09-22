import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from scipy.special import expit
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel


def kernel_rbf(x, y): # gaussian kernel, can use whichever is desired
    return np.exp((-1/2)*((np.linalg.norm(x-y))**2))

def kernel_linear(x, y): # linear kernel, can use whichever is desired
    return np.array(x).T @ np.array(y)

""""""""""""""""""""""""""""""""""""""""""""""""
def kernel_mat(X, Y, kernel):
    if kernel == "linear":
        return linear_kernel(X, Y)
    if kernel == "rbf":
        return rbf_kernel(X, Y)

""""""""""""""""""""""""""""""""""""""""""""""""

def optimize_a(K_a, y, lambda_, K_q):
    dud = np.zeros(shape = (K_a.shape[1], 1))
    a_0 = 0.5*np.ones(shape = (K_a.shape[1], 1))
    a_res = [dud, a_0] 
    while (np.linalg.norm(a_res[-1]) - np.linalg.norm(a_res[-2]) > 0.01):
        W = np.diag(np.array(expit(K_a @ np.array(a_res[-1]).reshape(-1,1))*(1-expit(K_a @ np.array(a_res[-1]).reshape(-1,1)))).reshape(-1, ))
        z = K_a @ np.array(a_res[-1]).reshape(-1,1) + np.linalg.inv(W) @ (np.array(y) - expit(K_a @ np.array(a_res[-1]).reshape(-1,1)))
        a = np.linalg.inv(K_a.T @ W @ K_a + lambda_ * K_q) @ K_a.T @ W @ z
        a_res.append(a)
    return a_res[-1]

""""""""""""""""""""""""""""""""""""""""""""""""
def compute_H_rbf(S, X, y):
    K_a_l = kernel_mat(X, S, "rbf")
    lambda_ = 0.5
    K_q_l = kernel_mat(S, S, "rbf")
    a_l = optimize_a(K_a_l, y, lambda_, K_q_l)
    H_ = -1*y.T @ (K_a_l @ a_l) + np.sum(np.log(1 + np.exp(K_a_l @ a_l))) + (lambda_/2) * a_l.T @ K_q_l @ a_l
    return np.array(H_), np.array(a_l)

def compute_H_linear(S, X, y):
    K_a_l = kernel_mat(X, S, "linear")
    lambda_ = 0.5
    K_q_l = kernel_mat(S, S, "linear")
    a_l = optimize_a(K_a_l, y, lambda_, K_q_l)
    H_ = -1*y.T @ (K_a_l @ a_l) + np.sum(np.log(1 + np.exp(K_a_l @ a_l))) + (lambda_/2) * a_l.T @ K_q_l @ a_l
    return np.array(H_), np.array(a_l)
""""""""""""""""""""""""""""""""""""""""""""""""

def fit_rbf(X,y):
    S = []
    R = X
    k = 0
    err_H = []

    while True:
        if k > 1 and np.abs(err_H[k-1] - err_H[k-2])/np.abs(err_H[k-1]) < 0.01:
            break
        res_x = []
        res_H = []
        for x_l in R: 
            H, a = compute_H_rbf(S + [x_l],X,y)
            res_x.append(x_l)
            res_H.append(H)
        x_star = res_x[np.argmin(np.array(res_H))]
        err_H.append(min(res_H))
        S.append(x_star)
        index = np.where(R == x_star)[0][0]
        R = np.delete(R, (index), axis = 0)
        print("S: ", len(S), "R: ", len(R))
        k = k + 1
    print("S: ", len(S), "R: ", len(R), "a: ", a)
    return S, err_H, a

def fit_linear(X,y):
    S = []
    R = X
    k = 0
    err_H = []

    while True:
        if k > 1 and np.abs(err_H[k-1] - err_H[k-2])/np.abs(err_H[k-1]) < 0.01:
            break
        res_x = []
        res_H = []
        for x_l in R: 
            H, a = compute_H_linear(S + [x_l],X,y)
            res_x.append(x_l)
            res_H.append(H)
        x_star = res_x[np.argmin(np.array(res_H))]
        err_H.append(min(res_H))
        S.append(x_star)
        index = np.where(R == x_star)[0][0]
        R = np.delete(R, (index), axis = 0)
        print("S: ", len(S), "R: ", len(R))
        k = k + 1
    print("S: ", len(S), "R: ", len(R), "a: ", a)
    return S, err_H, a
""""""""""""""""""""""""""""""""""""""""""""""""
def predict_one_rbf(a, S, x):
    sum = 0
    for ind_S, s in enumerate(S):
        sum += a[ind_S]*np.exp((-1/2)*(np.linalg.norm(x-s)**2))
    pred = expit(sum)
    if pred >= 0.5:
        return 1
    else: 
        return 0
    
def predict_one_rbf_multiclass(a, S, x):
    sum = 0
    for ind_S, s in enumerate(S):
        sum += a[ind_S]*np.exp((-1/2)*(np.linalg.norm(x-s)**2))
    pred = expit(sum)
    return pred

def predict_one_linear(a, S, x):
    sum = 0
    for ind_S, s in enumerate(S):
        sum += a[ind_S]*np.array(x).T@np.array(s)
    pred = expit(sum)
    if pred >= 0.5:
        return 1
    else: 
        return 0
""""""""""""""""""""""""""""""""""""""""""""""""

def plot_training_data_with_decision_boundary(kernel, X, y):
    # Train the SVC
    svm = SVC(kernel=kernel, gamma=2).fit(X, y)

    # Settings for plotting
    _, ax = plt.subplots(figsize=(5, 3))

    # Plot decision boundary and margins
    common_params = {"estimator": svm, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.5,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="auto",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=250,
        facecolors="none",
        edgecolors="k",
    )
    # Plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    ax.set_title(f"Decision boundaries of {kernel} kernel in SVC")

    _ = plt.show()

def plot_training_data_with_decision_boundary_IVM(kernel, X, y, import_vectors):
    # Train the SVC
    svm = SVC(kernel=kernel, gamma=2).fit(X, y)

    # Settings for plotting
    _, ax = plt.subplots(figsize=(5, 3))

    # Plot decision boundary and margins
    common_params = {"estimator": svm, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.5,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="auto",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=250,
        facecolor="none",
        edgecolors="k",
    )
    ax.scatter(
        import_vectors[:, 0],
        import_vectors[:, 1],
        s = 250,
        facecolor="red",
        edgecolors="k"
    )
    # Plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    ax.set_title(f"Decision boundaries of {kernel} kernel in SVC (red points are Import Vectors)")

    _ = plt.show()