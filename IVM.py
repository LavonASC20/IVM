import numpy as np
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.metrics import accuracy_score, log_loss

class ImportVectorMachine:
    def __init__(self, kernel="rbf", kernel_params=None, lambda_=0.5,
                 tol=1e-2, max_sel=200, top_k=10, proxy_type="gradient",
                 verbose=True):
        self.kernel = kernel
        self.kernel_params = kernel_params or {"gamma": 1.0}
        self.lambda_ = lambda_
        self.tol = tol
        self.max_sel = max_sel
        self.top_k = top_k
        self.proxy_type = proxy_type
        self.verbose = verbose
        self.S_idx = []
        self.err_H = []
        self.a = None
        self.K_train = None
        self.X_train = None

    def _precompute_kernel(self, X, Y=None):
        if self.kernel == "rbf":
            return rbf_kernel(X, X if Y is None else Y, **self.kernel_params)
        elif self.kernel == "linear":
            return linear_kernel(X, X if Y is None else Y)
        else:
            raise ValueError("Unsupported kernel")

    def _optimize_a(self, K_a, y, K_q, a_init=None):
        n, m = K_a.shape
        y = y.reshape(-1, 1)
        a = np.zeros((m, 1)) if a_init is None else a_init.reshape(m, 1)
        eps = 1e-12
        for _ in range(100):
            Ka_a = K_a @ a
            p = expit(Ka_a)
            W = (p * (1 - p)).reshape(-1)
            KW = (K_a.T * W) @ K_a
            W_safe = np.maximum(W, 1e-12)
            z = Ka_a + (y - p) / W_safe.reshape(-1, 1)
            A = KW + self.lambda_ * K_q + eps * np.eye(m)
            b = K_a.T @ (W.reshape(-1, 1) * z)
            try:
                c, lower = cho_factor(A, check_finite=False)
                a_new = cho_solve((c, lower), b, check_finite=False)
            except np.linalg.LinAlgError:
                a_new = np.linalg.solve(A, b)
            if np.linalg.norm(a_new - a) < 1e-6:
                return a_new
            a = a_new
        return a

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n = X.shape[0]
        self.K_train = self._precompute_kernel(X)
        S_idx, R_idx = [], np.arange(n)
        err_H, a = [], np.zeros((0, 1))
        k = 0

        while True:
            m = len(S_idx)
            K_XS = self.K_train[:, S_idx] if m > 0 else np.zeros((n, 0))
            K_SS = self.K_train[np.ix_(S_idx, S_idx)] if m > 0 else np.zeros((0, 0))
            p = expit(K_XS @ a).reshape(-1) if m > 0 else expit(np.zeros(n))

            K_XR = self.K_train[:, R_idx]
            if self.proxy_type == "gradient":
                grad_all = K_XR.T @ (y - p)
            elif self.proxy_type == "var_grad":
                grad_all = K_XR.T @ ((y - p) * (p * (1 - p)))
            elif self.proxy_type == "fisher":
                W = p * (1 - p)
                grad_all = np.einsum("ij,i,ij->j", K_XR, W, K_XR)
            elif self.proxy_type == "wald":
                W = p * (1 - p)
                fisher_info = np.einsum("ij,i,ij->jk", K_XR, W, K_XR) + self.lambda_ * np.eye(m + 1)
                try:
                    c, lower = cho_factor(fisher_info, check_finite=False)
                    fisher_info_inv = cho_solve((c, lower), np.eye(m + 1), check_finite=False)
                except np.linalg.LinAlgError:
                    fisher_info_inv = np.linalg.inv(fisher_info)
                grad_all = (K_XR.T @ (y - p)).reshape(-1, 1)
                grad_all = (grad_all.T @ fisher_info_inv).reshape(-1)
            else:
                raise ValueError("Unknown proxy_type")

            ranks = min(self.top_k, len(R_idx))
            if ranks <= 0: 
                break
            
            top_local = np.argsort(-np.abs(grad_all))[:ranks]

            best_H, best_idx, best_a = np.inf, None, None
            for local_j in top_local:
                cand_idx = int(R_idx[local_j])
                K_a = np.hstack([K_XS, self.K_train[:, cand_idx:cand_idx+1]]) if m > 0 else self.K_train[:, cand_idx:cand_idx+1]
                if m == 0:
                    K_q = np.array([[self.K_train[cand_idx, cand_idx]]])
                else:
                    K_q = np.zeros((m+1, m+1))
                    K_q[:m, :m] = K_SS
                    K_q[m, m] = self.K_train[cand_idx, cand_idx]
                    K_q[:m, m] = self.K_train[np.ix_(S_idx, [cand_idx])].reshape(m,)
                    K_q[m, :m] = K_q[:m, m]
                a_init = np.vstack([a, [[0.0]]]) if m > 0 else np.zeros((1, 1))
                a_new = self._optimize_a(K_a, y, K_q, a_init=a_init)
                Ka_a_new = K_a @ a_new
                H = -np.dot(y, Ka_a_new.reshape(-1)) + np.sum(np.logaddexp(0, Ka_a_new)) + (self.lambda_/2) * float(a_new.T @ K_q @ a_new)
                if H < best_H:
                    best_H, best_idx, best_a = H, cand_idx, a_new

            if best_idx is None: 
                break
            S_idx.append(best_idx)
            R_idx = R_idx[R_idx != best_idx]
            err_H.append(best_H)
            a = best_a
            k += 1
            if self.verbose:
                print(f"Iter {k}: idx {best_idx}, |S|={len(S_idx)}, H={best_H:.6f}")
            if k > 1 and (abs(err_H[-1] - err_H[-2]) / (abs(err_H[-1]) + 1e-12)) < self.tol:
                if self.verbose: 
                    print("Converged")
                break
            if k >= self.max_sel:
                if self.verbose: 
                    print("Reached max_sel")
                break

        self.S_idx, self.err_H, self.a = S_idx, err_H, a
        return self

    def predict_proba(self, X):
        K_test = self._precompute_kernel(X, Y=self.X_train)
        return expit(K_test[:, self.S_idx] @ self.a).reshape(-1)

    def predict(self, X, thresh=0.5):
        return (self.predict_proba(X) >= thresh).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def evaluate(self, X, y):
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "logloss": log_loss(y, y_prob),
            "num_iv": len(self.S_idx)
        }
