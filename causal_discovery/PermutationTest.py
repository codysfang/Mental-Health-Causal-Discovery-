import torch
from torch.distributions import Normal, MultivariateNormal
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as npl

class PseudoLikelihood(object):
    std_gaussian = Normal(0, 1)

    def __init__(self, X, mask):
        torch.manual_seed(100)
        self.size = X.shape[1]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.theta = torch.eye(self.size, self.size, requires_grad=True)
        self.theta.to(self.device)
        self.data = torch.tensor(X)
        self.is_cont_mask = mask # 1 if cont otherwise discrete
        self.R_hat = torch.cov(self.data, correction=0)
        self.N = X.shape[0]
        self.thresholds = None #T_hat_l is the threshold between l-1 to l

    def calculate_thresholds(self):
        self.thresholds = {}
        for i in range(self.size):
            if not self.is_cont_mask[i]:
                col = self.data[:, i]
                threshold_map = [] 
                max_val = int(torch.max(col).item())
                for k in range(max_val + 2):
                    if k == 0:
                        threshold_map.append(-torch.inf)
                    elif k == max_val + 1 or k == max_val + 2:
                        threshold_map.append(torch.inf)
                    else:
                        leq_array = (col <= k - 1)
                        threshold_map.append(self.std_gaussian.icdf(torch.sum(leq_array)/self.N))
                self.thresholds[i] = torch.tensor(threshold_map, requires_grad=False)
        return


    def get_loss(self, X, theta):

        R = self.get_R(theta)
        def numerical_integrate(lx, ly, ux, uy, gaussian, grid=4):
            # This roughly gives the cdf
            batch_num = lx.shape[0]
            clipped_lx = torch.clamp(lx, min=-99, max=99)
            clipped_ly = torch.clamp(ly, min=-99, max=99)
            clipped_ux = torch.clamp(ux, min=-99, max=99)
            clipped_uy = torch.clamp(uy, min=-99, max=99)
            volumes = []
            for i in range(batch_num):
                x1 = clipped_lx[i]; x2= clipped_ux[i]; y1 = clipped_ly[i]; y2= clipped_uy[i]
                x_coords = torch.linspace(x1, x2, grid)
                y_coords = torch.linspace(y1, y2, grid)
                x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
                coords = torch.stack([x.flatten(), y.flatten()], dim=1)

                log_area = torch.log((x2 - x1) / grid * (y2 - y1) / grid)
                log_joint_evals = torch.logsumexp(gaussian.log_prob(coords) + log_area, dim=0)
        
                volumes.append(log_joint_evals)

            return torch.stack(volumes)

        def pairwise_likelihood(a, b):
            r_ab = R[a, b]
            row1 = torch.stack([torch.tensor(1.0), r_ab])
            row2 = torch.stack([r_ab, torch.tensor(1.0)])
            r = torch.stack([row1, row2])
            if self.is_cont_mask[a] and self.is_cont_mask[b]:
                r_hat = torch.Tensor([[1, self.R_hat[a, b]], [self.R_hat[a, b], 1]])
                p = 0.5 * torch.trace(torch.inverse(r) @ r_hat) + torch.log(torch.det(r))
            elif self.is_cont_mask[a] or self.is_cont_mask[b]:
                discrete_var = a if not self.is_cont_mask[a] else b
                cont_var = a if self.is_cont_mask[a] else b
                disc_tensor = X[:, discrete_var].clone().long()
                T_t = torch.take(self.thresholds[discrete_var], disc_tensor)
                T_t_plus_one = torch.take(self.thresholds[discrete_var], disc_tensor + 1)
                stable_val = torch.clamp(1 - R[a, b] ** 2, min=1e-04)
                scale = 1 / stable_val
                T_t = torch.clamp(T_t, min=-99, max=99)
                T_t_plus_one = torch.clamp(T_t_plus_one, min=-99, max=99)
                p = (1 / self.N) * torch.sum(
                    self.std_gaussian.log_prob(X[:, cont_var]) + torch.log(
                        (
                            self.std_gaussian.cdf(scale * (T_t_plus_one - R[a, b] * X[:, cont_var])) - 
                            self.std_gaussian.cdf(scale * (T_t - R[a, b] * X[:, cont_var]))
                        )
                    )
                )
            else:
                tensor_a = X[:, a].clone().long()
                tensor_b = X[:, b].clone().long()
                T_a_t = torch.take(self.thresholds[a], tensor_a)
                T_b_t = torch.take(self.thresholds[b], tensor_b)
                T_a_t_plus_one = torch.take(self.thresholds[a], tensor_a + 1)
                T_b_t_plus_one = torch.take(self.thresholds[b], tensor_b + 1)
                mvr_gaussian = MultivariateNormal(loc=torch.tensor([0, 0]), covariance_matrix=r)
                p = (1 / self.N) * torch.sum(
                    numerical_integrate(T_a_t, T_b_t, T_a_t_plus_one, T_b_t_plus_one, mvr_gaussian)
                )
            return p

        loss = 0
        for i in range(self.size - 1):
            for j in range(i+1,self.size):
                loss -= pairwise_likelihood(i, j)

        # diag_sum = torch.sum(torch.abs(torch.diagonal(theta)))
        # + lreg * diag_sum
        return loss
    
    def get_R(self, theta):
        theta_constrained = theta.clone()
        theta_constrained = theta_constrained - torch.diag(torch.diagonal(theta_constrained))
        U = torch.zeros_like(theta)
        for i in range(U.shape[0]):
            U[i, i] = torch.cos(theta_constrained[0, i])
            sin_values_pre_computed = torch.sin(theta_constrained[:i, i])
            cumulative_sin_prod = torch.cumprod(sin_values_pre_computed, dim=0)
            for j in range(i):
                U[j, i] = torch.cos(theta_constrained[i-j, i]) * cumulative_sin_prod[i-j-1]
        R = torch.matmul(U.T, U)
        return R
    
    def get_optimised_cov(self, rate=0.035, epochs=8, display=False, clipping=False):
        batched_data = DataLoader(self.data, batch_size=5, shuffle=True)
        self.calculate_thresholds()
        optimizer = torch.optim.Adam([self.theta], lr=rate)
        iterate = range(epochs) if not display else tqdm(range(epochs), desc="Gradient Progress")
        loss_hist = []
        torch.autograd.set_detect_anomaly(True)
        for _ in iterate:
            total_loss = 0
            for batch in batched_data:
                optimizer.zero_grad()
                loss = self.get_loss(batch, self.theta)
                loss.backward()
                if clipping:
                    torch.nn.utils.clip_grad_norm_([self.theta], max_norm=100)
                optimizer.step()
                total_loss += loss.item()
            
            loss_hist.append(total_loss)
        R_tensor = self.get_R(self.theta).detach()
        return R_tensor.numpy(), self.theta



class PermutationTest(object):

    def __init__(self, data, mask, clipping=False):

        self.data = data
        self.N = data.shape[0]
        self.cached_dict = {}

        estimator = PseudoLikelihood(data, mask)
        self.R, _= estimator.get_optimised_cov(display=True, clipping=clipping)
        self.R = self.R.astype(np.float64)

        self.data = self.data - self.data.mean(axis=0)
        self.data = self.data/(2 * self.data.std(axis=0))

    def get_cachekey(self, pcols_, qcols_):

        pcols = pcols_.copy()
        qcols = qcols_.copy()
        pcols.sort()
        qcols.sort()

        key = ''
        for i in range(self.data.shape[1]):
            if i in pcols and i in qcols:
                key = key+str(3)
            elif i in pcols and not i in qcols:
                key = key+str(2)
            elif not i in pcols and i in qcols:
                key = key+str(1)
            else:
                key = key+str(0)

        return key
    
    def test(self, pcols, qcols, r, alpha, total_sample=10000):

        def compute_lambda(N, P, Q, r_vec):
            scale_factor = -(N - (P + Q + 3) / 2)
            prod = np.prod(1 - r_vec ** 2)
            return scale_factor * np.log(prod)

        def root_inv(M):
            e_vals, P = npl.eig(M)
            inv_e_vals = 1 / np.sqrt(e_vals)
            inv_e_vals = np.diag(inv_e_vals)
            res = P @ inv_e_vals @ P.T
            mask = np.abs(res) < 1e-12
            res[mask] = 0
            return res
        
        cachekey = self.get_cachekey(pcols, qcols)
        if cachekey in self.cached_dict:
            sigma_p, sigma_q, sigma_p_q, A, B, perm_corr_matrices = self.cached_dict[cachekey]
        else:
            sigma_p = self.R[np.ix_(pcols, pcols)]
            sigma_q = self.R[np.ix_(qcols, qcols)]
            sigma_p_q = self.R[np.ix_(pcols, qcols)]

            U, _, V = npl.svd(root_inv(sigma_p) @ sigma_p_q @ root_inv(sigma_q) + 1e-04 * np.eye(sigma_p.size[0], sigma_q.size[0]))
            A = root_inv(sigma_p).T @ U
            B = root_inv(sigma_q).T @ V.T

            perm_corr_matrices = []
            rng = np.random.default_rng(seed=12345)
            for i in range(total_sample):
                if i == 0:
                    perm_corr_matrices.append(self.data[:, pcols].T @ self.data[:, qcols])
                    continue
                perm = rng.permutation(self.N)
                data_copy = self.data.copy()
                data_copy[:, qcols] = data_copy[np.ix_(perm, qcols)]
                # pl_est = PseudoLikelihood(
                #     data_copy[:, pcols + qcols], mask=self.cont_mask[pcols + qcols]
                # )
                # perm_mle_corr, _ = pl_est.get_optimised_cov()
                perm_corr = data_copy[:, pcols].T @ data_copy[:, qcols]
                perm_corr_matrices.append(perm_corr)

            self.cached_dict[cachekey] = (sigma_p, sigma_q, sigma_p_q, A, B, perm_corr_matrices)

        
        p_col_size = len(pcols)
        q_col_size = len(qcols)
        K = min(q_col_size, p_col_size)


        def get_r_vector(permute_R_pq):
            m1 = A.T @ sigma_p @ A
            m2 = (1 / (self.N - 1)) * A.T @ permute_R_pq[:p_col_size, :q_col_size] @ B
            m3 = B.T @ sigma_q @ B
            sub_product = root_inv(m1[r:K, r:K]) @ m2[r:K, r:K] @ root_inv(m3[r:K, r:K])
            _, singular, _ = npl.svd(sub_product)
            if sum(singular >= 1) > 0:
                return -1
            return np.diag(singular)
        

        l_base = compute_lambda(self.N, p_col_size, q_col_size, get_r_vector(perm_corr_matrices[0]))
        num_geq = 0
        for m_p in perm_corr_matrices:
            r_vec = get_r_vector(m_p)
            if isinstance(r_vec, int):
                total_sample -= 1
                continue
            l_perm = compute_lambda(self.N, p_col_size, q_col_size, r_vec)
            if l_perm >= l_base:
                num_geq += 1

        # Return true if failing to reject
        return (num_geq / total_sample) >= alpha



# if __name__ == "__main__":
#     X = np.random.randint(low=0, high=5, size=(10000, 10))
#     import time
#     start = time.time()
#     testt = PermutationTest(X, mask=np.array([True for i in range(10)]))
#     print(testt.test([1,3,4], [2,5,8], 2, 0.25))
#     print(time.time() - start)