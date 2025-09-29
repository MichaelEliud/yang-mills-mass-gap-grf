import numpy as np
from scipy.optimize import curve_fit
from scipy import linalg
import pickle
import time

# SU(3) matrix generator for gauge fields
def su3_matrix():
    h = np.random.randn(3, 3).astype(np.float32) + 1j * np.random.randn(3, 3).astype(np.float32)
    h = (h + h.conj().T) / 2
    u = linalg.expm(1j * 0.08 * h)
    u, _ = np.linalg.qr(u)
    return u / np.power(np.linalg.det(u), 1/3)

# Compute staples for Wilson action
def compute_staples(U, x, y, z, t, mu, Nx, Nt):
    staples = np.zeros((3, 3), dtype=np.complex64)
    for nu in range(4):
        if nu != mu:
            x_nu = (x + (nu==0), y + (nu==1), z + (nu==2), t + (nu==3))
            x_nu = (x_nu[0] % Nx, x_nu[1] % Nx, x_nu[2] % Nx, x_nu[3] % Nt)
            x_mu = (x + (mu==0), y + (mu==1), z + (mu==2), t + (mu==3))
            x_mu = (x_mu[0] % Nx, x_mu[1] % Nx, x_mu[2] % Nx, x_mu[3] % Nt)
            staple = U[x_nu[0], x_nu[1], x_nu[2], x_nu[3], nu] @ \
                     U[x_mu[0], x_mu[1], x_mu[2], x_mu[3], nu].conj().T @ \
                     U[x, y, z, t, mu].conj().T
            staples += staple
    return staples

# Cabibbo-Marinari update for SU(3) links
def cabibbo_marinari_update(beta_g, staples):
    u = su3_matrix()
    for _ in range(3):  # Multiple SU(2) subgroup updates
        delta_S = beta_g * np.real(np.trace(staples @ u - staples @ su3_matrix()))
        if np.random.rand() < np.exp(-delta_S):
            u = su3_matrix()
    return u

# Main lattice simulation with GRF stochastic metric
def lattice_ym_stochastic(Nx=32, Nt=64, beta=6.0, sigma=1e-5, kappa=1e-3, G=1/(1.22e19)**2, configs=5000):
    """
    Lattice QCD simulation for pure SU(3) Yang-Mills in GRF.
    Parameters:
        Nx: Spatial lattice size (default 32)
        Nt: Temporal lattice size (default 64)
        beta: 6/g_YM^2 (default 6.0)
        sigma: Stochastic field variance (default 1e-5)
        kappa: GRF coupling (default 1e-3)
        G: Gravitational constant (default 1/(1.22e19)^2 GeV^-2)
        configs: Number of configurations (default 5000)
    Returns:
        Delta_mean: Mass gap in GeV
        Delta_std: Standard deviation in GeV
        correlators: List of correlators for analysis
    """
    print(f"Starting simulation: {Nx}^3 x {Nt}, {configs} configurations")
    start_time = time.time()
    np.random.seed(int(time.time()))  # Random seed for reproducibility
    lattice_shape = (Nx, Nx, Nx, Nt, 4, 3, 3)
    U = np.array([su3_matrix() for _ in range(np.prod(lattice_shape[:-2]))], dtype=np.complex64).reshape(lattice_shape)
    # GRF stochastic metric: g(x) = 1 - 8pi G kappa xi^2 to regulate UV behavior
    xi = np.random.normal(0, sigma * 0.7, (Nx, Nx, Nx, Nt)).astype(np.float32)
    g = 1 - 8 * np.pi * G * kappa * xi**2
    Delta_values = []
    correlators = []

    for config in range(configs):
        for mu_batch in range(0, 4, 2):  # Batch 2 directions for memory
            for x in range(Nx):
                for y in range(Nx):
                    for z in range(Nx):
                        for t in range(Nt):
                            for mu in range(mu_batch, min(mu_batch + 2, 4)):
                                staples = compute_staples(U, x, y, z, t, mu, Nx, Nt)
                                U[x,y,z,t,mu] = cabibbo_marinari_update(beta * g[x,y,z,t], staples)
        
        if config % 1000 == 0 and config > 0:
            with open(f'lattice_config_{config}.pkl', 'wb') as f:
                pickle.dump(U, f)
            print(f"Checkpoint at config {config}, time: {(time.time() - start_time)/3600:.2f} hours")

        # Compute 0++ glueball correlator
        C = np.zeros(Nt, dtype=np.float32)
        for t in range(Nt):
            P_t = 0
            for x in range(Nx):
                for y in range(Nx):
                    for z in range(Nx):
                        for mu in range(3):
                            for nu in range(mu+1, 4):
                                P = U[x,y,z,t,mu] @ U[(x+(mu==0))%Nx, (y+(mu==1))%Nx, (z+(mu==2))%Nx, (t+(mu==3))%Nt, nu] @ \
                                    U[(x+(nu==0))%Nx, (y+(nu==1))%Nx, (z+(nu==2))%Nx, (t+(nu==3))%Nt, mu].conj().T @ \
                                    U[x,y,z,t,nu].conj().T
                                P_t += np.real(np.trace(P))
            P_t /= Nx**3 * 6
            P_0 = P_t if t == 0 else P_0
            C[t] = np.mean(P_t * P_0)
        
        correlators.append(C)
        if config % 500 == 0:
            with open(f'correlator_{config}.pkl', 'wb') as f:
                pickle.dump(C, f)
        
        def cosh_fit(t, A, Delta):
            return A * np.cosh(Delta * (t - Nt/2))
        try:
            popt, _ = curve_fit(cosh_fit, np.arange(1, 26), C[1:26], p0=[1, 1], maxfev=15000)
            Delta = popt[1]
            Delta_values.append(Delta)
            if config % 500 == 0:
                print(f"Config {config}, current Delta mean: {np.mean(Delta_values)/0.1:.3f} GeV")
        except:
            print(f"Fit failed at config {config}")

    Delta_mean = np.mean(Delta_values) / 0.1  # Convert to GeV
    Delta_std = np.std(Delta_values) / 0.1
    print(f"Simulation complete. Time: {(time.time() - start_time)/3600:.2f} hours")
    with open('correlators_final.pkl', 'wb') as f:
        pickle.dump(correlators, f)
    return Delta_mean, Delta_std, correlators

if __name__ == "__main__":
    Delta_mean, Delta_std, correlators = lattice_ym_stochastic()
    print(f"Mass gap Delta = {Delta_mean:.3f} ± {Delta_std:.3f} GeV")
