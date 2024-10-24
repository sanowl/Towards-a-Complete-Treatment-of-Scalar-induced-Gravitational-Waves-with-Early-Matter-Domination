import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import gamma
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import logging
import warnings

warnings.filterwarnings("ignore")

@dataclass
class CosmologicalParams:
    H: float = 4e13
    Mpl: float = 2.4e18
    T_RH: float = 1e15
    N_e: int = 60
    rho_end: float = None
    k_end: float = None

    def __post_init__(self):
        self.rho_end = self.H**2 * self.Mpl**2
        self.k_end = self.H * np.exp(self.N_e)

class ScenarioBase:
    def __init__(self, params: CosmologicalParams):
        self.params = params
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def compute_power_spectrum(self, k: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Must be implemented by specific scenario")

class StochasticCurvaton(ScenarioBase):
    def __init__(self, params: CosmologicalParams, m2_H2: float = 0.2, lambda_: float = 0.1):
        super().__init__(params)
        self.m2_H2 = m2_H2
        self.lambda_ = lambda_
        self.logger.info(f"Initialized with m2_H2={self.m2_H2}, lambda_={self.lambda_}")

    def compute_eigenvalues(self, n_max: int = 10) -> np.ndarray:
        self.logger.info("Computing eigenvalues...")
        n = np.arange(1, n_max + 1)
        Lambda = self.m2_H2 * self.params.H * n
        return Lambda

    def compute_power_spectrum(self, k: np.ndarray) -> np.ndarray:
        self.logger.info("Computing power spectrum for Stochastic Curvaton...")
        Lambda = self.compute_eigenvalues()
        Delta2_S = np.zeros_like(k)
        for lambda_n in Lambda:
            g_n = 1.0
            exponent = (2 * lambda_n) / self.params.H
            gamma_term = gamma(2 - exponent)
            sin_term = np.sin(np.pi * exponent / 2)
            Delta2_S += (2 / np.pi) * g_n**2 * gamma_term * sin_term * (k / self.params.k_end)**exponent
        return Delta2_S

class MisalignedCurvaton(ScenarioBase):
    def __init__(self, params: CosmologicalParams, m2_H2: float = 0.4, chi0_end: float = 3.0):
        super().__init__(params)
        self.m2_H2 = m2_H2
        self.chi0_end = chi0_end * params.H
        self.logger.info(f"Initialized with m2_H2={self.m2_H2}, chi0_end={self.chi0_end}")

    def compute_power_spectrum(self, k: np.ndarray) -> np.ndarray:
        self.logger.info("Computing power spectrum for Misaligned Curvaton...")
        exponent = (2 * self.m2_H2) / (3 * self.params.H**2)
        Delta2_S = (self.params.H**2 / (np.pi**2 * self.chi0_end**2)) * (k / self.params.k_end)**exponent
        return Delta2_S

class RollingRadialMode(ScenarioBase):
    def __init__(self, params: CosmologicalParams, lambda_Phi: float = 5e-3,
                 fa: float = 6.0, m: float = 0.05):
        super().__init__(params)
        self.lambda_Phi = lambda_Phi
        self.fa = fa * params.H
        self.m = m * params.H
        self.logger.info(f"Initialized with lambda_Phi={self.lambda_Phi}, fa={self.fa}, m={self.m}")

    def compute_radial_evolution(self, t_max: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        self.logger.info("Computing radial mode evolution...")

        def deriv(t, y):
            s, ds_dt = y
            d2s_dt2 = -3 * self.params.H * ds_dt - self.lambda_Phi * (s**2 - self.fa**2) * s
            return [ds_dt, d2s_dt2]

        s0 = 10 * self.fa
        ds0_dt = 0.0
        y0 = [s0, ds0_dt]
        t_span = (0, t_max / self.params.H)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval, method='RK45')

        if not sol.success:
            self.logger.error("Radial mode evolution failed to integrate.")
            raise RuntimeError("Radial mode evolution integration failed.")

        return sol.t, sol.y[0]

    def compute_power_spectrum(self, k: np.ndarray) -> np.ndarray:
        self.logger.info("Computing power spectrum for Rolling Radial Mode...")
        k_star = 20.0
        k_c = 3.2e6
        Delta2_S = np.zeros_like(k)
        mask = k < k_c
        Delta2_S[mask] = (self.params.H**2 / (np.pi**2 * self.fa**2)) * (k[mask] / k_star)**0.5
        Delta2_S[~mask] = (self.params.H**2 / (np.pi**2 * self.fa**2))
        return Delta2_S

class GravitationalWaveComputation:
    def __init__(self, params: CosmologicalParams):
        self.params = params
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def compute_green_function(self, k: float, eta: np.ndarray) -> np.ndarray:
        eta_diff = np.subtract.outer(eta, eta)
        G = (np.sin(k * eta_diff) - k * eta_diff * np.cos(k * eta_diff)) / k
        G *= (eta_diff >= 0)
        return G

    def compute_source_term(self, k: float, eta: np.ndarray,
                            Phi: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
        w = 1/3
        dPhi_deta = np.gradient(Phi, eta)
        S = (12 * w * (1 - 3 * w) / (1 + w)) * v_rel**2 + \
            (2 * (5 + 3 * w) / (3 * (1 + w))) * Phi**2 + \
            (4 / (3 * (1 + w))) * Phi * dPhi_deta
        return S

    def compute_GW_spectrum(self, k_array: np.ndarray, Delta2_zeta: np.ndarray) -> np.ndarray:
        self.logger.info("Computing GW spectrum...")
        Omega_GW = np.zeros_like(k_array)

        for i, k in enumerate(k_array):
            if i % 100 == 0:
                self.logger.info(f"Processing k={k:.2e} ({i+1}/{len(k_array)})")
            eta = np.logspace(-10, 2, 500)
            Phi = self.compute_transfer_functions(k, eta)
            v_rel = self.compute_velocity_transfer(k, eta)
            S = self.compute_source_term(k, eta, Phi, v_rel)
            G = self.compute_green_function(k, eta)
            Omega_GW[i] = self.integrate_GW_spectrum(k, eta, G, S, Delta2_zeta[i])
        return Omega_GW

    def compute_transfer_functions(self, k: float, eta: np.ndarray) -> np.ndarray:
        return np.where(k * eta < 1, 1.0, np.sin(k * eta) / (k * eta))

    def compute_velocity_transfer(self, k: float, eta: np.ndarray) -> np.ndarray:
        return np.where(k * eta < 1, 0.0, np.cos(k * eta))

    def integrate_GW_spectrum(self, k: float, eta: np.ndarray, G: np.ndarray,
                              S: np.ndarray, Delta2_zeta: float) -> float:
        integrand = G * S * Delta2_zeta
        integral_eta_prime = np.trapz(integrand, eta, axis=1)
        integral = np.trapz(integral_eta_prime**2, eta)
        Omega_GW = (k**3 / (2 * np.pi**2)) * integral
        return Omega_GW

def plot_results(k, Delta2_S_stoch, Delta2_S_misal, Delta2_S_roll,
                 Omega_GW_stoch, Omega_GW_misal, Omega_GW_roll):
    logging.info("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    ax1.loglog(k, Delta2_S_stoch, label='Stochastic Curvaton')
    ax1.loglog(k, Delta2_S_misal, label='Misaligned Curvaton')
    ax1.loglog(k, Delta2_S_roll, label='Rolling Radial Mode')
    ax1.set_xlabel('k [Mpc$^{-1}$]', fontsize=14)
    ax1.set_ylabel('$\Delta^2_\\zeta(k)$', fontsize=14)
    ax1.set_title('Scalar Power Spectrum', fontsize=16)
    ax1.grid(True, which='both', ls='--', lw=0.5)
    ax1.legend(fontsize=12)
    f = k * 4.7e-3
    ax2.loglog(f, Omega_GW_stoch, label='Stochastic Curvaton')
    ax2.loglog(f, Omega_GW_misal, label='Misaligned Curvaton')
    ax2.loglog(f, Omega_GW_roll, label='Rolling Radial Mode')
    ax2.set_xlabel('Frequency [Hz]', fontsize=14)
    ax2.set_ylabel('$\\Omega_{GW,0}h^2$', fontsize=14)
    ax2.set_title('Induced Gravitational Wave Spectrum', fontsize=16)
    ax2.grid(True, which='both', ls='--', lw=0.5)
    ax2.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    logging.info("Plots generated successfully.")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting cosmological simulation...")
    params = CosmologicalParams()
    logging.info(f"Cosmological parameters: {params}")
    stochastic = StochasticCurvaton(params)
    misaligned = MisalignedCurvaton(params)
    rolling = RollingRadialMode(params)
    k = np.logspace(-4, 17, 1000)
    Delta2_S_stoch = stochastic.compute_power_spectrum(k)
    Delta2_S_misal = misaligned.compute_power_spectrum(k)
    Delta2_S_roll = rolling.compute_power_spectrum(k)
    gw_comp = GravitationalWaveComputation(params)
    Omega_GW_stoch = gw_comp.compute_GW_spectrum(k, Delta2_S_stoch)
    Omega_GW_misal = gw_comp.compute_GW_spectrum(k, Delta2_S_misal)
    Omega_GW_roll = gw_comp.compute_GW_spectrum(k, Delta2_S_roll)
    plot_results(k, Delta2_S_stoch, Delta2_S_misal, Delta2_S_roll,
                 Omega_GW_stoch, Omega_GW_misal, Omega_GW_roll)
    logging.info("Cosmological simulation completed successfully.")

if __name__ == "__main__":
    main()
