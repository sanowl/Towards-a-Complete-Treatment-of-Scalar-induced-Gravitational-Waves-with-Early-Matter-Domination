import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import gamma
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

@dataclass
class CosmologicalParams:
    """Cosmological parameters following paper conventions"""
    H: float = 4e13  # Hubble parameter during inflation (GeV)
    Mpl: float = 2.4e18  # Planck mass (GeV)
    T_RH: float = 1e15  # Reheating temperature (GeV)
    N_e: int = 60  # Number of e-folds
    rho_end: float = None  # Energy density at end of inflation
    k_end: float = None  # Comoving wavenumber at end of inflation
    
    def __post_init__(self):
        self.rho_end = self.H**2 * self.Mpl**2
        # Using e-folds to estimate k_end more accurately
        self.k_end = self.H * np.exp(self.N_e)

class ScenarioBase:
    """Base class for different cosmological scenarios"""
    def __init__(self, params: CosmologicalParams):
        self.params = params
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Prevent adding multiple handlers in interactive environments
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def compute_power_spectrum(self, k: np.ndarray) -> np.ndarray:
        """Compute the scalar power spectrum"""
        raise NotImplementedError("Must be implemented by specific scenario")

class StochasticCurvaton(ScenarioBase):
    """Implementation of Scenario 1: Stochastic Curvaton"""
    def __init__(self, params: CosmologicalParams, m2_H2: float = 0.2, lambda_: float = 0.1):
        super().__init__(params)
        self.m2_H2 = m2_H2  # m^2/H^2
        self.lambda_ = lambda_
        self.logger.info(f"Initialized with m2_H2={self.m2_H2}, lambda_={self.lambda_}")
        
    def compute_eigenvalues(self, n_max: int = 10) -> np.ndarray:
        """Compute eigenvalues of the Fokker-Planck equation"""
        self.logger.info("Computing eigenvalues...")
        # Placeholder for a proper eigenvalue computation
        # For demonstration, we'll use a harmonic oscillator analogy
        Lambda = np.array([(n + 0.5) * np.sqrt(self.m2_H2) for n in range(n_max)])
        self.logger.debug(f"Eigenvalues: {Lambda}")
        return Lambda
    
    def compute_power_spectrum(self, k: np.ndarray) -> np.ndarray:
        """Compute power spectrum following equation (3.4)"""
        self.logger.info("Computing power spectrum for Stochastic Curvaton...")
        Lambda = self.compute_eigenvalues()
        Delta2_S = np.zeros_like(k)
        
        for n, lambda_n in enumerate(Lambda):
            # Compute g_n from equation (3.6); here, we assume g_n = 1 for simplicity
            g_n = 1.0
            exponent = (2 * lambda_n) / self.params.H
            # Ensure the argument of gamma and sin is within valid range
            gamma_term = gamma(2 - exponent)
            sin_term = np.sin(np.pi * exponent / 2)
            Delta2_S += (2 / np.pi) * g_n**2 * gamma_term * sin_term * (k / self.params.k_end)**exponent
        
        self.logger.debug(f"Power spectrum computed: {Delta2_S}")
        return Delta2_S

class MisalignedCurvaton(ScenarioBase):
    """Implementation of Scenario 2: Misaligned Curvaton"""
    def __init__(self, params: CosmologicalParams, m2_H2: float = 0.4, chi0_end: float = 3.0):
        super().__init__(params)
        self.m2_H2 = m2_H2
        self.chi0_end = chi0_end * params.H
        self.logger.info(f"Initialized with m2_H2={self.m2_H2}, chi0_end={self.chi0_end}")
        
    def compute_power_spectrum(self, k: np.ndarray) -> np.ndarray:
        """Compute power spectrum following equation (4.7)"""
        self.logger.info("Computing power spectrum for Misaligned Curvaton...")
        exponent = (2 * self.m2_H2) / (3 * self.params.H**2)
        Delta2_S = (self.params.H**2 / (np.pi**2 * self.chi0_end**2)) * (k / self.params.k_end)**exponent
        self.logger.debug(f"Power spectrum computed: {Delta2_S}")
        return Delta2_S

class RollingRadialMode(ScenarioBase):
    """Implementation of Scenario 3: Rolling Radial Mode"""
    def __init__(self, params: CosmologicalParams, lambda_Phi: float = 5e-3,
                 fa: float = 6.0, m: float = 0.05):
        super().__init__(params)
        self.lambda_Phi = lambda_Phi
        self.fa = fa * params.H
        self.m = m * params.H
        self.logger.info(f"Initialized with lambda_Phi={self.lambda_Phi}, fa={self.fa}, m={self.m}")
        
    def compute_radial_evolution(self, t_max: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute evolution of radial mode following differential equations"""
        self.logger.info("Computing radial mode evolution...")
        
        def deriv(t, y):
            s, ds_dt = y
            d2s_dt2 = -3 * self.params.H * ds_dt - self.lambda_Phi * (s**2 - self.fa**2) * s
            return [ds_dt, d2s_dt2]
        
        # Initial conditions: s starts significantly displaced from fa
        s0 = 10 * self.fa
        ds0_dt = 0.0
        y0 = [s0, ds0_dt]
        
        # Time span: from 0 to t_max / H to capture dynamics
        t_span = (0, t_max / self.params.H)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval, method='RK45')
        
        if not sol.success:
            self.logger.error("Radial mode evolution failed to integrate.")
            raise RuntimeError("Radial mode evolution integration failed.")
        
        self.logger.debug("Radial mode evolution computed successfully.")
        return sol.t, sol.y[0]
    
    def compute_power_spectrum(self, k: np.ndarray) -> np.ndarray:
        """Compute power spectrum following equations (4.8)-(4.9)"""
        self.logger.info("Computing power spectrum for Rolling Radial Mode...")
        # Parameters for the spectrum
        k_star = 20.0  # Mpc^-1
        k_c = 3.2e6  # Mpc^-1
        
        # Initialize power spectrum
        Delta2_S = np.zeros_like(k)
        
        # Define mask for different k regimes
        mask = k < k_c
        
        # Power spectrum before cutoff
        Delta2_S[mask] = (self.params.H**2 / (np.pi**2)) * (1 / self.fa**2) * (k[mask] / k_star)**0.5  # Simplified slope
        
        # Power spectrum after cutoff (flattened)
        Delta2_S[~mask] = (self.params.H**2 / (np.pi**2)) * (1 / self.fa**2)
        
        self.logger.debug("Power spectrum computed.")
        return Delta2_S

class GravitationalWaveComputation:
    """Compute induced gravitational waves"""
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
        """Compute Green's function following equation (5.6)"""
        self.logger.debug(f"Computing Green's function for k={k}")
        # Vectorized computation using Heaviside step function
        eta_diff = np.subtract.outer(eta, eta)
        G = k * (np.cos(k * eta_diff) - 1) * (eta_diff >= 0)
        self.logger.debug(f"Green's function shape: {G.shape}")
        return G
    
    def compute_source_term(self, k: float, eta: np.ndarray, 
                            Phi: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
        """Compute source term following equation (5.5)"""
        self.logger.debug("Computing source term...")
        # Equation of state parameter (assuming radiation domination w=1/3)
        w = 1/3  
        
        # Compute derivatives
        dPhi_deta = np.gradient(Phi, eta)
        
        # Compute source term components
        S = (12 * w * (1 - 3 * w) / (1 + w)) * v_rel**2 + \
            (2 * (5 + 3 * w) / (3 * (1 + w))) * Phi**2 + \
            (4 / (3 * (1 + w))) * Phi * dPhi_deta
        
        self.logger.debug(f"Source term shape: {S.shape}")
        return S
    
    def compute_GW_spectrum(self, k: np.ndarray, Delta2_zeta: np.ndarray) -> np.ndarray:
        """Compute GW spectrum following equation (5.7)"""
        self.logger.info("Computing GW spectrum...")
        Omega_GW = np.zeros_like(k)
        
        for i, ki in enumerate(k):
            if i % 100 == 0:
                self.logger.info(f"Processing k={ki:.2e} ({i+1}/{len(k)})")
            
            eta = np.logspace(-10, 2, 1000)  # Example eta range from early to late times
            
            # Compute transfer functions
            Phi = self.compute_transfer_functions(ki, eta)
            v_rel = self.compute_velocity_transfer(ki, eta)
            
            # Compute source term
            S = self.compute_source_term(ki, eta, Phi, v_rel)
            
            # Compute Green's function
            G = self.compute_green_function(ki, eta)
            
            # Integrate to get GW spectrum
            Omega_GW[i] = self.integrate_GW_spectrum(ki, eta, G, S, Delta2_zeta[i])
        
        self.logger.info("GW spectrum computation completed.")
        return Omega_GW
    
    def compute_transfer_functions(self, k: float, eta: np.ndarray) -> np.ndarray:
        """Compute transfer functions for Phi"""
        self.logger.debug(f"Computing transfer functions for k={k}")
        # Simplified transfer function: 1 on super-horizon, (k*eta)^-2 on sub-horizon
        return np.where(k * eta < 1, 1.0, (k * eta)**-2)
    
    def compute_velocity_transfer(self, k: float, eta: np.ndarray) -> np.ndarray:
        """Compute transfer functions for velocity"""
        self.logger.debug(f"Computing velocity transfer for k={k}")
        # Simplified velocity transfer function: 0 on super-horizon, (k*eta)^-1 on sub-horizon
        return np.where(k * eta < 1, 0.0, (k * eta)**-1)
    
    def integrate_GW_spectrum(self, k: float, eta: np.ndarray, G: np.ndarray, 
                              S: np.ndarray, Delta2_zeta: float) -> float:
        """Perform the integration for GW spectrum"""
        self.logger.debug("Integrating GW spectrum...")
        # Compute integrand
        integrand = G * S * Delta2_zeta  # Shape: (eta, eta)
        self.logger.debug(f"Integrand shape: {integrand.shape}")
        
        # First integration over eta' (axis=1)
        integral_eta_prime = np.trapz(integrand, eta, axis=1)  # Shape: (eta,)
        self.logger.debug(f"Integral over eta' shape: {integral_eta_prime.shape}")
        
        # Second integration over eta
        integral = np.trapz(integral_eta_prime, eta)  # Scalar
        self.logger.debug(f"Final integral value: {integral}")
        
        # Compute Omega_GW using equation (5.7)
        Omega_GW = (k / (2 * np.pi))**2 * integral**2
        self.logger.debug(f"Omega_GW for k={k}: {Omega_GW}")
        return Omega_GW

def plot_results(k, Delta2_S_stoch, Delta2_S_misal, Delta2_S_roll,
                Omega_GW_stoch, Omega_GW_misal, Omega_GW_roll):
    """Create plots similar to those in the paper"""
    logging.info("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Power spectrum plot
    ax1.loglog(k, Delta2_S_stoch, label='Stochastic Curvaton')
    ax1.loglog(k, Delta2_S_misal, label='Misaligned Curvaton')
    ax1.loglog(k, Delta2_S_roll, label='Rolling Radial Mode')
    ax1.set_xlabel('k [Mpc$^{-1}$]', fontsize=14)
    ax1.set_ylabel('$\Delta^2_\zeta(k)$', fontsize=14)
    ax1.set_title('Scalar Power Spectrum', fontsize=16)
    ax1.grid(True, which='both', ls='--', lw=0.5)
    ax1.legend(fontsize=12)
    
    # GW spectrum plot
    # Convert k [Mpc^-1] to frequency [Hz]
    # Using relation: f = k / (2 * pi) * c / a, assuming c=1 and a normalized scale factor
    # Here, we use a simplified conversion
    # 1 Mpc ≈ 3.0857e22 meters
    # Hubble parameter H in GeV to Hz: 1 GeV ≈ 1.519e24 Hz
    # Thus, f = k * H / (2*pi) * (1.519e24) / (3.0857e22)
    # Simplifying constants:
    f = k * 4.7e-3  # Hz
    
    ax2.loglog(f, Omega_GW_stoch, label='Stochastic Curvaton')
    ax2.loglog(f, Omega_GW_misal, label='Misaligned Curvaton')
    ax2.loglog(f, Omega_GW_roll, label='Rolling Radial Mode')
    ax2.set_xlabel('Frequency [Hz]', fontsize=14)
    ax2.set_ylabel('$\Omega_{GW,0}h^2$', fontsize=14)
    ax2.set_title('Induced Gravitational Wave Spectrum', fontsize=16)
    ax2.grid(True, which='both', ls='--', lw=0.5)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    logging.info("Plots generated successfully.")

def main():
    # Set up logging for the main function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Starting cosmological simulation...")
    
    # Initialize cosmological parameters
    params = CosmologicalParams()
    logging.info(f"Cosmological parameters: {params}")
    
    # Initialize scenario instances
    stochastic = StochasticCurvaton(params)
    misaligned = MisalignedCurvaton(params)
    rolling = RollingRadialMode(params)
    
    # Define k values over a wide range
    k = np.logspace(-4, 17, 1000)  # Mpc^-1
    
    # Compute power spectra for each scenario
    Delta2_S_stoch = stochastic.compute_power_spectrum(k)
    Delta2_S_misal = misaligned.compute_power_spectrum(k)
    Delta2_S_roll = rolling.compute_power_spectrum(k)
    
    # Compute GW spectra
    gw_comp = GravitationalWaveComputation(params)
    Omega_GW_stoch = gw_comp.compute_GW_spectrum(k, Delta2_S_stoch)
    Omega_GW_misal = gw_comp.compute_GW_spectrum(k, Delta2_S_misal)
    Omega_GW_roll = gw_comp.compute_GW_spectrum(k, Delta2_S_roll)
    
    # Plot the results
    plot_results(k, Delta2_S_stoch, Delta2_S_misal, Delta2_S_roll,
                Omega_GW_stoch, Omega_GW_misal, Omega_GW_roll)
    
    logging.info("Cosmological simulation completed successfully.")

if __name__ == "__main__":
    main()
