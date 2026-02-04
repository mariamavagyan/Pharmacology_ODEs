"""
ODE Fitting Tutorial for Clinical Pharmacology Modeling & Simulation Interview
============================================================================

This tutorial covers:
1. Creating synthetic pharmacokinetic/pharmacodynamic (PK/PD) data
2. Building ODE models from scratch
3. Fitting models to data
4. Evaluating model performance
5. Visualizing results

Key Concepts:
- Ordinary Differential Equations (ODEs) for drug dynamics
- Parameter estimation/optimization
- Model validation metrics
- Practical implementation in Python
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: SIMPLE ONE-COMPARTMENT PK MODEL
# ============================================================================

print("=" * 80)
print("PART 1: ONE-COMPARTMENT PHARMACOKINETIC MODEL")
print("=" * 80)

class OneCompartmentPKModel:
    """
    Simple one-compartment PK model with first-order absorption and elimination.
    
    dC/dt = -k_e * C
    
    Where:
    - C = drug concentration in central compartment
    - k_e = elimination rate constant
    - V = volume of distribution
    """
    
    def __init__(self, V=100, k_e=0.1):
        """
        Parameters:
        -----------
        V : volume of distribution (L)
        k_e : elimination rate constant (1/h)
        """
        self.V = V
        self.k_e = k_e
    
    def ode_system(self, C, t, k_a, dose, t_dose):
        """
        ODE system for one-compartment model with first-order absorption.
        
        Parameters:
        -----------
        C : concentration state
        t : time point
        k_a : absorption rate constant
        dose : dose amount
        t_dose : time of dose administration
        """
        if t < t_dose:
            dC_dt = 0
        else:
            # Amount in absorption compartment decays exponentially
            A_absorption = dose * np.exp(-k_a * (t - t_dose))
            # Transfer from absorption to central compartment
            dC_dt = (k_a * A_absorption) / self.V - self.k_e * C
        
        return dC_dt
    
    def simulate(self, t_span, dose, t_dose, k_a):
        """Simulate the model"""
        C0 = 0
        solution = odeint(self.ode_system, C0, t_span, args=(k_a, dose, t_dose))
        return solution.flatten()


# Generate synthetic PK data
print("\nGenerating synthetic one-compartment PK data...")

# True parameters
true_params_pk = {'V': 100, 'k_e': 0.1, 'k_a': 0.5}
model_pk = OneCompartmentPKModel(V=true_params_pk['V'], k_e=true_params_pk['k_e'])

# Simulate with true parameters
time_points = np.linspace(0, 24, 50)
dose = 500  # mg
t_dose = 0.5
y_true = model_pk.simulate(time_points, dose, t_dose, true_params_pk['k_a'])

# Add noise to simulate real data
np.random.seed(42)
noise = np.random.normal(0, 0.05 * np.max(y_true), len(y_true))
y_observed = y_true + noise
y_observed[y_observed < 0] = 0.01  # Can't have negative concentrations

print(f"  True parameters: V={true_params_pk['V']}, k_e={true_params_pk['k_e']}, k_a={true_params_pk['k_a']}")
print(f"  Max concentration: {np.max(y_observed):.2f} mg/L")
print(f"  Time to peak: {time_points[np.argmax(y_observed)]:.2f} hours")


# Define objective function for optimization
def objective_pk(params, time_points, y_observed, dose, t_dose):
    """Objective function to minimize (sum of squared errors)"""
    k_a, k_e = params
    
    # Validate parameters (must be positive)
    if k_a <= 0 or k_e <= 0:
        return 1e10
    
    V = true_params_pk['V']  # Keep volume fixed for simplicity
    model = OneCompartmentPKModel(V=V, k_e=k_e)
    y_pred = model.simulate(time_points, dose, t_dose, k_a)
    
    # Sum of squared errors
    sse = np.sum((y_observed - y_pred) ** 2)
    return sse


# Fit the model using optimization
print("\nFitting one-compartment PK model to synthetic data...")

# Initial guess
x0 = [0.3, 0.15]

# Use differential evolution for global optimization
result_pk = differential_evolution(
    objective_pk,
    bounds=[(0.01, 2), (0.01, 0.5)],
    args=(time_points, y_observed, dose, t_dose),
    seed=42,
    maxiter=1000,
    workers=1
)

fitted_k_a, fitted_k_e = result_pk.x
print(f"  Fitted k_a: {fitted_k_a:.4f} (true: {true_params_pk['k_a']:.4f})")
print(f"  Fitted k_e: {fitted_k_e:.4f} (true: {true_params_pk['k_e']:.4f})")

# Calculate model performance
model_fitted_pk = OneCompartmentPKModel(V=true_params_pk['V'], k_e=fitted_k_e)
y_fitted = model_fitted_pk.simulate(time_points, dose, t_dose, fitted_k_a)

rmse_pk = np.sqrt(mean_squared_error(y_observed, y_fitted))
r2_pk = r2_score(y_observed, y_fitted)

print(f"  RMSE: {rmse_pk:.4f} mg/L")
print(f"  R²: {r2_pk:.4f}")

# Visualize PK model fitting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data and fit
ax1.plot(time_points, y_observed, 'o', color='#2E86AB', label='Observed data', markersize=6)
ax1.plot(time_points, y_true, '--', color='green', label='True model', linewidth=2, alpha=0.7)
ax1.plot(time_points, y_fitted, '-', color='red', label='Fitted model', linewidth=2)
ax1.set_xlabel('Time (hours)', fontsize=11)
ax1.set_ylabel('Concentration (mg/L)', fontsize=11)
ax1.set_title('One-Compartment PK Model Fitting', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_observed - y_fitted
ax2.scatter(y_fitted, residuals, color='#2E86AB', s=50, alpha=0.6)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Fitted values (mg/L)', fontsize=11)
ax2.set_ylabel('Residuals (mg/L)', fontsize=11)
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/pk_model_fitting.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: pk_model_fitting.png")
plt.close()


# ============================================================================
# PART 2: TWO-COMPARTMENT PK/PD MODEL
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: TWO-COMPARTMENT PK/PD MODEL")
print("=" * 80)

class TwoCompartmentPKPDModel:
    """
    Two-compartment PK model with PD effect (e.g., tumor growth inhibition)
    
    Compartments:
    - Central (C_c): Drug in plasma/blood
    - Peripheral (C_p): Drug in tissue
    - Tumor (T): Tumor burden
    
    Equations:
    dC_c/dt = -(k_12 + k_e) * C_c + k_21 * C_p
    dC_p/dt = k_12 * C_c - k_21 * C_p
    dT/dt = λ * T - k_kill * C_c * T   (effect proportional to drug concentration)
    """
    
    def __init__(self, k_12=0.05, k_21=0.02, k_e=0.1, k_kill=0.001, lambda_growth=0.02):
        """
        Parameters:
        -----------
        k_12 : transfer from central to peripheral
        k_21 : transfer from peripheral to central
        k_e : elimination rate constant
        k_kill : drug efficacy (killing rate)
        lambda_growth : tumor growth rate
        """
        self.k_12 = k_12
        self.k_21 = k_21
        self.k_e = k_e
        self.k_kill = k_kill
        self.lambda_growth = lambda_growth
    
    def ode_system(self, state, t, dose, t_dose, V_c):
        """
        ODE system: [C_c, C_p, T]
        
        Parameters:
        -----------
        state : [C_c, C_p, T] - concentrations and tumor
        t : time
        dose : drug dose
        t_dose : time of dose
        V_c : central volume of distribution
        """
        C_c, C_p, T = state
        
        # Drug input (bolus at t_dose)
        if abs(t - t_dose) < 0.1:  # Approximation of bolus
            dC_c_dt = -((self.k_12 + self.k_e) * C_c - self.k_21 * C_p) + dose / V_c
        else:
            dC_c_dt = -(self.k_12 + self.k_e) * C_c + self.k_21 * C_p
        
        dC_p_dt = self.k_12 * C_c - self.k_21 * C_p
        
        # Tumor dynamics with drug effect
        dT_dt = self.lambda_growth * T - self.k_kill * C_c * T
        
        return [dC_c_dt, dC_p_dt, dT_dt]
    
    def simulate(self, t_span, dose, t_dose, V_c, T0=100):
        """Simulate the model"""
        state0 = [0, 0, T0]
        solution = odeint(self.ode_system, state0, t_span, args=(dose, t_dose, V_c))
        return solution


# Generate synthetic two-compartment PK/PD data
print("\nGenerating synthetic two-compartment PK/PD data...")

true_params_pkpd = {
    'k_12': 0.05, 'k_21': 0.02, 'k_e': 0.1,
    'k_kill': 0.001, 'lambda_growth': 0.02, 'V_c': 100
}

model_pkpd = TwoCompartmentPKPDModel(
    k_12=true_params_pkpd['k_12'],
    k_21=true_params_pkpd['k_21'],
    k_e=true_params_pkpd['k_e'],
    k_kill=true_params_pkpd['k_kill'],
    lambda_growth=true_params_pkpd['lambda_growth']
)

time_points_pkpd = np.linspace(0, 100, 100)
dose_pkpd = 1000
t_dose_pkpd = 0
solution_true = model_pkpd.simulate(time_points_pkpd, dose_pkpd, t_dose_pkpd, 
                                      true_params_pkpd['V_c'], T0=100)

# Extract tumor observations and add noise
T_true = solution_true[:, 2]
T_observed = T_true * np.exp(np.random.normal(0, 0.05, len(T_true)))

print(f"  True parameters: k_kill={true_params_pkpd['k_kill']}, λ={true_params_pkpd['lambda_growth']}")
print(f"  Initial tumor: {T_true[0]:.1f} mm³")
print(f"  Final tumor: {T_true[-1]:.1f} mm³")
print(f"  Tumor control: {(1 - T_true[-1]/T_true[0])*100:.1f}%")


# Define objective function for PK/PD model
def objective_pkpd(params, time_points, T_observed, dose, t_dose, V_c, T0):
    """Objective function for PK/PD fitting"""
    k_kill, lambda_growth = params
    
    if k_kill <= 0 or lambda_growth < 0:
        return 1e10
    
    # Fixed parameters
    model = TwoCompartmentPKPDModel(
        k_12=true_params_pkpd['k_12'],
        k_21=true_params_pkpd['k_21'],
        k_e=true_params_pkpd['k_e'],
        k_kill=k_kill,
        lambda_growth=lambda_growth
    )
    
    solution = model.simulate(time_points, dose, t_dose, V_c, T0=T0)
    T_pred = solution[:, 2]
    
    # Log-transformed error (common in PK/PD)
    sse = np.sum((np.log(T_observed) - np.log(T_pred)) ** 2)
    return sse


# Fit PK/PD model
print("\nFitting two-compartment PK/PD model...")

result_pkpd = differential_evolution(
    objective_pkpd,
    bounds=[(0.0001, 0.01), (0, 0.1)],
    args=(time_points_pkpd, T_observed, dose_pkpd, t_dose_pkpd, 
          true_params_pkpd['V_c'], T_true[0]),
    seed=42,
    maxiter=1000,
    workers=1
)

fitted_k_kill, fitted_lambda = result_pkpd.x
print(f"  Fitted k_kill: {fitted_k_kill:.6f} (true: {true_params_pkpd['k_kill']:.6f})")
print(f"  Fitted λ: {fitted_lambda:.6f} (true: {true_params_pkpd['lambda_growth']:.6f})")

# Get fitted predictions
model_fitted_pkpd = TwoCompartmentPKPDModel(
    k_12=true_params_pkpd['k_12'],
    k_21=true_params_pkpd['k_21'],
    k_e=true_params_pkpd['k_e'],
    k_kill=fitted_k_kill,
    lambda_growth=fitted_lambda
)
solution_fitted = model_fitted_pkpd.simulate(time_points_pkpd, dose_pkpd, t_dose_pkpd,
                                               true_params_pkpd['V_c'], T0=T_true[0])
T_fitted = solution_fitted[:, 2]

rmse_pkpd = np.sqrt(np.mean((T_observed - T_fitted) ** 2))
r2_pkpd = r2_score(T_observed, T_fitted)

print(f"  RMSE: {rmse_pkpd:.4f} mm³")
print(f"  R²: {r2_pkpd:.4f}")

# Visualize PK/PD model fitting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Tumor growth (linear scale)
ax1.plot(time_points_pkpd, T_observed, 'o', color='#2E86AB', label='Observed', markersize=4, alpha=0.7)
ax1.plot(time_points_pkpd, T_true, '--', color='green', label='True model', linewidth=2, alpha=0.7)
ax1.plot(time_points_pkpd, T_fitted, '-', color='red', label='Fitted model', linewidth=2)
ax1.set_xlabel('Time (days)', fontsize=11)
ax1.set_ylabel('Tumor burden (mm³)', fontsize=11)
ax1.set_title('Tumor Growth Inhibition Model', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Tumor growth (log scale)
ax2.semilogy(time_points_pkpd, T_observed, 'o', color='#2E86AB', label='Observed', markersize=4, alpha=0.7)
ax2.semilogy(time_points_pkpd, T_true, '--', color='green', label='True model', linewidth=2, alpha=0.7)
ax2.semilogy(time_points_pkpd, T_fitted, '-', color='red', label='Fitted model', linewidth=2)
ax2.set_xlabel('Time (days)', fontsize=11)
ax2.set_ylabel('Tumor burden (mm³, log scale)', fontsize=11)
ax2.set_title('Tumor Growth Inhibition (Log Scale)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('graphs/pkpd_model_fitting.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: pkpd_model_fitting.png")
plt.close()


# ============================================================================
# PART 3: MODEL VALIDATION AND DIAGNOSTICS
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: MODEL VALIDATION AND DIAGNOSTICS")
print("=" * 80)

def create_diagnostic_plots(y_obs, y_pred, model_name, filename):
    """Create comprehensive diagnostic plots"""
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Observed vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_obs, y_pred, alpha=0.6, color='#2E86AB', s=50)
    min_val, max_val = min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Observed', fontsize=10)
    ax1.set_ylabel('Predicted', fontsize=10)
    ax1.set_title('Observed vs Predicted', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Fitted
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_obs - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='#2E86AB', s=50)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Fitted values', fontsize=10)
    ax2.set_ylabel('Residuals', fontsize=10)
    ax2.set_title('Residuals vs Fitted', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot (normality of residuals)
    from scipy import stats
    ax3 = fig.add_subplot(gs[0, 2])
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of residuals
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(residuals, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Residuals', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Distribution of Residuals', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Absolute residuals vs fitted
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(y_pred, np.abs(residuals), alpha=0.6, color='#2E86AB', s=50)
    ax5.set_xlabel('Fitted values', fontsize=10)
    ax5.set_ylabel('Absolute Residuals', fontsize=10)
    ax5.set_title('Absolute Residuals vs Fitted', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Residuals vs Observation Order
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(range(len(residuals)), residuals, alpha=0.6, color='#2E86AB', s=50)
    ax6.axhline(y=0, color='r', linestyle='--', lw=2)
    ax6.set_xlabel('Observation Index', fontsize=10)
    ax6.set_ylabel('Residuals', fontsize=10)
    ax6.set_title('Residuals vs Order', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Calculate and display statistics
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
    mae = np.mean(np.abs(residuals))
    r2 = r2_score(y_obs, y_pred)
    
    stats_text = f"""
    MODEL DIAGNOSTICS: {model_name}
    
    RMSE (Root Mean Squared Error): {rmse:.6f}
    MAE (Mean Absolute Error): {mae:.6f}
    R² (Coefficient of Determination): {r2:.6f}
    
    Mean of Residuals: {np.mean(residuals):.6e} (should be close to 0)
    Std Dev of Residuals: {np.std(residuals):.6f}
    Min/Max Residuals: [{np.min(residuals):.6f}, {np.max(residuals):.6f}]
    """
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                  verticalalignment='center', bbox=dict(boxstyle='round', 
                  facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{model_name} - Diagnostic Plots', fontsize=13, fontweight='bold', y=0.995)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


# Create diagnostics for both models
create_diagnostic_plots(y_observed, y_fitted, 'One-Compartment PK Model',
                       'graphs/diagnostics_pk.png')

create_diagnostic_plots(T_observed, T_fitted, 'Two-Compartment PK/PD Model',
                       'graphs/diagnostics_pkpd.png')


# ============================================================================
# PART 4: SUMMARY AND KEY INTERVIEW TALKING POINTS
# ============================================================================

print("\n" + "=" * 80)
print("KEY CONCEPTS FOR YOUR INTERVIEW")
print("=" * 80)

summary_text = """
1. ODE SYSTEMS FOR PHARMACOLOGY
   - Used to model drug concentrations, effects, and disease dynamics
   - Multi-compartment models: central, peripheral, effect compartments
   - Key parameters: absorption (k_a), distribution (V), elimination (k_e)

2. PARAMETER ESTIMATION TECHNIQUES
   - Optimization algorithms: least squares, maximum likelihood
   - Global optimization: differential evolution, genetic algorithms
   - Local optimization: Nelder-Mead, Powell
   - Weighted vs unweighted fitting

3. MODEL PERFORMANCE METRICS
   - RMSE: magnitude of prediction errors
   - R²: proportion of variance explained (0-1)
   - MAE: average absolute error
   - AIC/BIC: model comparison and complexity trade-off

4. DIAGNOSTIC PLOTS INDICATE:
   - Observed vs Predicted: how well model captures data
   - Residuals: systematic biases (should be random)
   - Q-Q plot: whether residuals are normally distributed
   - Homoscedasticity: constant variance of residuals

5. COMMON PK/PD MODELS
   - Linear Compartmental Models: simple, interpretable
   - Nonlinear PK: Michaelis-Menten elimination
   - Pharmacodynamic: Emax model, indirect response models
   - Time-to-event models: survival analysis

6. BIOLOGICAL CONSIDERATIONS
   - First-pass metabolism
   - Protein binding effects
   - Active metabolites
   - Drug-drug interactions
   - Disease state effects on PK/PD

7. PRACTICAL IMPLEMENTATION
   - Use scipy.integrate.odeint() for ODE solving
   - scipy.optimize for parameter estimation
   - Always validate on independent data
   - Document assumptions and limitations

8. QUESTIONS TO ASK INTERVIEWER
   - What therapeutic areas do you focus on?
   - What are common challenges in your modeling projects?
   - How do you handle sparse sampling scenarios?
   - What tools/languages does your team use? (NONMEM, mrgsolve, etc.)
"""

print(summary_text)

# Save summary to file
with open('interview_preparation_notes.txt', 'w') as f:
    f.write("ODE FITTING FOR CLINICAL PHARMACOLOGY INTERVIEW\n")
    f.write("=" * 70 + "\n\n")
    f.write(summary_text)
    f.write("\n" + "=" * 70 + "\n")
    f.write("\nEXPECTED PARAMETER RECOVERY IN SYNTHETIC EXAMPLES\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"One-Compartment PK Model:\n")
    f.write(f"  True k_a: {true_params_pk['k_a']:.4f} → Fitted: {fitted_k_a:.4f}\n")
    f.write(f"  True k_e: {true_params_pk['k_e']:.4f} → Fitted: {fitted_k_e:.4f}\n")
    f.write(f"  Model R²: {r2_pk:.4f}\n\n")
    f.write(f"Two-Compartment PK/PD Model:\n")
    f.write(f"  True k_kill: {true_params_pkpd['k_kill']:.6f} → Fitted: {fitted_k_kill:.6f}\n")
    f.write(f"  True λ: {true_params_pkpd['lambda_growth']:.6f} → Fitted: {fitted_lambda:.6f}\n")
    f.write(f"  Model R²: {r2_pkpd:.4f}\n")

print("✓ Saved: interview_preparation_notes.txt\n")

print("=" * 80)
print("TUTORIAL COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. pk_model_fitting.png - One-compartment model results")
print("  2. pkpd_model_fitting.png - Two-compartment PK/PD results")
print("  3. diagnostics_pk.png - Diagnostic plots for PK model")
print("  4. diagnostics_pkpd.png - Diagnostic plots for PK/PD model")
print("  5. interview_preparation_notes.txt - Key concepts and talking points")
print("\nPractice explaining these concepts to someone!")
