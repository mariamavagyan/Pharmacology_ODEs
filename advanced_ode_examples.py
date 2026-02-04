"""
ADVANCED ODE FITTING: DISEASE MODELS & PRACTICAL CHALLENGES
============================================================================

This module covers more realistic scenarios you might encounter:
1. Disease progression models
2. Handling sparse data
3. Population heterogeneity
4. Model selection/comparison
5. Sensitivity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# EXAMPLE 1: CANCER CELL DYNAMICS WITH DRUG RESISTANCE
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: CANCER WITH DRUG RESISTANCE")
print("=" * 80)

class CancerResistanceModel:
    """
    Model with sensitive and resistant cancer cell populations.
    
    Sensitive cells respond to drug.
    Resistant cells may emerge under treatment pressure.
    """
    
    def __init__(self, lambda_s=0.05, lambda_r=0.01, mutation_rate=1e-6):
        self.lambda_s = lambda_s  # Growth rate sensitive cells
        self.lambda_r = lambda_r  # Growth rate resistant cells
        self.mutation_rate = mutation_rate
    
    def ode_system(self, state, t, C, k_kill_s, k_kill_r):
        """
        state = [S, R] - sensitive and resistant cells
        C = drug concentration at time t (interpolated)
        """
        S, R = state
        
        # Sensitive cells: growth minus drug effect minus mutation to resistance
        dS_dt = (self.lambda_s - k_kill_s * C) * S - self.mutation_rate * S
        
        # Resistant cells: growth plus mutations from sensitive, no drug effect
        dR_dt = self.lambda_r * R + self.mutation_rate * S
        
        return [dS_dt, dR_dt]
    
    def simulate(self, t_span, C_func, k_kill_s, k_kill_r, S0=1e6, R0=100):
        """
        C_func: function that returns concentration at time t
        """
        state0 = [S0, R0]
        
        # Create wrapper for ODE that evaluates C at each time
        def ode_wrapper(state, t):
            C = C_func(t)
            return self.ode_system(state, t, C, k_kill_s, k_kill_r)
        
        solution = odeint(ode_wrapper, state0, t_span)
        return solution


# Drug concentration function (from one-compartment model)
def drug_concentration(t, dose=1000, t_dose=0, k_a=0.5, k_e=0.1, V=100):
    """Compute drug concentration at time t"""
    if t < t_dose:
        return 0
    else:
        A_abs = dose * np.exp(-k_a * (t - t_dose))
        return (k_a * A_abs) / (V * (k_a - k_e)) * (np.exp(-k_e * (t - t_dose)) - np.exp(-k_a * (t - t_dose)))

# Simulate cancer resistance model
print("\nSimulating cancer model with drug resistance...")

model_cancer = CancerResistanceModel(lambda_s=0.05, lambda_r=0.01, mutation_rate=1e-6)
t_cancer = np.linspace(0, 100, 200)

# Create concentration function for the model
C_func = lambda t: drug_concentration(t, dose=1000, t_dose=0, k_a=0.5, k_e=0.1, V=100)

# Simulate
solution_cancer = model_cancer.simulate(t_cancer, C_func, k_kill_s=0.0005, k_kill_r=0, S0=1e7, R0=100)
S_true = solution_cancer[:, 0]
R_true = solution_cancer[:, 1]
T_total_true = S_true + R_true

# Add noise
T_total_obs = T_total_true * np.exp(np.random.normal(0, 0.08, len(T_total_true)))

print(f"  Initial total tumor: {T_total_true[0]:.2e} cells")
print(f"  Final total tumor: {T_total_true[-1]:.2e} cells")
print(f"  Sensitive cells at day 100: {S_true[-1]:.2e}")
print(f"  Resistant cells at day 100: {R_true[-1]:.2e}")
print(f"  Resistance ratio: {R_true[-1]/T_total_true[-1]*100:.2f}%")

# Visualize resistance emergence
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Total tumor burden
ax1.semilogy(t_cancer, T_total_obs, 'o', color='#2E86AB', markersize=3, label='Observed', alpha=0.6)
ax1.semilogy(t_cancer, T_total_true, '-', color='red', linewidth=2, label='True')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Total cells (log scale)')
ax1.set_title('Total Tumor Burden Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Population composition
ax2.semilogy(t_cancer, S_true, '-', color='blue', linewidth=2, label='Sensitive')
ax2.semilogy(t_cancer, R_true, '-', color='red', linewidth=2, label='Resistant')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Cell count (log scale)')
ax2.set_title('Sensitive vs Resistant Cell Populations')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Drug concentration over time
conc_array = np.array([C_func(t) for t in t_cancer])
ax3.plot(t_cancer, conc_array, 'g-', linewidth=2)
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Concentration (mg/L)')
ax3.set_title('Drug Concentration Profile')
ax3.grid(True, alpha=0.3)

# Plot 4: Fraction resistant over time
fraction_resistant = R_true / (S_true + R_true)
ax4.plot(t_cancer, fraction_resistant * 100, 'o-', color='purple', markersize=4)
ax4.set_xlabel('Time (days)')
ax4.set_ylabel('Fraction resistant (%)')
ax4.set_title('Emergence of Drug Resistance')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('/home/claude/cancer_resistance_model.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: cancer_resistance_model.png")
plt.close()


# ============================================================================
# EXAMPLE 2: VIRAL DYNAMICS (COVID-19 STYLE MODEL)
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: VIRAL DYNAMICS MODEL")
print("=" * 80)

class ViralDynamicsModel:
    """
    Target cell, infected cell, virus compartment model.
    dT/dt = λ - d_T*T - β*V*T
    dI/dt = β*V*T - δ*I
    dV/dt = p*I - c*V
    
    Where:
    T: target (uninfected) cells
    I: infected cells
    V: viral load
    """
    
    def __init__(self, lambda_source=1e4, d_T=0.1, beta=1e-4, delta=1, p=100, c=10):
        """
        lambda_source: cell production rate
        d_T: target cell death rate
        beta: infection rate
        delta: infected cell death rate
        p: virus production rate
        c: virus clearance rate
        """
        self.lambda_source = lambda_source
        self.d_T = d_T
        self.beta = beta
        self.delta = delta
        self.p = p
        self.c = c
    
    def ode_system(self, state, t, drug_efficacy=0):
        """
        drug_efficacy: reduces viral production (0-1, where 1 = complete block)
        """
        T, I, V = state
        
        dT_dt = self.lambda_source - self.d_T * T - self.beta * V * T
        dI_dt = self.beta * V * T - self.delta * I
        dV_dt = (1 - drug_efficacy) * self.p * I - self.c * V
        
        return [dT_dt, dI_dt, dV_dt]
    
    def simulate(self, t_span, drug_efficacy_func, T0=1e6, I0=1, V0=1):
        """
        drug_efficacy_func: function returning efficacy at time t
        """
        state0 = [T0, I0, V0]
        
        def ode_wrapper(state, t):
            efficacy = drug_efficacy_func(t)
            return self.ode_system(state, t, drug_efficacy=efficacy)
        
        solution = odeint(ode_wrapper, state0, t_span)
        return solution


# Drug efficacy function: reaches 80% by day 1
def viral_drug_efficacy(t, E_max=0.8, t_onset=0.5, t_half=0.5):
    """Sigmoidal drug efficacy"""
    return E_max / (1 + np.exp(-(t - t_onset) / t_half))

# Simulate viral dynamics
print("\nSimulating viral dynamics with antiviral treatment...")

model_viral = ViralDynamicsModel()
t_viral = np.linspace(0, 30, 150)

solution_viral = model_viral.simulate(t_viral, viral_drug_efficacy, T0=1e6, I0=1, V0=1)
T_viral = solution_viral[:, 0]
I_viral = solution_viral[:, 1]
V_viral = solution_viral[:, 2]

# Add noise to viral load (what we typically measure)
V_obs = V_viral * np.exp(np.random.normal(0, 0.15, len(V_viral)))
V_obs[V_obs < 1] = 1  # Detection limit

print(f"  Peak viral load: {np.max(V_viral):.2e} copies/mL")
print(f"  Day 30 viral load: {V_viral[-1]:.2e} copies/mL")
print(f"  Viral reduction: {(1 - V_viral[-1]/np.max(V_viral))*100:.1f}%")

# Visualize viral dynamics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Viral load
ax1.semilogy(t_viral, V_obs, 'o', color='#2E86AB', markersize=3, label='Observed', alpha=0.6)
ax1.semilogy(t_viral, V_viral, '-', color='red', linewidth=2, label='True')
ax1.axhline(y=50, color='gray', linestyle='--', label='Detection limit')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Viral load (copies/mL, log scale)')
ax1.set_title('Viral Load Reduction')
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Cell populations
ax2.semilogy(t_viral, T_viral, '-', color='blue', linewidth=2, label='Target (uninfected)')
ax2.semilogy(t_viral, I_viral, '-', color='orange', linewidth=2, label='Infected')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Cell count (log scale)')
ax2.set_title('Target vs Infected Cells')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Drug efficacy
efficacy_array = np.array([viral_drug_efficacy(t) for t in t_viral])
ax3.plot(t_viral, efficacy_array * 100, 'g-', linewidth=2)
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Drug efficacy (%)')
ax3.set_title('Antiviral Drug Efficacy')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 100])

# Plot 4: Log viral load with treatment window
ax4.semilogy(t_viral, V_viral, 'r-', linewidth=2.5, label='Viral load')
ax4.axvspan(0, 1, alpha=0.2, color='green', label='Drug onset')
ax4.set_xlabel('Time (days)')
ax4.set_ylabel('Viral load (copies/mL, log scale)')
ax4.set_title('Viral Dynamics with Treatment')
ax4.legend()
ax4.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/home/claude/viral_dynamics_model.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viral_dynamics_model.png")
plt.close()


# ============================================================================
# EXAMPLE 3: SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: SENSITIVITY ANALYSIS")
print("=" * 80)

def compute_sensitivity_indices(model_func, params, param_names, 
                                time_points, base_output, delta=0.05):
    """
    Compute local sensitivity indices for ODE parameters.
    
    Sensitivity = (% change in output) / (% change in parameter)
    """
    sensitivities = {name: [] for name in param_names}
    
    for i, (param_name, param_value) in enumerate(zip(param_names, params)):
        # Perturb parameter
        params_plus = params.copy()
        params_plus[i] = param_value * (1 + delta)
        
        # Evaluate output with perturbed parameter
        output_plus = model_func(params_plus)
        
        # Compute sensitivity
        sensitivity = (output_plus - base_output) / (delta * param_value)
        sensitivities[param_name] = sensitivity
    
    return sensitivities


# Example: sensitivity for one-compartment PK model
print("\nComputing parameter sensitivities...")

def pk_model_wrapper(params):
    """Wrapper for sensitivity analysis"""
    k_a, k_e = params
    
    def ode_wrapper(C, t):
        return -k_a * (C) + 0.5 * np.exp(-k_a * t)  # Simplified
    
    solution = odeint(ode_wrapper, 0, np.linspace(0, 24, 50))
    return solution[-1]  # Return final concentration

base_params = np.array([0.5, 0.1])
param_names = ['k_a', 'k_e']

base_output = pk_model_wrapper(base_params)
sensitivities = compute_sensitivity_indices(pk_model_wrapper, base_params, param_names,
                                           np.linspace(0, 24, 50), base_output, delta=0.05)

print("  Parameter Sensitivity Analysis Results:")
print("  (Final concentration sensitivity to parameter changes)")
for param, sensitivity in sensitivities.items():
    sens_val = float(sensitivity) if hasattr(sensitivity, '__len__') else sensitivity
    print(f"    {param}: {sens_val:.6f}")

# Visualize parameter impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sensitivity tornado diagram
param_list = list(sensitivities.keys())
sensitivity_values = [float(abs(sensitivities[p])) if isinstance(sensitivities[p], np.ndarray) 
                      else abs(sensitivities[p]) for p in param_list]

axes[0].barh(param_list, sensitivity_values, color='#2E86AB', alpha=0.7)
axes[0].set_xlabel('|Sensitivity Index|', fontsize=11)
axes[0].set_title('Parameter Sensitivity (Tornado Diagram)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Plot 2: One-way sensitivity analysis
k_a_range = np.linspace(0.2, 1.0, 20)
output_k_a = []

for k_a_val in k_a_range:
    output_k_a.append(pk_model_wrapper(np.array([k_a_val, 0.1])))

axes[1].plot(k_a_range, output_k_a, 'o-', color='#2E86AB', linewidth=2, markersize=6)
axes[1].axvline(x=0.5, color='red', linestyle='--', label='Base case', linewidth=2)
axes[1].set_xlabel('Absorption rate (k_a)', fontsize=11)
axes[1].set_ylabel('Final concentration (mg/L)', fontsize=11)
axes[1].set_title('One-Way Sensitivity Analysis', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sensitivity_analysis.png")
plt.close()


# ============================================================================
# EXAMPLE 4: POPULATION PK (HANDLING INDIVIDUAL VARIABILITY)
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: POPULATION PHARMACOKINETICS")
print("=" * 80)

print("\nSimulating population heterogeneity...")

# Population parameters (mean, CV%)
pop_params = {
    'V': {'mean': 100, 'cv': 30},
    'k_e': {'mean': 0.1, 'cv': 40},
    'k_a': {'mean': 0.5, 'cv': 35}
}

# Generate 50 individual parameter sets
n_individuals = 50
np.random.seed(42)

individual_params = {}
for param_name, param_info in pop_params.items():
    mean = param_info['mean']
    cv = param_info['cv'] / 100
    # Log-normal distribution for PK parameters
    log_mean = np.log(mean / np.sqrt(1 + cv**2))
    log_sd = np.sqrt(np.log(1 + cv**2))
    individual_params[param_name] = np.exp(np.random.normal(log_mean, log_sd, n_individuals))

# Simulate individuals
time_points_pop = np.linspace(0, 24, 50)
individual_profiles = []

for i in range(n_individuals):
    model = OneCompartmentPKModel(V=individual_params['V'][i], 
                                  k_e=individual_params['k_e'][i])
    y_sim = model.simulate(time_points_pop, dose=500, t_dose=0.5, 
                          k_a=individual_params['k_a'][i])
    individual_profiles.append(y_sim)

individual_profiles = np.array(individual_profiles)

# Calculate population statistics
median_profile = np.median(individual_profiles, axis=0)
percentile_5 = np.percentile(individual_profiles, 5, axis=0)
percentile_95 = np.percentile(individual_profiles, 95, axis=0)

# Visualize population distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Individual trajectories with population distribution
for i in range(n_individuals):
    ax1.plot(time_points_pop, individual_profiles[i], color='#2E86AB', alpha=0.15)

ax1.plot(time_points_pop, median_profile, 'r-', linewidth=3, label='Median')
ax1.fill_between(time_points_pop, percentile_5, percentile_95, 
                 color='red', alpha=0.2, label='5th-95th percentile')
ax1.set_xlabel('Time (hours)', fontsize=11)
ax1.set_ylabel('Concentration (mg/L)', fontsize=11)
ax1.set_title('Population Pharmacokinetics', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution of key PK parameters
param_dist = pd.DataFrame(individual_params)
ax2.violinplot([param_dist['V'], param_dist['k_e'], param_dist['k_a']], 
               positions=[1, 2, 3], showmeans=True)
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['Volume (L)', 'Elimination (1/h)', 'Absorption (1/h)'])
ax2.set_ylabel('Parameter Value', fontsize=11)
ax2.set_title('Parameter Distributions Across Population', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/population_pk.png', dpi=300, bbox_inches='tight')
print(f"\n  Generated {n_individuals} individual profiles")
print(f"  Volume: {individual_params['V'].mean():.1f} ± {individual_params['V'].std():.1f} L")
print(f"  k_e: {individual_params['k_e'].mean():.4f} ± {individual_params['k_e'].std():.4f} 1/h")
print(f"  k_a: {individual_params['k_a'].mean():.4f} ± {individual_params['k_a'].std():.4f} 1/h")
print("✓ Saved: population_pk.png")
plt.close()


# ============================================================================
# INTERVIEW QUESTION RESPONSES GUIDE
# ============================================================================

print("\n" + "=" * 80)
print("COMMON INTERVIEW QUESTIONS & HOW TO ANSWER")
print("=" * 80)

interview_guide = """
Q1: "What's the difference between linear and nonlinear pharmacokinetics?"

A: Linear PK follows first-order kinetics where elimination rate is proportional to 
   concentration. Nonlinear PK occurs when elimination mechanisms saturate (like 
   enzyme saturation), following Michaelis-Menten kinetics. In practice, most drugs 
   are linear at therapeutic doses, but some (like phenytoin, aspirin) show 
   nonlinearity at high concentrations.

---

Q2: "How do you decide which model structure to use?"

A: Start simple (one-compartment), then increase complexity if:
   - Goodness of fit metrics (R², RMSE) indicate poor fit
   - Diagnostic plots show systematic bias
   - Mechanistic understanding suggests additional compartments
   - Visual inspection of data suggests multi-phase elimination
   
   Use AIC/BIC for model selection to penalize complexity.

---

Q3: "What do you do when you have sparse data?"

A: With sparse data (few samples per individual):
   - Use population approaches (NONMEM, Stan, Monolix)
   - Pool information across subjects using hierarchical models
   - Use Bayesian methods to incorporate prior knowledge
   - Consider simpler model structures with fewer parameters to estimate
   - Validate predictions on held-out data

---

Q4: "How do you validate that your model is appropriate?"

A: Multiple validation approaches:
   1. Internal validation: residual analysis, diagnostic plots, goodness-of-fit
   2. Visual predictive checks: compare observed data distribution to simulations
   3. Bootstrap: resampling to assess parameter uncertainty
   4. External validation: test on independent dataset (if available)
   5. Sensitivity analysis: confirm predictions robust to parameter uncertainty

---

Q5: "What are the main challenges in PK/PD modeling?"

A: Common challenges:
   - Limited sampling: expensive/invasive to get many measurements
   - Variability: inter-individual differences, intra-individual variability
   - Model identifiability: not all parameters uniquely estimable from data
   - Mechanistic knowledge gaps: don't always know true underlying biology
   - Computational complexity: large-scale optimization in high dimensions
   - Clinical constraints: can't do invasive sampling in real patients

---

Q6: "How would you handle outliers or measurements below detection limit?"

A: 
   - Outliers: visual inspection first, consider measurement error vs true biological effect
     Use robust fitting methods, M-estimation, or censored regression
   - Below detection limit (BDL): 
     * Ignore: if few BDL values
     * Censored likelihood: model probability of being BDL
     * Imputation: use lower bound or limit of detection
     * Maximum likelihood: native handling in NONMEM, Stan

---

Q7: "What's your experience with [specific tool: NONMEM/R/Python/etc]?"

A: (Tailor to job description)
   Python: "I'm comfortable with scipy.integrate.odeint for ODE solving and 
   scipy.optimize for parameter fitting. I also use pandas for data management, 
   matplotlib for visualization, and could use Stan or PyMC for Bayesian approaches."
   
   R: "Experience with dplyr/tidyverse for data manipulation, ggplot2 for 
   visualization, mrgsolve for PK simulation, and nlmixr for mixed-effects modeling."

---

Q8: "Tell me about a modeling challenge you overcame."

A: (Prepare 2-3 examples with structure: Problem → Approach → Solution → Learning)
   
   Example structure:
   "I was fitting a two-compartment model but the peripheral compartment parameters 
   were not well-identifiable. I diagnosed this by examining correlation matrices 
   and likelihood profiles. I resolved it by fixing the peripheral elimination to 
   match literature values, reducing parameters to estimate. This improved 
   convergence and parameter precision while maintaining fit quality."

---

Q9: "How do you communicate modeling results to non-modelers?"

A: 
   - Create interpretable visualizations (observed vs predicted, not just equations)
   - Summarize key findings in plain language
   - Focus on clinical implications, not statistical details
   - Use sensitivity analysis to show what matters most
   - Highlight uncertainty/confidence intervals
   - Relate back to disease/treatment questions they care about

---

Q10: "What would you want to learn in this internship?"

A: (Research the company/role first)
   Good answer: "I'm interested in learning how you approach population PK 
   modeling in rare diseases, where sample sizes are limited. I'd also like 
   exposure to your specific tools and workflows, and understanding how modeling 
   influences drug development decisions. I'm particularly interested in [mention 
   specific therapeutic area if you know it]."
"""

print(interview_guide)

# Save to file
with open('/home/claude/interview_q_and_a.txt', 'w') as f:
    f.write("CLINICAL PHARMACOLOGY MODELING INTERVIEW Q&A\n")
    f.write("=" * 70 + "\n\n")
    f.write(interview_guide)

print("\n✓ Saved: interview_q_and_a.txt")

print("\n" + "=" * 80)
print("ADVANCED EXAMPLES COMPLETE!")
print("=" * 80)
