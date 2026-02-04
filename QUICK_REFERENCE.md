# Clinical Pharmacology Modeling - Quick Reference Card

Keep this handy during your interview prep!

---

## 🔑 Key Equations

### One-Compartment PK
```
dC/dt = -k_e * C
Solution: C(t) = C₀ * e^(-k_e*t)

Key parameters:
- k_e: elimination rate constant (1/hour)
- V: volume of distribution (L)
- Clearance (CL) = k_e * V
- Half-life (t₁/₂) = 0.693 / k_e
```

### Two-Compartment PK
```
dC_c/dt = -(k₁₂ + k_e)*C_c + k₂₁*C_p
dC_p/dt = k₁₂*C_c - k₂₁*C_p

- C_c: central compartment (blood)
- C_p: peripheral compartment (tissue)
```

### Pharmacodynamic (Emax) Model
```
E = E_max * C / (EC₅₀ + C)

- E: effect (0 to E_max)
- C: drug concentration
- EC₅₀: concentration at 50% max effect
```

### Parameter Fitting
```
Objective: Minimize SSE = Σ(y_obs - y_pred)²

R² = 1 - (Σ(y_obs - y_pred)²) / (Σ(y_obs - ȳ)²)
   Range: 0 to 1 (higher is better)

RMSE = √(1/n * Σ(y_obs - y_pred)²)
```

---

## 📊 Diagnostic Plots - What They Mean

| Plot | Good Sign | Bad Sign | Action |
|------|-----------|----------|--------|
| Observed vs Predicted | Points on y=x line | Curved pattern | Try different model |
| Residuals vs Fitted | Random scatter at 0 | Curved pattern | May need transformation |
| Histogram of Residuals | Normal, centered at 0 | Bimodal/skewed | Check error distribution |
| Q-Q Plot | Points on diagonal | S-shape | Normality violated |
| Scale-Location | Random scatter | Funnel shape | Variance not constant |
| Residuals vs Order | Random | Cyclic pattern | Measurements not independent |

---

## 🐍 Python Code Templates

### Import Essentials
```python
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
```

### Define ODE System
```python
def ode_system(state, t, params):
    """Define differential equations"""
    C = state
    k_e = params['k_e']
    dC_dt = -k_e * C
    return dC_dt

# Solve
t_span = np.linspace(0, 24, 100)
C_solution = odeint(ode_system, C0, t_span, args=(params,))
```

### Fit Model to Data
```python
def objective(params, t_obs, y_obs):
    """Minimize sum of squared errors"""
    y_pred = odeint(ode_system, y0, t_obs, args=(params,)).flatten()
    sse = np.sum((y_obs - y_pred)**2)
    return sse

# Global optimization
result = differential_evolution(objective, bounds=[(0.01, 1)],
                              args=(t_obs, y_obs))
best_params = result.x
```

### Calculate Metrics
```python
rmse = np.sqrt(np.mean((y_obs - y_pred)**2))
r2 = 1 - (np.sum((y_obs - y_pred)**2) / 
          np.sum((y_obs - y_obs.mean())**2))
```

---

## 💬 Common Interview Answers (Short Version)

**"What's a one-compartment model?"**
> A simplified representation assuming drug distributes evenly throughout the body. Good starting point, assumes first-order elimination.

**"Why use two compartments?"**
> When drug distributes into tissue slowly. Captures biphasic concentration decline: fast initial distribution, slow elimination.

**"What's the difference between PK and PD?"**
> PK is what the body does to the drug (absorption, distribution, elimination). PD is what the drug does to the body (effect on biomarkers/disease).

**"How do you know if your model is good?"**
> Diagnostic plots, R² metric, visual inspection of data vs fit, residual analysis for patterns.

**"What does this residual plot mean?"**
> Residuals should be randomly scattered around zero. If there's a pattern, the model systematically over/under-predicts certain regions.

**"How do you fit parameters?"**
> Minimize the sum of squared differences between observed and predicted values using optimization algorithms.

**"What if you have sparse data?"**
> Use population methods (NONMEM, Stan) to share information across subjects. Model variability explicitly.

**"What are key assumptions?"**
> Linear/first-order kinetics, homogeneous distribution within compartments, constant parameters, measurement error is random.

---

## 📈 Model Complexity Decision Tree

```
START: Do you have concentration-time data?
│
├─→ Fit one-compartment model
│   │
│   └─→ Check fit quality
│       │
│       ├─→ Good (R² > 0.95)? ✓ DONE
│       │
│       └─→ Poor? Check where it fails
│           │
│           ├─→ Early times? Add absorption phase
│           ├─→ Late times? Add peripheral compartment
│           ├─→ Curved residuals? Try nonlinear kinetics
│           └─→ Systematic bias? Check model assumptions
```

---

## 🎯 Interview Preparation Checklist

### Before Interview
- [ ] Run `ode_fitting_tutorial.py` and understand outputs
- [ ] Study diagnostic plots - can you explain each one?
- [ ] Practice 2-min explanation of one-compartment model
- [ ] Practice 5-min explanation of your fitting approach
- [ ] Read sample interview answers
- [ ] Research Amgen's therapeutic focus areas
- [ ] Prepare 3 specific questions for interviewer

### During Interview
- [ ] Listen carefully, don't rush to answer
- [ ] Draw diagrams if explaining compartmental models
- [ ] Ask clarifying questions if confused
- [ ] Admit what you don't know yet
- [ ] Show enthusiasm about learning

---

## 🚫 Common Mistakes to Avoid

| Mistake | Why It's Bad | How to Avoid |
|---------|-------------|-------------|
| Overfitting (too many parameters) | Model fits noise, won't generalize | Start simple, use AIC/BIC |
| Ignoring uncertainty | Parameters seem precise but aren't | Report confidence intervals |
| Assuming perfect model | All models are approximations | Discuss limitations |
| Ignoring residual patterns | May indicate systematic bias | Always check diagnostic plots |
| Not validating on new data | Overfitting undetected | Use hold-out test set |
| Complicated explanation | Non-experts won't understand | Use plain language, diagrams |

---

## 🧮 Units & Conversions (Remember!)

```
Concentration:
- mg/L (most common in PK)
- ng/mL (for trace amounts)
- μM (micromolar, for some assays)

Time:
- Hours (standard in PK)
- Minutes (sometimes)
- Days (for long-term studies)

Volume:
- Liters (L) - volume of distribution
- mL = 0.001 L

Rate constants:
- 1/hour (most common)
- 1/minute
- Always check units when reading papers!

Clearance:
- L/hour = (k_e in 1/h) × (V in L)
- mL/min/kg = normalized to body weight
```

---

## 📚 Reference Values for Common Drugs

```
Most drugs follow first-order kinetics (linear PK)

Typical half-lives:
- Short: 1-5 hours (warfarin, acetaminophen)
- Medium: 5-15 hours (amoxicillin)
- Long: 15-72 hours (digoxin, warfarin)
- Very long: >72 hours (some biologics, antibodies)

Typical volumes of distribution:
- Small (lipophobic): 0.1 L/kg
- Medium: 0.3-0.7 L/kg
- Large (lipophilic): 1-10 L/kg
- Very large: >10 L/kg (binds to tissue)

Typical clearance:
- Hepatic: depends on liver function, enzyme metabolism
- Renal: depends on filtration, reabsorption, secretion
```

---

## 🔗 Linking PK to PD

```
Drug Administration
       ↓
Absorption (input)
       ↓
Distribution → PK Model → C(t) = drug concentration
       ↓
Elimination
       ↓ [Uses C(t) as input]
       ↓
Effect on target → PD Model → E(t) = biomarker/effect
       ↓
Disease response → Disease Model → Outcome (tumor, virus, etc.)
```

Example: Cancer
- PK: predict drug concentration over time
- PD: concentration kills tumor cells (rate depends on C)
- Disease: tumor regrowth vs drug killing = net effect

---

## ✅ Final Checklist Before Interview

Day before:
- [ ] Review key equations (above)
- [ ] Practice explaining diagnostic plots
- [ ] Read through this card one more time

Day of:
- [ ] Get good sleep night before
- [ ] Eat a good meal
- [ ] Arrive 10 minutes early
- [ ] Bring: notepad, pen, water

During:
- [ ] Smile, make eye contact
- [ ] Speak clearly and at moderate pace
- [ ] Pause to think before complex questions
- [ ] Ask clarifying questions
- [ ] Show enthusiasm

---

## 🎓 After Interview

Good sign:
✓ They asked technical questions (means they're interested)
✓ They spent time explaining their work
✓ They asked about your interests/learning goals
✓ Positive, collaborative tone

Next steps:
1. Send thank you email within 24 hours
2. Mention specific topics discussed
3. Reiterate interest in role
4. Wait 3-5 business days for response

---

**Keep this card handy while studying!**

Print it, review it, reference it until you know this material cold.

Good luck! 🚀
