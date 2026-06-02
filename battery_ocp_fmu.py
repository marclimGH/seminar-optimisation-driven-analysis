# %% Battery Thermal OCP — Price-Optimised Charging
# Seminar: "Optimization-Driven Analysis with Physical Models"
#
# Problem: charge a battery to a target SoC while minimising electricity cost.
# Time-varying price signal: the optimizer wants to charge
# during the cheap window, but thermal limits may prevent cramming it all there.
#
# This file works with BOTH FMU versions — toggle by commenting/uncommenting:
#   NAIVE:    BatteryThermal.fmu       (if-else thermostat, no analytic Jacobians)
#   IMPROVED: BatteryThermalSmooth.fmu (tanh thermostat, analytic Jacobians)
#
# Model parameters (R0, c_conv, Q_cool_max, T_cool_on, ...) are exposed via
# annotation(Evaluate=false) in Modelica -> they appear as the 'p' vector in
# CasADi, settable from Python without re-exporting the FMU.

# Setup (run once in a terminal, from the Seminar/ folder):
#   python -m venv .venv
#   .venv\Scripts\activate.bat        # Windows
#   pip install casadi numpy matplotlib
#
# Then in VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> pick .venv
# Run cells with Shift+Enter (the # %% markers are VS Code interactive cells).

import casadi as ca
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% ══════════════════════════════════════════════════════════
#     SELECT FMU
# ══════════════════════════════════════════════════════════════

# ── NAIVE: if-else thermostat + finite differences ───────────
fmu_file = 'BatteryThermal.fmu'

# ── IMPROVED: tanh thermostat + ( analytic Jacobians ? TBD) ───────────
# fmu_file = 'BatteryThermalSmooth.fmu'

# %% ── Load FMU ──────────────────────────────────────────────

dae = ca.DaeBuilder('batt', fmu_file)


### params overrides
# CasADi treats FMI variability="fixed" as compile-time constants (baked in).
# Promote all parameters to "tunable" so they appear in dae.p() and can be
# overridden from Python without re-exporting the FMU.
for name in dae.all_variables():
    if dae.causality(name) == 'parameter' and dae.variability(name) == 'fixed':
        dae.set_variability(name, 'tunable')

dae.disp(True)

x0   = dae.start(dae.x())                             # [SoC=0.2, T_batt=25]
f    = dae.create('f', ['x', 'u', 'p'], ['ode'])      # f(x, u, p) -> ode

nx   = dae.nx()
nu   = dae.nu()
xvar = dae.x()
uvar = dae.u()
pvar = dae.p()
p0   = dae.start(dae.p())   # default parameter values from the FMU

print(f'\nFMU:        {fmu_file}')
print(f'nx={nx}, nu={nu}, np={len(pvar)}, nz={dae.nz()}')
print(f'States:     {xvar}')
print(f'Controls:   {uvar}')
print(f'Parameters: {pvar}')
print(f'x0 = {x0}')
print(f'p0 = {p0}')

# %% ── Tunable parameters ───────────────────────────────────
#    Override any FMU default here — no re-export needed.
#    Build a parameter vector in the same order as dae.p().

param_overrides = {
    # ── Thermal mass ─────────────────────────────────────────
    'C_batt':     6.0,     # [J/K] 

    # ── Electrical ───────────────────────────────────────────
    'R0':         0.20,    # [Ω]

    # ── Convection ───────────────────────────────────────────
    'c_conv':     3.0,     # [W/(m²·K)]

    # ── Cooling ──────────────────────────────────────────────
    'Q_cool_max': 1.5,     # [W] 
 
    'T_cool_on':  35.0,    # [°C]
}

# Assemble p vector: use override if given, else FMU default
p_val = []
for i, name in enumerate(pvar):
    val = param_overrides.get(name, float(p0[i]))
    p_val.append(val)
    if name in param_overrides:
        print(f'  {name} = {val}  (override)')
    else:
        print(f'  {name} = {val}  (FMU default)')
p_val = ca.DM(p_val)

# %% ── Collocation setup ───────────────────
# evaluate problem at collocation points
# no forward integration 
# all points are "solved" at the same time
# ipopt proposes a state trajectory at each iteration
# are dynamics feasible?
# feasibility is linked to a "defect" estimation
# defect = 0 is enforced as a constraint for the problem
# can terminate only if defect is within tolerance ( user-defined)

tf     = 300.0      # Horizon [s]
N      = 50         # Number of intervals
dt     = tf / N
degree = 3

tau       = ca.collocation_points(degree, 'radau')
[C, D, B] = ca.collocation_coeff(tau)

# %% ══════════════════════════════════════════════════════════
#     OCP DEFINITION
#
#     States:      SoC (charge level), T_batt (core temperature)
#     Control:     P_charge (charging power) 
#     Objective:   minimise total electricity cost
#     Constraints: temperature limit, power bounds, target SoC ( boundary)
# ══════════════════════════════════════════════════════════════

opti = ca.Opti()

# ── Constraints ──────────────────────────────────────────────
T_max      = 45.0    # Temperature safety limit [°C] 
P_max      = 15.0    # Maximum charging power [W]
SoC_target = 0.85    # Minimum final SoC - boundary constraint

# ── Price signal ─────────────────────────────────────────────
#    Time-varying electricity price [euro/kWh-equivalent].
#    Three zones over the horizon:
#      0–90 s    medium    (1.0)  — moderate demand
#      90–210 s  cheap     (0.5)  — off-peak / surplus
#      210–300 s expensive (1.5)  — peak demand
t_mid = np.linspace(dt / 2, tf - dt / 2, N)
price = np.where(t_mid < 90,  1.0,
        np.where(t_mid < 210, 0.5, 1.5))

# ── Objective ────────────────────────────────────────────────
#    minimise  Σ  price(k) × P_charge(k) × Δt

# ══════════════════════════════════════════════════════════════
# %% 
# ── Trajectory containers ────────────────────────────────────
# xk: the LEFT boundary state for the current interval.
#   Starts as the known initial condition x0 = [SoC=0.2, T_batt=25].
#   NOT a decision variable in the first interval — it's fixed data.
#   In subsequent intervals, xk IS a decision variable (the previous
#   interval's xk_next), linking the chain together.
xk     = ca.MX(x0)
x_traj = [xk]       # collects boundary nodes only (N+1 total)
u_traj = []          # collects one control per interval (N total)
cost   = 0

# State indices — used to apply component-specific constraints
i_T   = xvar.index('T_batt')
i_SoC = xvar.index('SoC')

# ── Build the NLP interval by interval ───────────────────────
# Each iteration creates decision variables + constraints for one interval [tk, tk+1].  
# After the loop, IPOPT sees one large sparse NLP


for k in range(N):
    # ── Decision variables for this interval ─────────────────
    # Xc: interior collocation states, shape (nx, degree) = (2, 3).
    Xc      = opti.variable(nx, degree)

    # uk: control input (P_charge), held constant over [tk, tk+1].
    uk      = opti.variable(1)

    # xk_next: state at the RIGHT boundary t = tk+1.
    #   Becomes xk for the next interval (continuity chain).
    xk_next = opti.variable(nx)

    # ── Collocation dynamics (defect constraints) ────────────
    # Z: prepend the left boundary xk to the interior collocation states.
    Z     = ca.horzcat(xk, Xc)

    # Pidot: polynomial time-derivative at each collocation point.
    #   Division by dt scales from normalised [0,1] to physical time.
    Pidot = (Z @ C) / dt

    # Defect constraint: polynomial slope must equal the true ODE
    #   f(x=Xc, u=uk, p=p_val)['ode'] calls the FMU derivative evaluation
    #   at all 3 collocation states simultaneously 
    #   Under the hood this is fmi2SetContinuousStates + fmi2GetDerivatives
    #   residual = f(x_mid, u_mid) − ẋ_p(t_mid) = 0,
    #   at convergence, Pidot == f everywhere -> trajectory is dynamically  consistent. 
    opti.subject_to(Pidot == f(x=Xc, u=uk, p=p_val)['ode'])

    # ── Continuity ───────────────────────────────────────────
    #  polynomial at right point = state start at next point
    opti.subject_to(Z @ D == xk_next)

    # ── Path constraints (at collocation points) ─────────────
    # Enforced at each interior collocation point, not just boundaries.
    for j in range(degree):
        opti.subject_to(Xc[i_T, j] <= T_max)          # T_batt ≤ 40°C
        opti.subject_to(0 <= (Xc[i_SoC, j] <= 1))     # 0 ≤ SoC ≤ 1

    # ── Control bounds ───────────────────────────────────────
    # Box constraint: 0 ≤ P_charge ≤ P_max.
    #   opti.bounded() generates both inequalities as a single box constraint 
    opti.subject_to(opti.bounded(0, uk, P_max))

    # ── Cost accumulation ────────────────────────────────────
    cost += price[k] * (uk / 1000) * (dt / 3600)

    # ── Initial guesses ──────────────────────────────────────
    # Warm-start IPOPT: replicate x0 across all collocation points.
    #   This is a flat (constant-state) initial guess.  uk =10
    #   A better guess (e.g. from a forward simulation) would reduce iteration count.
    opti.set_initial(Xc, ca.repmat(x0, 1, degree))
    opti.set_initial(xk_next, x0)
    opti.set_initial(uk, 10)

    # ── Advance to next interval ─────────────────────────────
    # xk_next becomes the left boundary of interval k+1.
    xk = xk_next
    x_traj.append(xk)      # boundary node (N+1 total including x0)
    u_traj.append(uk)      # one control per interval (N total)

# ── Stack into matrices for post-processing ──────────────────
x_traj = ca.hcat(x_traj)   # nx × (N+1) = 2 × 51
u_traj = ca.hcat(u_traj)   # 1  × N     = 1 × 50

# ── Path constraints (at interval boundaries) ────────────────
# These complement the interior-point constraints inside the loop.
#   Together: constraints checked at 3 interior + boundary = 4 points  per interval.  
opti.subject_to(x_traj[i_T, :] <= T_max)
opti.subject_to(0 <= (x_traj[i_SoC, :] <= 1))

# ── Terminal constraint: reach minimum charge level ──────────
opti.subject_to(x_traj[i_SoC, -1] >= SoC_target)

# ── Objective: minimise total electricity cost ───────────────
opti.minimize(cost)

# %% ── Solve ─────────────────────────────────────────────────

opts = {
    'ipopt.print_level': 5,
    'ipopt.max_iter': 1000,
}
opti.solver('ipopt', opts)

try:
    sol    = opti.solve()
    solved = True
    get    = sol.value
    print('\n>> Solved successfully!')
except RuntimeError as e:
    solved = False
    get    = opti.debug.value
    print(f'\nX Solver failed: {e}')

# %% ── Extract results ──────────────────────────────────────

x_opt = np.array(get(x_traj))
u_opt = np.array(get(u_traj))

t_nodes       = np.linspace(0, tf, N + 1)
SoC_vals      = x_opt[i_SoC, :]
T_batt_vals   = x_opt[i_T, :]
P_charge_vals = u_opt.flatten()
total_cost    = float(get(cost)) *1000

print(f'\nFMU:         {fmu_file}')
print(f'Final SoC:   {SoC_vals[-1]:.3f}')
print(f'Max T_batt:  {max(T_batt_vals):.1f} °C')
print(f'Total cost:  {total_cost:.2f} 1e-3')

# %% ── Plot ──────────────────────────────────────────────────

# Read parameter values by name for plot labels
def p_get(name):
    """Return the active value of a named parameter."""
    if name in pvar:
        return float(p_val[pvar.index(name)])
    return None

T_cool_on_val = p_get('T_cool_on')
Q_cool_max_val = p_get('Q_cool_max')
R0_val        = p_get('R0')

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# Price zone shading (behind all subplots)
zones = [(0, 90,  '#fff3cd'),     # medium    — warm yellow
         (90, 210, '#d4edda'),    # cheap     — green
         (210, 300, '#f8d7da')]   # expensive — red
for ax in axes:
    for t0, t1, colour in zones:
        ax.axvspan(t0, t1, alpha=0.25, color=colour, zorder=0)

# 1. Price signal
axes[0].step(t_mid, price, 'k-', where='mid', linewidth=2)
axes[0].set_ylabel('Price [€/kWh]')
axes[0].set_ylim(0, 2)
axes[0].set_title(
    f'Battery OCP — {fmu_file}\n'
    f'R0={R0_val} Ω  T_cool_on={T_cool_on_val} °C  '
    f'Q_cool_max={Q_cool_max_val} W  T_max={T_max} °C  —  Total cost: {total_cost:.2f}'
)

# 2. Temperature
axes[1].plot(t_nodes, T_batt_vals, 'r-', linewidth=1.5, label='T_batt')
axes[1].axhline(T_max, color='r', ls=':', alpha=0.5, label=f'T_max = {T_max} °C')
if T_cool_on_val is not None:
    axes[1].axhline(T_cool_on_val, color='orange', ls=':', alpha=0.5,
                    label=f'T_cool_on = {T_cool_on_val} °C')
axes[1].set_ylabel('Temperature [°C]')
axes[1].legend(loc='upper left', fontsize=8)

# 3. State of charge
axes[2].plot(t_nodes, SoC_vals, 'k-', linewidth=1.5)
axes[2].axhline(SoC_target, color='g', ls=':', alpha=0.5, label=f'SoC target = {SoC_target}')
axes[2].set_ylabel('SoC [-]')
axes[2].legend(loc='lower right', fontsize=8)

# 4. Control
axes[3].step(t_nodes[:-1], P_charge_vals, 'b-', where='post', linewidth=1.5)
axes[3].axhline(P_max, color='r', ls=':', alpha=0.5, label=f'P_max = {P_max} W')
axes[3].set_ylabel('P_charge [W]')
axes[3].set_xlabel('Time [s]')
axes[3].legend(loc='upper right', fontsize=8)

plt.tight_layout()

tag = 'naive' if 'BatteryThermal.fmu' == fmu_file else 'improved'
out = f'fmu_{tag}_result.png'
plt.savefig(out, dpi=150)
plt.show()
print(f'Plot saved to Seminar/{out}')
