"""Microbio Unified Medium — Effective Quantum-Classical 2D Particle Model
---------------------------------------------------------------------------
Scientific & Modeling Scope:
This is an effective quantum-classical (semiclassical) N-body particle model,
NOT an exact solution of the many-body Schrödinger equation. Quantum effects
are incorporated via effective pairwise potentials whose parameters must be
calibrated against target substance properties and temperature.

Core Physics Architecture:
1. No artificial mechanical springs or damping between ions and electrons.
2. Full all-to-all Coulomb interaction across all free charges in the medium
   and surface dipole nodes (no pairwise-restricted binding), computed with
   Periodic Boundary Conditions (PBC) and Minimum Image Convention (MIC).
3. Quantum stability of bound electron-ion states is modeled via an effective
   quantum kinetic confinement energy V_qkin(r) = Q_KIN_SCALE / (r^2 + eps_q^2),
   which dominates over regularized Coulomb attraction at short distances
   (eps_q < COULOMB_SOFTENING) and establishes a global potential minimum at r_eq.
4. Pauli / exchange exclusion between identical fermions (electron-electron
   and identical ionic cores) is modeled via an effective Fermi-Dirac
   statistical repulsion potential V_FD(r) = k_B T_eff * ln(1 + exp(-(r - r_F)/lambda_F))
   with smooth analytical forces.
5. Artificial viscous friction, stochastic boundary thermal baths, soft-wall
   springs, and pressure-induced shape deformation are completely removed.
   Domain boundaries are strictly periodic (continuous medium, zero wall bounce).
6. Medium-bacterium momentum and angular momentum transfer occurs strictly through
   conservative inter-particle forces (Newton's third law, exact action-reaction)
   plus elastic hard-core exclusion at the geometric boundary, preserving total
   linear and angular momentum strictly and avoiding artificial energy injection
   (aside from the prescribed kinematic wave work of the hull).

Run:
    pip install numpy matplotlib gradio imageio[ffmpeg]
    python ionic_bionic_quantum_effective_pbc.py
"""

import tempfile
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import gradio as gr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath


# ----------------------------- numerical & physical parameters -----------------------------
SEED = 7
BOX_X, BOX_Y = 6.0, 3.0
N_PAIRS_DEFAULT = 90
DT = 0.05
SUBSTEPS_PER_FRAME = 12

# Fundamental / calibrated effective scales (dimensionless simulation units)
PARTICLE_MASS = 1.0
ELECTRON_MASS = 0.08

COULOMB_K = 0.0036
COULOMB_SOFTENING = 0.25    # Было 0.035 — теперь это эффективный радиус облака макрочастицы

# Масштабируем квантовое отталкивание под новый радиус ядра (Q_KIN_EPS < COULOMB_SOFTENING)
Q_KIN_EPS = 0.10            # Было 0.015 (сохраняем соотношение ~0.4 от COULOMB_SOFTENING)
Q_KIN_SCALE = 0.00003       # Было 0.000215 — снижаем, чтобы убрать жесткую потенциальную яму

DAMPING = 0.015  # Default calibrated to the water-like CFD visualization setting

# Effective Fermi-Dirac Statistical Repulsion Potential for Identical Fermions
# V_FD(r) = FD_KT * ln(1 + exp(-(r - R_F) / LAMBDA_F))
# Force: F_FD(r) = (FD_KT / LAMBDA_F) / (1 + exp((r - R_F) / LAMBDA_F)) * r_hat
FD_KT_ELEC = 0.018          # Effective Fermi energy / thermal scale for electrons
FD_RF_ELEC = 0.095          # Effective electron exchange radius
FD_LAMBDA_ELEC = 0.025      # Thermal de Broglie smoothing width

FD_KT_CORE = 0.045          # Effective core exclusion scale for ions
FD_RF_CORE = 0.190          # Ionic core radius
FD_LAMBDA_CORE = 0.030      # Core smoothing width
FD_CUTOFF_MULT = 8.0        # Smooth cutoff where exp(-(r-R_F)/lambda) < 3e-4

# Bacterium / hull geometry (initially centered in the domain)
HULL_LENGTH = 2.40
HULL_HALF_LEN = 0.5 * HULL_LENGTH
HULL_R_MAX = 0.22
HULL_CAP_START = 0.80
HULL_INIT_X = BOX_X / 2.0
HULL_INIT_Y = BOX_Y / 2.0

# Base number of surface nodes at surface_density = 1.0
HULL_NODES_BASE = 64
HULL_MASS_MULTIPLIER = 1.0  # во сколько раз корпус тяжелее суммы своих поверхностных узлов
INTERNAL_PARTICLES_BASE = 40  # базовое число внутренних нейтральных частиц при mass = 1.0

SURFACE_DIPOLE_OFFSET = 0.035
WALL_RESTITUTION = 1.0      # Elastic core exclusion at geometric hull boundary

# Curvature-coupled running charge parameters (preserved phenomenology)
FISH_SPATIAL_K = 1.8 * np.pi
CURVATURE_GAIN = 6.0


# ----------------------------- periodic boundary helpers -----------------------------
def minimum_image(d: np.ndarray) -> np.ndarray:
    """
    Minimum Image Convention (MIC) displacement vector in a 2D periodic box
    [0, BOX_X) x [0, BOX_Y).
    """
    d_mic = d.copy()
    d_mic[..., 0] -= BOX_X * np.round(d_mic[..., 0] / BOX_X)
    d_mic[..., 1] -= BOX_Y * np.round(d_mic[..., 1] / BOX_Y)
    return d_mic


def rotate_2d(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rotates 2D points around the origin by `angle` radians."""
    if len(pts) == 0:
        return pts.copy()
    c, s = np.cos(angle), np.sin(angle)
    x = pts[:, 0] * c - pts[:, 1] * s
    y = pts[:, 0] * s + pts[:, 1] * c
    return np.column_stack((x, y))


# ----------------------------- effective quantum potentials -----------------------------
def effective_ei_potential(r: np.ndarray) -> np.ndarray:
    """
    Effective electron-ion interaction potential:
        V_ei(r) = -COULOMB_K / sqrt(r^2 + COULOMB_SOFTENING^2)
                  + Q_KIN_SCALE / (r^2 + Q_KIN_EPS^2)
    """
    v_coul = -COULOMB_K / np.sqrt(r ** 2 + COULOMB_SOFTENING ** 2)
    v_qkin = Q_KIN_SCALE / (r ** 2 + Q_KIN_EPS ** 2)
    return v_coul + v_qkin


def effective_ei_equilibrium_radius() -> float:
    """
    Finds the global equilibrium separation r_eq > 0 for an isolated
    ion-electron pair under V_ei(r) = V_coul(r) + V_qkin(r).
    """
    r_grid = np.linspace(0.0, 1.0, 40001)
    v_tot = effective_ei_potential(r_grid)
    idx_min = int(np.argmin(v_tot))
    return float(r_grid[idx_min])


R_EQ_CALIBRATED = effective_ei_equilibrium_radius()


def pairwise_medium_forces(pos: np.ndarray, charge: np.ndarray, species: np.ndarray):
    """
    Computes conservative all-to-all forces within the periodic medium:
    1. All-to-all Coulomb interaction across every charge pair (i != j) using MIC.
    2. Effective quantum kinetic confinement force between opposite charges (e-i)
       preventing Coulomb collapse and maintaining stable bound states:
       F_qkin = -dV_qkin/dr * r_hat = 2 * Q_KIN_SCALE / (r^2 + Q_KIN_EPS^2)^2 * d
    3. Effective Fermi-Dirac statistical repulsion between identical fermions
       (electron-electron and ion-ion).
    """
    d = minimum_image(pos[:, None, :] - pos[None, :, :])
    r2_raw = np.sum(d * d, axis=2)
    r_raw = np.sqrt(r2_raw + 1e-18)

    # 1. All-to-all Coulomb force
    r2_soft = r2_raw + COULOMB_SOFTENING ** 2
    np.fill_diagonal(r2_soft, np.inf)
    invr3 = r2_soft ** (-1.5)
    qq = charge[:, None] * charge[None, :]
    f_coul = COULOMB_K * (qq * invr3)[:, :, None] * d

    # 2. Effective quantum kinetic confinement force between opposite-sign charges (e-i)
    opp_mask = (species[:, None] != species[None, :])
    np.fill_diagonal(opp_mask, False)
    denom_q = (r2_raw + Q_KIN_EPS ** 2) ** 2
    qkin_coeff = (2.0 * Q_KIN_SCALE / denom_q) * opp_mask
    f_qkin = qkin_coeff[:, :, None] * d

    # 3. Effective Fermi-Dirac repulsion between identical fermions
    same_core = (species[:, None] == 0) & (species[None, :] == 0)
    same_elec = (species[:, None] == 1) & (species[None, :] == 1)
    same_mask = same_core | same_elec
    np.fill_diagonal(same_mask, False)

    kt = np.where(same_core, FD_KT_CORE, FD_KT_ELEC)
    r_f = np.where(same_core, FD_RF_CORE, FD_RF_ELEC)
    lam = np.where(same_core, FD_LAMBDA_CORE, FD_LAMBDA_ELEC)

    arg = np.clip((r_raw - r_f) / lam, -40.0, 40.0)
    fermi_occ = 1.0 / (1.0 + np.exp(arg))
    mag_fd = (kt / lam) * fermi_occ * same_mask
    mag_fd[r_raw > (r_f + FD_CUTOFF_MULT * lam)] = 0.0
    f_fd = (mag_fd / r_raw)[:, :, None] * d

    return np.sum(f_coul + f_qkin + f_fd, axis=1)


def hull_surface_forces(
    pos: np.ndarray,
    species: np.ndarray,
    charge: np.ndarray,
    hull_ions: np.ndarray,
    hull_elecs: np.ndarray,
    hull_center: np.ndarray,
    surface_charge_weight: float,
    running_charge_weights=None,
):
    """
    Conservative periodic interaction between medium charges and bacterium surface charges:
    - All-to-all Coulomb interaction (MIC)
    - Effective quantum kinetic confinement repulsion between opposite charges
      (medium e <-> hull ion, medium ion <-> hull electron)
    - Effective Fermi-Dirac statistical repulsion between identical fermions
      (medium ion <-> hull ion, medium electron <-> hull electron)
    Exact action-reaction: reaction_on_hull = -sum(total_force_on_medium),
    and torque on hull about hull_center is computed strictly from the reaction
    forces applied at each surface charge node (hull_ions and hull_elecs).
    """
    n_nodes = len(hull_ions)
    if n_nodes == 0:
        return np.zeros_like(pos), np.zeros(2), 0.0

    hull_pos_charges = np.vstack([hull_ions, hull_elecs])

    if running_charge_weights is not None:
        redist = np.asarray(running_charge_weights, dtype=float)
        redist = redist - np.mean(redist)
        max_abs = float(np.max(np.abs(redist)))
        if max_abs > 1e-9:
            redist = redist / max_abs
        ion_w = np.maximum(0.05, surface_charge_weight * (1.0 + 0.75 * redist))
        elec_w = np.maximum(0.05, surface_charge_weight * (1.0 - 0.75 * redist))
        ion_w = ion_w * (surface_charge_weight / max(float(np.mean(ion_w)), 1e-9))
        elec_w = elec_w * (surface_charge_weight / max(float(np.mean(elec_w)), 1e-9))
        hull_q = np.concatenate([ion_w, -elec_w])
    else:
        ion_w = np.full(n_nodes, surface_charge_weight)
        elec_w = np.full(n_nodes, surface_charge_weight)
        hull_q = np.concatenate([ion_w, -elec_w])

    # Pairwise force components from each hull charge node j to each medium particle i
    # 1. Coulomb interaction (MIC)
    dr = minimum_image(pos[:, None, :] - hull_pos_charges[None, :, :])
    r2_raw = np.sum(dr * dr, axis=2)
    r2_soft = r2_raw + COULOMB_SOFTENING ** 2
    qq = charge[:, None] * hull_q[None, :]
    pair_f_coul = COULOMB_K * ((qq * (r2_soft ** (-1.5)))[:, :, None] * dr)

    pair_f_qkin = np.zeros_like(dr)
    pair_f_fd = np.zeros_like(dr)

    core_mask = (species == 0)
    elec_mask = (species == 1)

    # 2. Medium ions (species 0) vs hull ions (Fermi-Dirac) and hull electrons (Quantum Kinetic)
    if np.any(core_mask):
        p_core = pos[core_mask]

        # Ion-Ion Fermi-Dirac repulsion (hull nodes 0 .. n_nodes-1)
        dr_cc = minimum_image(p_core[:, None, :] - hull_ions[None, :, :])
        r_cc = np.linalg.norm(dr_cc, axis=2) + 1e-12
        arg_cc = np.clip((r_cc - FD_RF_CORE) / FD_LAMBDA_CORE, -40.0, 40.0)
        mag_cc = (FD_KT_CORE / FD_LAMBDA_CORE) * ion_w[None, :] / (1.0 + np.exp(arg_cc))
        mag_cc[r_cc > (FD_RF_CORE + FD_CUTOFF_MULT * FD_LAMBDA_CORE)] = 0.0
        pair_f_fd[core_mask, :n_nodes, :] += (mag_cc / r_cc)[:, :, None] * dr_cc

        # Ion-Electron Quantum Kinetic confinement repulsion (hull nodes n_nodes .. 2*n_nodes-1)
        dr_ce = minimum_image(p_core[:, None, :] - hull_elecs[None, :, :])
        r2_ce = np.sum(dr_ce * dr_ce, axis=2)
        coeff_ce = (2.0 * Q_KIN_SCALE * elec_w[None, :]) / ((r2_ce + Q_KIN_EPS ** 2) ** 2)
        pair_f_qkin[core_mask, n_nodes:, :] += coeff_ce[:, :, None] * dr_ce

    # 3. Medium electrons (species 1) vs hull electrons (Fermi-Dirac) and hull ions (Quantum Kinetic)
    if np.any(elec_mask):
        p_elec = pos[elec_mask]

        # Electron-Electron Fermi-Dirac repulsion (hull nodes n_nodes .. 2*n_nodes-1)
        dr_ee = minimum_image(p_elec[:, None, :] - hull_elecs[None, :, :])
        r_ee = np.linalg.norm(dr_ee, axis=2) + 1e-12
        arg_ee = np.clip((r_ee - FD_RF_ELEC) / FD_LAMBDA_ELEC, -40.0, 40.0)
        mag_ee = (FD_KT_ELEC / FD_LAMBDA_ELEC) * elec_w[None, :] / (1.0 + np.exp(arg_ee))
        mag_ee[r_ee > (FD_RF_ELEC + FD_CUTOFF_MULT * FD_LAMBDA_ELEC)] = 0.0
        pair_f_fd[elec_mask, n_nodes:, :] += (mag_ee / r_ee)[:, :, None] * dr_ee

        # Electron-Ion Quantum Kinetic confinement repulsion (hull nodes 0 .. n_nodes-1)
        dr_ec = minimum_image(p_elec[:, None, :] - hull_ions[None, :, :])
        r2_ec = np.sum(dr_ec * dr_ec, axis=2)
        coeff_ec = (2.0 * Q_KIN_SCALE * ion_w[None, :]) / ((r2_ec + Q_KIN_EPS ** 2) ** 2)
        pair_f_qkin[elec_mask, :n_nodes, :] += coeff_ec[:, :, None] * dr_ec

    pair_total_on_med = pair_f_coul + pair_f_qkin + pair_f_fd
    total_force_on_medium = np.sum(pair_total_on_med, axis=1)

    # Exact action-reaction on each hull charge node j
    force_on_hull_nodes = -np.sum(pair_total_on_med, axis=0)
    reaction_on_hull = np.sum(force_on_hull_nodes, axis=0)

    # Lever arm from hull_center to each hull charge node
    r_arm = hull_pos_charges - hull_center[None, :]
    torque_on_hull = float(
        np.sum(r_arm[:, 0] * force_on_hull_nodes[:, 1] - r_arm[:, 1] * force_on_hull_nodes[:, 0])
    )
    return total_force_on_medium, reaction_on_hull, torque_on_hull


# ----------------------------- bacterium geometry & kinematics (preserved) -----------------------------
def _bacterium_radius_profile(
    x_norm: np.ndarray,
    r_max: float = HULL_R_MAX,
    cap_start: float = HULL_CAP_START,
) -> np.ndarray:
    abs_x = np.abs(x_norm)
    cap_t = np.clip((abs_x - cap_start) / max(1.0 - cap_start, 1e-6), 0.0, 1.0)
    return np.where(
        abs_x < cap_start,
        r_max,
        r_max * np.sqrt(np.maximum(0.0, 1.0 - cap_t ** 2)),
    )


def _reference_perimeter_length(n_samples: int = 400) -> float:
    u = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    theta = 2.0 * np.pi * u
    s = 0.5 - 0.5 * np.cos(theta)
    x_norm = 2.0 * s - 1.0
    r_prof = _bacterium_radius_profile(x_norm)
    sign_y = np.where(np.sin(theta) >= 0.0, 1.0, -1.0)
    x = HULL_INIT_X + x_norm * HULL_HALF_LEN
    y = HULL_INIT_Y + sign_y * r_prof
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    return float(np.sum(np.hypot(dx, dy)))


REF_PERIMETER = _reference_perimeter_length()


def _mode_blend(mode) -> float:
    if isinstance(mode, (int, float, np.floating)):
        return float(np.clip(mode, 0.0, 1.0))
    if isinstance(mode, str) and "jellyfish" in mode.lower():
        return 0.0
    return 1.0


def _evaluate_bacterium_deformation(
    x_norm: np.ndarray,
    phase: float,
    amplitude: float,
    mode_fish: float,
):
    """Evaluates local transverse centerline offset y_local_center and curvature for x_norm in [-1, 1]."""
    asym_strength = (1.0 - mode_fish) * 0.45
    asym_phase = phase + asym_strength * np.sin(phase)

    s = 0.5 * (x_norm + 1.0)
    ds_dx = 1.0 / max(HULL_LENGTH, 1e-9)

    center_bow = np.maximum(0.0, 1.0 - x_norm ** 2)
    bow_amp = amplitude * 0.22
    bow_offset = bow_amp * np.sin(asym_phase) * center_bow * (1.0 - mode_fish)

    envelope_fish = np.maximum(0.0, s) ** 1.35
    wave_amp = amplitude * 0.16
    wave_arg = asym_phase - FISH_SPATIAL_K * s
    wave_offset = wave_amp * envelope_fish * np.sin(wave_arg) * mode_fish

    y_local_center = bow_offset + wave_offset

    denv_dx = 1.35 * (np.maximum(s, 1e-9) ** 0.35) * ds_dx
    d2y_dx2_bow = bow_amp * np.sin(asym_phase) * (-4.0 * ds_dx ** 2) * (1.0 - mode_fish)
    d2y_dx2_wave = (
        wave_amp
        * (
            -2.0 * denv_dx * FISH_SPATIAL_K * ds_dx * np.cos(wave_arg)
            - envelope_fish * (FISH_SPATIAL_K * ds_dx) ** 2 * np.sin(wave_arg)
        )
        * mode_fish
    )
    d2y_dx2 = d2y_dx2_bow + d2y_dx2_wave
    return y_local_center, d2y_dx2, asym_phase, wave_arg


def _evaluate_bacterium_contour(
    u: np.ndarray,
    phase: float,
    amplitude: float,
    mode_fish: float,
):
    """Returns unrotated local contour coordinates (centered near origin) and running charge weights."""
    theta = 2.0 * np.pi * u
    s = 0.5 - 0.5 * np.cos(theta)
    x_norm = 2.0 * s - 1.0

    y_local_center, d2y_dx2, asym_phase, wave_arg = _evaluate_bacterium_deformation(
        x_norm, phase, amplitude, mode_fish
    )

    r_prof = _bacterium_radius_profile(x_norm, r_max=HULL_R_MAX, cap_start=HULL_CAP_START)
    side_sign = np.where(np.sin(theta) >= 0.0, 1.0, -1.0)
    x_raw = x_norm * HULL_HALF_LEN
    y_raw = y_local_center + side_sign * r_prof

    curvature = -d2y_dx2

    # 1. Логика для рыбы (асимметричная волна от головы к хвосту)
    asymmetry_fish = 1.0 + 0.35 * x_norm
    phase_travel_fish = np.cos(wave_arg + 0.25 * asym_phase)
    q_fish = np.tanh(side_sign * curvature * CURVATURE_GAIN + 0.45 * phase_travel_fish) * asymmetry_fish

    # 2. Логика для медузы (симметрично от центральной вертикальной оси x_norm = 0 к краям и обратно)
    abs_x = np.abs(x_norm)
    jelly_k = 1.2 * np.pi
    radial_flow = np.cos(asym_phase - jelly_k * abs_x)
    q_jelly = np.tanh(side_sign * curvature * CURVATURE_GAIN + 0.65 * radial_flow)

    q_running = mode_fish * q_fish + (1.0 - mode_fish) * q_jelly
    return x_raw, y_raw, q_running


def generate_internal_particles_local(n_internal: int) -> np.ndarray:
    """
    Generates uniform deterministic grid/quasi-uniform coordinates inside the undeformed
    bacterium body in normalized coordinates (x_norm, y_rel_norm), where
    x_norm in (-1, 1) and y_rel_norm in (-1, 1) scales with local radius profile.
    """
    n_internal = max(0, int(n_internal))
    if n_internal == 0:
        return np.zeros((0, 2), dtype=float)

    # Rejection-free uniform area sampling using deterministic Halton / stratified grid
    # Aspect ratio of capsule ~ HULL_LENGTH / (2 * HULL_R_MAX) ~ 5.45
    aspect = HULL_LENGTH / max(2.0 * HULL_R_MAX, 1e-6)
    nx = max(4, int(np.ceil(np.sqrt(n_internal * aspect * 1.35))))
    ny = max(2, int(np.ceil(np.sqrt(n_internal / max(aspect, 0.2) * 1.35))))

    xs = np.linspace(-0.92, 0.92, nx)
    ys = np.linspace(-0.82, 0.82, ny)
    gx, gy = np.meshgrid(xs, ys)
    cand_x = gx.ravel()
    cand_y = gy.ravel() * HULL_R_MAX

    r_lim = _bacterium_radius_profile(cand_x, r_max=HULL_R_MAX * 0.84, cap_start=HULL_CAP_START)
    valid = np.abs(cand_y) <= r_lim
    vx = cand_x[valid]
    vy = cand_y[valid]

    # Store normalized transverse fraction relative to local max radius so particles deform smoothly with hull
    r_local_full = np.maximum(_bacterium_radius_profile(vx), 1e-6)
    y_frac = vy / r_local_full

    if len(vx) >= n_internal:
        idx = np.linspace(0, len(vx) - 1, n_internal, dtype=int)
        return np.column_stack((vx[idx], y_frac[idx]))

    # If more points requested than initial grid, increase resolution
    scale = int(np.ceil(np.sqrt(n_internal / max(len(vx), 1)))) + 1
    xs = np.linspace(-0.94, 0.94, nx * scale)
    ys = np.linspace(-0.86, 0.86, ny * scale)
    gx, gy = np.meshgrid(xs, ys)
    cand_x = gx.ravel()
    cand_y = gy.ravel() * HULL_R_MAX
    r_lim = _bacterium_radius_profile(cand_x, r_max=HULL_R_MAX * 0.86, cap_start=HULL_CAP_START)
    valid = np.abs(cand_y) <= r_lim
    vx = cand_x[valid]
    vy = cand_y[valid]
    r_local_full = np.maximum(_bacterium_radius_profile(vx), 1e-6)
    y_frac = vy / r_local_full
    idx = np.linspace(0, len(vx) - 1, min(n_internal, len(vx)), dtype=int)
    return np.column_stack((vx[idx], y_frac[idx]))


INNER_WALL_CLEARANCE = SURFACE_DIPOLE_OFFSET * 0.5 + 0.045  # Физический зазор от границы оболочки и внутренних диполей


def _constrain_points_inside_hull(
    pts_local: np.ndarray,
    hull_local: np.ndarray,
    clearance: float = INNER_WALL_CLEARANCE,
) -> np.ndarray:
    """
    Guarantees every internal particle stays strictly inside the closed hull polygon
    and maintains at least `clearance` distance from every hull boundary segment
    (works for arbitrary non-convex / S-curved shapes and prevents overlap with shell particles).
    """
    if len(pts_local) == 0 or len(hull_local) < 3:
        return pts_local

    constrained = pts_local.copy()
    seg_a = hull_local
    seg_b = np.roll(hull_local, -1, axis=0)
    ab = seg_b - seg_a
    ab_len2 = np.sum(ab * ab, axis=1) + 1e-18

    # Exact local inward unit normal for each boundary segment via point-in-polygon probe
    seg_mid = 0.5 * (seg_a + seg_b)
    n_cand = np.column_stack((-ab[:, 1], ab[:, 0]))
    n_norm = np.linalg.norm(n_cand, axis=1, keepdims=True) + 1e-12
    n_unit = n_cand / n_norm
    probe_in = _point_in_polygon_mask(seg_mid + 1e-4 * n_unit, hull_local)
    inward_sign = np.where(probe_in[:, None], 1.0, -1.0)
    inward_n = n_unit * inward_sign

    for _ in range(3):
        inside = _point_in_polygon_mask(constrained, hull_local)
        ap = constrained[:, None, :] - seg_a[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab_len2[None, :], 0.0, 1.0)
        closest = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
        diff = constrained[:, None, :] - closest
        dist = np.linalg.norm(diff, axis=2)

        min_seg_idx = np.argmin(dist, axis=1)
        min_dist = dist[np.arange(len(constrained)), min_seg_idx]
        nearest_pt = closest[np.arange(len(constrained)), min_seg_idx]
        seg_in_n = inward_n[min_seg_idx]

        violated = (~inside) | (min_dist < clearance)
        if not np.any(violated):
            break

        constrained[violated] = nearest_pt[violated] + seg_in_n[violated] * clearance

    return constrained


def evaluate_internal_particles_world(
    internal_norm: np.ndarray,
    mode,
    phase: float,
    amplitude: float,
    center_xy: np.ndarray,
    hull_angle: float,
    length_scale: float = 1.0,
) -> np.ndarray:
    """
    Maps normalized internal neutral particles into world coordinates following hull wave & rotation,
    matching the exact contour centroid shift and enforcing strict containment inside the shell boundary.
    """
    if len(internal_norm) == 0:
        return np.zeros((0, 2), dtype=float)
    mode_fish = _mode_blend(mode)

    # Match exact centroid subtraction used in make_hull_nodes so internal body never drifts relative to shell
    u_fine = np.linspace(0.0, 1.0, 400, endpoint=False)
    x_f, y_f, _ = _evaluate_bacterium_contour(u_fine, phase, float(amplitude), mode_fish)
    cx_f = float(np.mean(x_f))
    cy_f = float(np.mean(y_f))
    hull_local = np.column_stack(((x_f - cx_f) * length_scale, (y_f - cy_f) * length_scale))

    x_norm = internal_norm[:, 0]
    y_frac = internal_norm[:, 1]
    y_center_off, _, _, _ = _evaluate_bacterium_deformation(
        x_norm, phase, float(amplitude), mode_fish
    )

    # Compute local centerline tangent/normal so cross-section stays perpendicular even at large bend
    eps_x = 1e-3
    x_plus = np.clip(x_norm + eps_x, -1.0, 1.0)
    x_minus = np.clip(x_norm - eps_x, -1.0, 1.0)
    y_plus, _, _, _ = _evaluate_bacterium_deformation(x_plus, phase, float(amplitude), mode_fish)
    y_minus, _, _, _ = _evaluate_bacterium_deformation(x_minus, phase, float(amplitude), mode_fish)
    dx_c = (x_plus - x_minus) * HULL_HALF_LEN
    dy_c = y_plus - y_minus
    seg_l = np.hypot(dx_c, dy_c) + 1e-12
    nx_c = -dy_c / seg_l
    ny_c = dx_c / seg_l

    r_prof = _bacterium_radius_profile(x_norm, r_max=HULL_R_MAX * 0.78, cap_start=HULL_CAP_START)
    radial_offset = y_frac * r_prof
    x_raw = x_norm * HULL_HALF_LEN + nx_c * radial_offset
    y_raw = y_center_off + ny_c * radial_offset

    x_local = (x_raw - cx_f) * length_scale
    y_local = (y_raw - cy_f) * length_scale
    local_pts = np.column_stack((x_local, y_local))

    # Hard physical boundary check: keep strictly inside hull_local with clearance from shell dipoles
    local_pts = _constrain_points_inside_hull(local_pts, hull_local, clearance=INNER_WALL_CLEARANCE)

    rot_pts = rotate_2d(local_pts, hull_angle)
    return rot_pts + center_xy[None, :]


def make_hull_nodes(
    mode,
    phase,
    amplitude,
    n_nodes: int,
    running_charges: bool = False,
    center_xy=None,
    hull_angle: float = 0.0,
):
    """
    Build closed bacterium contour strictly following the prescribed
    Fish / Jellyfish traveling-wave phenomenology and rigid-body angle
    without artificial pressure-induced shape deformation or soft-wall springs.
    """
    if center_xy is None:
        center_xy = np.array([HULL_INIT_X, HULL_INIT_Y], dtype=float)
    else:
        center_xy = np.asarray(center_xy, dtype=float)

    n_nodes = max(0, int(n_nodes))
    if n_nodes == 0:
        empty = np.zeros((0, 2))
        return empty, empty, empty, empty, None, 1.0

    mode_fish = _mode_blend(mode)
    u_fine = np.linspace(0.0, 1.0, 400, endpoint=False)
    x_f, y_f, q_f = _evaluate_bacterium_contour(
        u_fine, phase, float(amplitude), mode_fish
    )

    seg_dx = np.roll(x_f, -1) - x_f
    seg_dy = np.roll(y_f, -1) - y_f
    curr_len = float(np.sum(np.hypot(seg_dx, seg_dy)))
    length_scale = REF_PERIMETER / max(curr_len, 1e-9)

    cx_f = float(np.mean(x_f))
    cy_f = float(np.mean(y_f))
    x_s = (x_f - cx_f) * length_scale
    y_s = (y_f - cy_f) * length_scale

    d_s = np.hypot(np.roll(x_s, -1) - x_s, np.roll(y_s, -1) - y_s)
    cum_s = np.concatenate([[0.0], np.cumsum(d_s)])
    total_s = cum_s[-1]
    target_s = np.linspace(0.0, total_s, n_nodes, endpoint=False)

    x_wrap = np.append(x_s, x_s[0])
    y_wrap = np.append(y_s, y_s[0])
    q_wrap = np.append(q_f, q_f[0])
    x = np.interp(target_s, cum_s, x_wrap)
    y = np.interp(target_s, cum_s, y_wrap)
    q_nodes = np.interp(target_s, cum_s, q_wrap)

    tx = np.roll(x, -1) - np.roll(x, 1)
    ty = np.roll(y, -1) - np.roll(y, 1)
    t_norm = np.hypot(tx, ty) + 1e-12
    tx /= t_norm
    ty /= t_norm

    n1_x, n1_y = ty, -tx
    n2_x, n2_y = -ty, tx
    c_x = float(np.mean(x))
    c_y = float(np.mean(y))
    dot1 = float(np.sum((x - c_x) * n1_x + (y - c_y) * n1_y))
    if dot1 >= 0.0:
        normals_local = np.column_stack((n1_x, n1_y))
    else:
        normals_local = np.column_stack((n2_x, n2_y))

    local_hull = np.column_stack((x, y))
    rot_hull = rotate_2d(local_hull, hull_angle)
    normals = rotate_2d(normals_local, hull_angle)

    hull_pos = center_xy[None, :] + rot_hull
    hull_ions = hull_pos - normals * (SURFACE_DIPOLE_OFFSET * 0.5)
    hull_elecs = hull_pos + normals * (SURFACE_DIPOLE_OFFSET * 0.5)

    running_weights = q_nodes if running_charges else None
    return hull_pos, hull_ions, hull_elecs, normals, running_weights, length_scale


def _compute_hull_moment_of_inertia(
    hull_pos: np.ndarray,
    hull_center: np.ndarray,
    shell_mass: float,
    internal_world: np.ndarray,
    internal_mass: float,
) -> float:
    """Computes scalar 2D moment of inertia I_z of the bacterium about hull_center."""
    i_shell = 0.0
    if len(hull_pos) > 0 and shell_mass > 0.0:
        r_s = hull_pos - hull_center[None, :]
        r2_s = np.sum(r_s * r_s, axis=1)
        i_shell = (shell_mass / len(hull_pos)) * float(np.sum(r2_s))

    i_int = 0.0
    if len(internal_world) > 0 and internal_mass > 0.0:
        r_i = internal_world - hull_center[None, :]
        r2_i = np.sum(r_i * r_i, axis=1)
        i_int = (internal_mass / len(internal_world)) * float(np.sum(r2_i))

    # Fallback minimum inertia so rotation remains well-conditioned
    return max(i_shell + i_int, 1e-4)


def _point_in_polygon_mask(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    if len(poly) < 3:
        return np.zeros(len(points), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    px = poly[:, 0]
    py = poly[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cond_y = (py[i] > y) != (py[j] > y)
        slope = (px[j] - px[i]) / ((py[j] - py[i]) + 1e-15)
        x_int = px[i] + (y - py[i]) * slope
        inside ^= cond_y & (x < x_int)
        j = i
    return inside


# ----------------------------- visualization-only CFD (D2Q9) -----------------------------
# This grid is strictly a renderer: it is never sampled by the particle integrator,
# never contributes forces, and never changes bacterium state.
CFD_NX, CFD_NY = 180, 90
CFD_C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=int)
CFD_W = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=float)
CFD_CMAP = LinearSegmentedColormap.from_list(
    "cfd_multicolor",
    [(0.00, "#004d99"), (0.10, "#00b7ff"), (0.25, "#00ffd0"),
     (0.40, "#00ff66"), (0.60, "#dfff00"), (0.85, "#ff9a00"),
     (1.00, "#ff2200")], N=512)
CFD_CMAP.set_bad(color="#10151d")


def _cfd_equilibrium(rho, ux, uy):
    usq = ux*ux + uy*uy
    out = np.empty((9, CFD_NY, CFD_NX), dtype=float)
    for k, (cx, cy) in enumerate(CFD_C):
        cu = 3.0 * (cx*ux + cy*uy)
        out[k] = CFD_W[k] * rho * (1.0 + cu + 0.5*cu*cu - 1.5*usq)
    return out


def _cfd_init():
    rho = np.ones((CFD_NY, CFD_NX), dtype=float)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    return _cfd_equilibrium(rho, ux, uy)


def _cfd_wall_velocity_field(solid, hull_prev, hull_curr, center_prev, center_curr, dt):
    """Map the bacterium's measured contour velocity onto its rasterized body.

    This is a visualization-only kinematic boundary condition. The LBM field
    never feeds forces back into the particle/bacterium integrator.
    """
    wall_ux = np.zeros((CFD_NY, CFD_NX), dtype=float)
    wall_uy = np.zeros((CFD_NY, CFD_NX), dtype=float)
    iy, ix = np.where(solid)
    if len(ix) == 0 or len(hull_curr) == 0:
        return wall_ux, wall_uy

    center_prev = np.asarray(center_prev, dtype=float)
    center_curr = np.asarray(center_curr, dtype=float)
    rel_prev = minimum_image(np.asarray(hull_prev, dtype=float) - center_prev[None, :])
    rel_curr = minimum_image(np.asarray(hull_curr, dtype=float) - center_curr[None, :])
    center_velocity = minimum_image(center_curr - center_prev) / max(float(dt), 1e-12)
    vertex_velocity = center_velocity[None, :] + (rel_curr - rel_prev) / max(float(dt), 1e-12)

    gx = (ix + 0.5) * (BOX_X / CFD_NX)
    gy = (iy + 0.5) * (BOX_Y / CFD_NY)
    grid_pts = np.column_stack((gx, gy))
    grid_rel = minimum_image(grid_pts - center_curr[None, :])

    # Each solid lattice cell follows the nearest contour point's local velocity.
    # Chunking bounds temporary memory for higher surface-node counts.
    nearest = np.empty(len(grid_pts), dtype=int)
    chunk = 2048
    for a in range(0, len(grid_pts), chunk):
        b = min(a + chunk, len(grid_pts))
        delta = grid_rel[a:b, None, :] - rel_curr[None, :, :]
        nearest[a:b] = np.argmin(np.sum(delta * delta, axis=2), axis=1)

    # Convert world-space simulation velocity to a stable lattice-scale wall speed.
    # This factor is purely a visualization scale; it does not alter model physics.
    lattice_velocity = vertex_velocity[nearest] * DT * (CFD_NX / BOX_X) * 0.08
    lattice_velocity = np.clip(lattice_velocity, -0.22, 0.22)
    wall_ux[iy, ix] = lattice_velocity[:, 0]
    wall_uy[iy, ix] = lattice_velocity[:, 1]
    return wall_ux, wall_uy

def _cfd_step(f, solid, tau, wall_ux, wall_uy, p_damping=0.08):
    rho = np.maximum(np.sum(f, axis=0), 1e-8)
    ux = np.sum(f * CFD_C[:, 0, None, None], axis=0) / rho
    uy = np.sum(f * CFD_C[:, 1, None, None], axis=0) / rho

    # Быстрое затухание волн давления (отклонений плотности от 1.0) в жидкости
    if p_damping > 0.0:
        rho_damped = np.where(solid, rho, 1.0 + (rho - 1.0) * (1.0 - float(p_damping)))
        scale = np.clip(rho_damped / rho, 0.0, 2.0)
        f = f * scale[None, :, :]
        rho = rho_damped

    # BGK collision in the fluid.
    feq = _cfd_equilibrium(rho, ux, uy)
    post = f - (f - feq) / max(float(tau), 0.5001)


    # Moving solid: prescribe the actual local contour velocity, just as the
    # Bionic UFO model prescribes the resonator velocity inside its mask.
    rho[solid] = 1.0
    ux[solid] = wall_ux[solid]
    uy[solid] = wall_uy[solid]
    feq_wall = _cfd_equilibrium(rho, ux, uy)
    post[:, solid] = feq_wall[:, solid]

    # Periodic streaming, identical D2Q9 lattice directions and weights.
    streamed = np.empty_like(post)
    for k, (cx, cy) in enumerate(CFD_C):
        streamed[k] = np.roll(post[k], shift=(cy, cx), axis=(0, 1))
    return streamed


def _cfd_solid_mask(hull, center):
    gx = (np.arange(CFD_NX) + 0.5) * (BOX_X / CFD_NX)
    gy = (np.arange(CFD_NY) + 0.5) * (BOX_Y / CFD_NY)
    xx, yy = np.meshgrid(gx, gy)
    # Unwrap the contour around its center before rasterization so a body crossing
    # a periodic seam is rendered as one local object, not a line across the box.
    rel = minimum_image(np.asarray(hull) - np.asarray(center)[None, :])
    poly = np.asarray(center)[None, :] + rel
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    mask = MplPath(poly, closed=True).contains_points(pts).reshape(CFD_NY, CFD_NX)
    # Periodic copies are included at the domain edges.
    for sx in (-BOX_X, BOX_X):
        for sy in (-BOX_Y, BOX_Y):
            mask |= MplPath(poly + np.array([sx, sy]), closed=True).contains_points(pts).reshape(CFD_NY, CFD_NX)
    return mask


def _cfd_render_field(f, solid, vis_mode):
    rho = np.sum(f, axis=0)
    ux = np.sum(f * CFD_C[:, 0, None, None], axis=0) / np.maximum(rho, 1e-8)
    uy = np.sum(f * CFD_C[:, 1, None, None], axis=0) / np.maximum(rho, 1e-8)
    if vis_mode == "Pressure":
        data = rho - 1.0
        lim = float(np.percentile(np.abs(data[~solid]), 99.5)) if np.any(~solid) else 1e-5
        vmin, vmax = -max(lim, 1e-7), max(lim, 1e-7)
    elif vis_mode == "Pressure Deviation":
        data = np.abs(rho - 1.0)
        vmax = max(float(np.percentile(data[~solid], 99.7)) if np.any(~solid) else 1e-5, 1e-7)
        vmin = 0.0
    else:  # Vortices
        vort = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1) -
                np.roll(ux, -1, axis=0) + np.roll(ux, 1, axis=0)) * 0.5
        data = np.abs(vort)
        vmax = max(float(np.percentile(data[~solid], 99.7)) if np.any(~solid) else 1e-5, 1e-7)
        vmin = 0.0
    data = np.asarray(data, dtype=float)
    data[solid] = np.nan
    return np.nan_to_num(data, nan=0.0), vmin, vmax


def initialize(n_pairs, density, seed, n_hull_nodes):
    rng = np.random.default_rng(int(seed))
    n = int(n_pairs)
    scale = 1.0 / np.sqrt(max(float(density), 0.05))
    span_x = min(BOX_X, BOX_X * scale)
    span_y = min(BOX_Y, BOX_Y * scale)
    x0, x1 = (BOX_X - span_x) / 2.0, (BOX_X + span_x) / 2.0
    y0, y1 = (BOX_Y - span_y) / 2.0, (BOX_Y + span_y) / 2.0

    cx = rng.uniform(x0, x1, n)
    cy = rng.uniform(y0, y1, n)
    centers = np.column_stack((cx, cy))

    hull_center = np.array([HULL_INIT_X, HULL_INIT_Y], dtype=float)
    hull_init, _, _, _, _, _ = make_hull_nodes(
        "Fish — traveling wave", 0.0, 0.0, n_hull_nodes, False, center_xy=hull_center, hull_angle=0.0
    )
    rel = minimum_image(centers - hull_center)
    inside_poly = _point_in_polygon_mask(hull_center + rel, hull_init)
    scaled_r2 = (rel[:, 0] / (HULL_HALF_LEN + 0.15)) ** 2 + (rel[:, 1] / (HULL_R_MAX + 0.15)) ** 2
    inside = inside_poly | (scaled_r2 < 1.15)
    if np.any(inside):
        r_norm = np.linalg.norm(rel[inside], axis=1) + 1e-6
        dirs = rel[inside] / r_norm[:, None]
        target_r = np.maximum(r_norm, max(HULL_R_MAX + 0.25, 0.35)) * 1.25
        centers[inside] = (hull_center + dirs * target_r[:, None]) % np.array([BOX_X, BOX_Y])

    # Initialize near calibrated quantum equilibrium radius R_EQ_CALIBRATED
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    radii = rng.normal(R_EQ_CALIBRATED, R_EQ_CALIBRATED * 0.08, n)
    radii = np.clip(radii, 0.5 * R_EQ_CALIBRATED, 1.5 * R_EQ_CALIBRATED)
    offsets = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    # Place charges around center of mass so pair binding is initially relaxed
    m_tot = PARTICLE_MASS + ELECTRON_MASS
    pos_plus = (centers + (ELECTRON_MASS / m_tot) * offsets) % np.array([BOX_X, BOX_Y])
    pos_minus = (centers - (PARTICLE_MASS / m_tot) * offsets) % np.array([BOX_X, BOX_Y])
    pos = np.empty((2 * n, 2))
    pos[0::2] = pos_plus
    pos[1::2] = pos_minus

    # Zero-net-momentum thermal velocity initialization
    v_scale = 0.04 / np.sqrt(max(density, 0.05))
    vel = rng.normal(0.0, v_scale, pos.shape)
    vel[1::2] *= np.sqrt(PARTICLE_MASS / ELECTRON_MASS) * 0.35
    mass = np.tile([PARTICLE_MASS, ELECTRON_MASS], n)
    p_cm = np.sum(mass[:, None] * vel, axis=0) / np.sum(mass)
    vel -= p_cm[None, :]

    charge = np.tile([1.0, -1.0], n)
    species = np.tile([0, 1], n)
    return pos, vel, charge, mass, species


def run_experiment(
    mode,
    density,
    surface_density,
    body_mass_scale,
    damping_val,
    frequency,
    amplitude,
    frames,
    n_pairs,
    seed,
    running_charges=True,
    cfd_enabled=False,
    cfd_vis_mode="Vortices",
):
    n_hull_nodes = max(0, int(round(HULL_NODES_BASE * float(surface_density))))
    damping_coeff = float(np.clip(damping_val, 0.0, 1.0))

    # Neutral massive internal particles uniformly distributed inside bacterium
    n_internal_particles = max(0, int(round(INTERNAL_PARTICLES_BASE * float(body_mass_scale))))
    internal_norm = generate_internal_particles_local(n_internal_particles)

    shell_mass = float(n_hull_nodes) * (PARTICLE_MASS + ELECTRON_MASS) * HULL_MASS_MULTIPLIER
    internal_mass = float(n_internal_particles) * PARTICLE_MASS
    hull_mass = shell_mass + internal_mass

    pos, vel, charge, mass, species = initialize(n_pairs, density, seed, n_hull_nodes)
    n = len(pos)
    ion_idx = np.arange(0, n, 2)
    elec_idx = ion_idx + 1

    hull_center = np.array([HULL_INIT_X, HULL_INIT_Y], dtype=float)
    hull_vel = np.zeros(2, dtype=float)
    hull_angle = 0.0
    hull_omega = 0.0  # angular velocity (rad / unit time)

    collision_impulse_total = np.zeros(2)
    surface_force_total = np.zeros(2)
    phase = 0.0
    surf_weight = 1.0 if n_hull_nodes > 0 else 0.0
    use_running = bool(running_charges)

    total_momentum_history = []
    medium_momentum_history = []
    hull_momentum_history = []
    hull_force_history = []
    kinetic_history = []
    pos_history = []
    hull_history = []
    hull_ions_history = []
    hull_elecs_history = []
    internal_particles_history = []
    running_q_history = []
    hull_pos_history = []
    hull_center_history = []
    # CFD needs every physical substep: frame-only contours would alias the
    # 3 Hz travelling wave and produce an incorrect apparent wall velocity.
    cfd_hull_history = [] if cfd_enabled else None
    cfd_center_history = [] if cfd_enabled else None

    steps = int(frames) * SUBSTEPS_PER_FRAME
    for step in range(steps):
        phase += 2 * np.pi * float(frequency) * DT
        hull_pos, hull_ions, hull_elecs, _, run_w, len_scale = make_hull_nodes(
            mode,
            phase,
            float(amplitude),
            n_hull_nodes,
            running_charges=use_running,
            center_xy=hull_center,
            hull_angle=hull_angle,
        )
        internal_world = evaluate_internal_particles_world(
            internal_norm, mode, phase, float(amplitude), hull_center, hull_angle, len_scale
        )
        hull_inertia = _compute_hull_moment_of_inertia(
            hull_pos, hull_center, shell_mass, internal_world, internal_mass
        )

        # Conservative all-to-all medium forces (Coulomb + Quantum Kinetic + Fermi-Dirac, PBC)
        force = pairwise_medium_forces(pos, charge, species)

        # Conservative medium-surface interaction (PBC) — returns force and torque
        hull_force = np.zeros(2, dtype=float)
        hull_torque = 0.0
        if n_hull_nodes > 0 and surf_weight > 0.0:
            f_surf, f_react, t_react = hull_surface_forces(
                pos,
                species,
                charge,
                hull_ions,
                hull_elecs,
                hull_center,
                surf_weight,
                running_charge_weights=run_w,
            )
            force += f_surf
            hull_force = f_react
            hull_torque = t_react
            surface_force_total += f_react * DT

        # Velocity-Verlet first half-step
        acc = (force - damping_coeff * vel) / mass[:, None]
        vel += 0.5 * acc * DT
        pos = (pos + vel * DT) % np.array([BOX_X, BOX_Y])

        if hull_mass > 1e-12:
            hull_acc = hull_force / hull_mass
            hull_vel += 0.5 * hull_acc * DT
            hull_center = (hull_center + hull_vel * DT) % np.array([BOX_X, BOX_Y])
            disp_step = hull_vel * DT + 0.5 * hull_acc * (DT ** 2)
        else:
            hull_acc = np.zeros(2)
            disp_step = np.zeros(2)

        hull_alpha = hull_torque / hull_inertia
        hull_omega += 0.5 * hull_alpha * DT
        d_angle = hull_omega * DT
        hull_angle += d_angle

        if n_hull_nodes > 0:
            # Update hull geometry with step translation and rotation
            hull_pos, hull_ions, hull_elecs, _, run_w, len_scale = make_hull_nodes(
                mode,
                phase,
                float(amplitude),
                n_hull_nodes,
                running_charges=use_running,
                center_xy=hull_center,
                hull_angle=hull_angle,
            )

        # Elastic geometric core collision with linear & angular impulse exchange (MIC-aware)
        if n_hull_nodes >= 3:
            rel_to_hull = minimum_image(pos - hull_center[None, :])
            pos_in_hull_frame = hull_center[None, :] + rel_to_hull
            hit = _point_in_polygon_mask(pos_in_hull_frame, hull_pos)
            if np.any(hit):
                ids = np.where(hit)[0]
                diff = pos_in_hull_frame[ids][:, None, :] - hull_pos[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                nearest_idx = np.argmin(d2, axis=1)
                nearest_pts = hull_pos[nearest_idx]
                out_vec = pos_in_hull_frame[ids] - nearest_pts
                out_norm = np.linalg.norm(out_vec, axis=1) + 1e-12
                nn = out_vec / out_norm[:, None]

                # Lever arm r_arm from hull_center to contact point on boundary
                r_arm = nearest_pts - hull_center[None, :]
                pos[ids] = (nearest_pts + nn * (COULOMB_SOFTENING * 1.5)) % np.array([BOX_X, BOX_Y])

                if hull_mass > 1e-12:
                    m_p = mass[ids]
                    # Surface velocity at contact point includes rotational component: v_surf = v_cm + omega x r_arm
                    v_rot_surf = np.column_stack((-hull_omega * r_arm[:, 1], hull_omega * r_arm[:, 0]))
                    v_surf_contact = hull_vel[None, :] + v_rot_surf
                    rel_vn = np.sum((vel[ids] - v_surf_contact) * nn, axis=1)
                    approaching_rel = rel_vn < 0.0
                    if np.any(approaching_rel):
                        r_cross_n = (
                            r_arm[approaching_rel, 0] * nn[approaching_rel, 1]
                            - r_arm[approaching_rel, 1] * nn[approaching_rel, 0]
                        )
                        inv_eff_mass = (
                            (1.0 / m_p[approaching_rel])
                            + (1.0 / hull_mass)
                            + (r_cross_n ** 2) / hull_inertia
                        )
                        j_imp = (1.0 + WALL_RESTITUTION) * (-rel_vn[approaching_rel]) / inv_eff_mass
                        imp_vec = j_imp[:, None] * nn[approaching_rel]
                        vel[ids[approaching_rel]] += imp_vec / m_p[approaching_rel, None]

                        total_imp = np.sum(imp_vec, axis=0)
                        hull_vel -= total_imp / hull_mass
                        collision_impulse_total -= total_imp

                        # Angular impulse: dL = r_arm x (-imp_vec) = -j_imp * (r_arm x n)
                        ang_imp = float(np.sum(-j_imp * r_cross_n))
                        hull_omega += ang_imp / hull_inertia

        # Velocity-Verlet second half-step
        force2 = pairwise_medium_forces(pos, charge, species)

        hull_force2 = np.zeros(2, dtype=float)
        hull_torque2 = 0.0
        if n_hull_nodes > 0 and surf_weight > 0.0:
            hull_pos2, hull_ions2, hull_elecs2, _, run_w2, len_scale2 = make_hull_nodes(
                mode,
                phase,
                float(amplitude),
                n_hull_nodes,
                running_charges=use_running,
                center_xy=hull_center,
                hull_angle=hull_angle,
            )
            f_surf2, f_react2, t_react2 = hull_surface_forces(
                pos,
                species,
                charge,
                hull_ions2,
                hull_elecs2,
                hull_center,
                surf_weight,
                running_charge_weights=run_w2,
            )
            force2 += f_surf2
            hull_force2 = f_react2
            hull_torque2 = t_react2
            hull_ions = hull_ions2
            hull_elecs = hull_elecs2
            hull_pos = hull_pos2
            len_scale = len_scale2

        vel += 0.5 * ((force2 - damping_coeff * vel) / mass[:, None]) * DT
        if hull_mass > 1e-12:
            hull_vel += 0.5 * (hull_force2 / hull_mass) * DT

        internal_world2 = evaluate_internal_particles_world(
            internal_norm, mode, phase, float(amplitude), hull_center, hull_angle, len_scale
        )
        hull_inertia2 = _compute_hull_moment_of_inertia(
            hull_pos, hull_center, shell_mass, internal_world2, internal_mass
        )
        hull_omega += 0.5 * (hull_torque2 / hull_inertia2) * DT

        if step % SUBSTEPS_PER_FRAME == 0:
            p_med = np.sum(mass[:, None] * vel, axis=0)
            p_hull = hull_mass * hull_vel if hull_mass > 0 else np.zeros(2)
            medium_momentum_history.append(p_med)
            hull_momentum_history.append(p_hull)
            total_momentum_history.append(p_med + p_hull)
            hull_force_history.append(
                (collision_impulse_total + surface_force_total) /
                max((step + 1) * DT, 1e-12)
            )
            kin_medium = 0.5 * np.sum(mass[:, None] * vel * vel)
            kin_hull_trans = 0.5 * hull_mass * float(np.dot(hull_vel, hull_vel)) if hull_mass > 0 else 0.0
            kin_hull_rot = 0.5 * hull_inertia2 * (hull_omega ** 2)
            kinetic_history.append(kin_medium + kin_hull_trans + kin_hull_rot)
            pos_history.append(pos.copy())
            hull_history.append(hull_pos % np.array([BOX_X, BOX_Y]))
            hull_ions_history.append(hull_ions % np.array([BOX_X, BOX_Y]))
            hull_elecs_history.append(hull_elecs % np.array([BOX_X, BOX_Y]))
            internal_particles_history.append(internal_world2 % np.array([BOX_X, BOX_Y]))
            running_q_history.append(None if run_w is None else run_w.copy())
            hull_pos_history.append(hull_center.copy())
            hull_center_history.append(hull_center.copy())

        if cfd_enabled:
            cfd_hull_history.append(hull_pos % np.array([BOX_X, BOX_Y]))
            cfd_center_history.append(hull_center.copy())

    # Render animation
    tot_mom_arr = np.linalg.norm(np.asarray(total_momentum_history), axis=1)
    med_mom_arr = np.linalg.norm(np.asarray(medium_momentum_history), axis=1)
    force_arr = np.linalg.norm(np.asarray(hull_force_history), axis=1)
    kin_arr = np.asarray(kinetic_history)
    n_rendered = len(pos_history)
    frame_idx = np.arange(n_rendered)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=100)
    ax = axes[0]
    ax.set_facecolor("#10151d")
    ax.set_xlim(0, BOX_X)
    ax.set_ylim(0, BOX_Y)
    ax.set_aspect("equal")
    ax.set_title("Periodic Quantum-Classical Medium & Free Bacterium", color="white")
    ax.tick_params(colors="white")
    ax.set_xlabel("x (dimensionless units, periodic)", color="white")
    ax.set_ylabel("y (dimensionless units, periodic)", color="white")
    fig.patch.set_facecolor("#181e28")

    scat_ions = ax.scatter(
        pos_history[0][0::2, 0], pos_history[0][0::2, 1],
        s=13, c="#ff705c", label="Medium ions (+)", alpha=0.85
    )
    scat_elec = ax.scatter(
        pos_history[0][1::2, 0], pos_history[0][1::2, 1],
        s=8, c="#5db7ff", label="Medium electrons (−)", alpha=0.8
    )
    initial_hull_center = hull_center_history[0] if hull_center_history else np.array([HULL_INIT_X, HULL_INIT_Y])
    initial_hull_plot = initial_hull_center[None, :] + minimum_image(hull_history[0] - initial_hull_center[None, :]) if len(hull_history[0]) >= 3 else np.array([[0, 0], [0.01, 0], [0, 0.01]])
    hull_patch = plt.matplotlib.patches.Polygon(
        initial_hull_plot,
        closed=True,
        facecolor="#2b3340", edgecolor="none", linewidth=0.0, alpha=0.85
    )
    ax.add_patch(hull_patch)

    cfd_f = _cfd_init() if cfd_enabled else None
    cfd_image = None
    cfd_tau = 0.58 + 0.08 * (damping_coeff / max(DAMPING, 1e-9) - 1.0)
    cfd_tau = float(np.clip(cfd_tau, 0.51, 0.95))
    if cfd_enabled:
        cfd_solid0 = _cfd_solid_mask(hull_history[0], initial_hull_center)
        field0, cfd_vmin0, cfd_vmax0 = _cfd_render_field(cfd_f, cfd_solid0, cfd_vis_mode)

        cfd_image = ax.imshow(
            field0, extent=(0, BOX_X, 0, BOX_Y), origin="lower", aspect="equal",
            cmap=CFD_CMAP, vmin=cfd_vmin0, vmax=cfd_vmax0, interpolation="bilinear",
            alpha=0.88, zorder=0
        )

        
        

    scat_internal = ax.scatter(
        internal_particles_history[0][:, 0] if len(internal_particles_history[0]) else [],
        internal_particles_history[0][:, 1] if len(internal_particles_history[0]) else [],
        s=10, c="#b8c4d4", label="Internal neutral mass", alpha=0.85, zorder=3
    )

    scat_hull_ions = ax.scatter(
        hull_ions_history[0][:, 0] if len(hull_ions_history[0]) else [],
        hull_ions_history[0][:, 1] if len(hull_ions_history[0]) else [],
        s=9, c="#ff9e8f", label="Surface cores (+)", zorder=4
    )
    scat_hull_elecs = ax.scatter(
        hull_elecs_history[0][:, 0] if len(hull_elecs_history[0]) else [],
        hull_elecs_history[0][:, 1] if len(hull_elecs_history[0]) else [],
        s=7, c="#99d3ff", label="Surface electrons (−)", zorder=5
    )

    ax2 = axes[1]
    ax2.set_facecolor("#10151d")
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    ax2.title.set_color("white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#4a5568")

    line_tot_mom, = ax2.plot([], [], label="|Total momentum (med + hull)|", color="#ffd166")
    line_med_mom, = ax2.plot([], [], label="|Medium momentum|", color="#5db7ff", alpha=0.75)
    line_force, = ax2.plot([], [], label="Mean hull reaction force", color="#ff705c")
    line_kin, = ax2.plot([], [], label="Total kinetic energy", color="#06d6a0")
    y_max = max(
        float(np.max(tot_mom_arr)) if len(tot_mom_arr) else 1.0,
        float(np.max(med_mom_arr)) if len(med_mom_arr) else 1.0,
        float(np.max(force_arr)) if len(force_arr) else 1.0,
        float(np.max(kin_arr)) if len(kin_arr) else 1.0,
        1e-3,
    ) * 1.1
    ax2.set_xlim(0, max(n_rendered - 1, 1))
    ax2.set_ylim(0, y_max)
    ax2.set_title("Conservation & Force Diagnostics")
    ax2.set_xlabel("Frame")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    def update(i):
        nonlocal cfd_f
        p = pos_history[i]
        scat_ions.set_offsets(p[0::2])
        scat_elec.set_offsets(p[1::2])
        center_i = hull_center_history[i]
        hull_plot_i = center_i[None, :] + minimum_image(hull_history[i] - center_i[None, :])
        if len(hull_history[i]) >= 3:
            hull_patch.set_xy(hull_plot_i)
        if cfd_enabled:
            # Keep frame 0 perfectly undisturbed. Thereafter advance the CFD
            # once for each real bacterium physics substep between video frames.
            if i > 0:
                first_step = (i - 1) * SUBSTEPS_PER_FRAME + 1
                last_step = min(i * SUBSTEPS_PER_FRAME, len(cfd_hull_history) - 1)
                for cfd_step_idx in range(first_step, last_step + 1):
                    prev_hull = cfd_hull_history[cfd_step_idx - 1]
                    curr_hull = cfd_hull_history[cfd_step_idx]
                    prev_center = cfd_center_history[cfd_step_idx - 1]
                    curr_center = cfd_center_history[cfd_step_idx]
                    solid_step = _cfd_solid_mask(curr_hull, curr_center)
                    wall_ux, wall_uy = _cfd_wall_velocity_field(
                        solid_step, prev_hull, curr_hull,
                        prev_center, curr_center, DT
                    )
                    cfd_f = _cfd_step(cfd_f, solid_step, cfd_tau, wall_ux, wall_uy, p_damping=0.10)



                    
            solid_i = _cfd_solid_mask(hull_history[i], center_i)
            field_i, vmin_i, vmax_i = _cfd_render_field(cfd_f, solid_i, cfd_vis_mode)
            cfd_image.set_data(field_i)
            cfd_image.set_clim(vmin_i, vmax_i)
        if len(internal_particles_history[i]):
            scat_internal.set_offsets(internal_particles_history[i])
        else:
            scat_internal.set_offsets(np.empty((0, 2)))
        scat_hull_ions.set_offsets(center_i[None, :] + minimum_image(hull_ions_history[i] - center_i[None, :]))
        scat_hull_elecs.set_offsets(center_i[None, :] + minimum_image(hull_elecs_history[i] - center_i[None, :]))
        if running_q_history[i] is not None:
            rq = running_q_history[i]
            scat_hull_ions.set_color(["#ff3333" if v >= 0 else "#ff9e8f" for v in rq])
            scat_hull_elecs.set_color(["#238cff" if v <= 0 else "#99d3ff" for v in rq])
        else:
            scat_hull_ions.set_color("#ff9e8f")
            scat_hull_elecs.set_color("#99d3ff")
        x_slice = frame_idx[: i + 1]
        line_tot_mom.set_data(x_slice, tot_mom_arr[: i + 1])
        line_med_mom.set_data(x_slice, med_mom_arr[: i + 1])
        line_force.set_data(x_slice, force_arr[: i + 1])
        line_kin.set_data(x_slice, kin_arr[: i + 1])
        return (
            scat_ions, scat_elec, hull_patch, scat_internal,
            scat_hull_ions, scat_hull_elecs,
            *((cfd_image,) if cfd_enabled else ()),
            line_tot_mom, line_med_mom, line_force, line_kin
        )

    anim = FuncAnimation(fig, update, frames=n_rendered, interval=50, blit=True)
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.close()
    writer = FFMpegWriter(fps=20, bitrate=1800)
    anim.save(tmp_video.name, writer=writer)
    plt.close(fig)

    # Nearest-neighbor e-i distance across all free charges using MIC
    d_ei = np.linalg.norm(
        minimum_image(pos[ion_idx, None, :] - pos[None, elec_idx, :]), axis=2
    )
    mean_nn_ei = float(np.mean(np.min(d_ei, axis=1)))

    # Unwrapped total displacement of bacterium
    hull_traj = np.asarray(hull_pos_history)
    traj_diffs = minimum_image(np.diff(hull_traj, axis=0)) if len(hull_traj) > 1 else np.zeros((0, 2))
    displacement = np.sum(traj_diffs, axis=0)

    net_charge = float(np.sum(charge))
    final_medium_momentum = np.sum(mass[:, None] * vel, axis=0)
    final_hull_momentum = hull_mass * hull_vel if hull_mass > 0 else np.zeros(2)
    final_total_momentum = final_medium_momentum + final_hull_momentum
    mean_force = (collision_impulse_total + surface_force_total) / max(steps * DT, 1e-12)
    charge_mode_label = "Running (curvature + phase)" if use_running else "Uniform neutral dipole"
    avg_spacing = REF_PERIMETER / max(n_hull_nodes, 1) if n_hull_nodes > 0 else float("inf")

    report = (
        f"Model: Periodic (PBC) Effective Quantum-Classical 2D Medium\n"
        f"Medium particles: {n} ({n_pairs} ions + {n_pairs} electrons); net charge: {net_charge:.3g} e\n"
        f"Calibrated e-i equilibrium radius r_eq: {R_EQ_CALIBRATED:.4f} | "
        f"Mean nearest-neighbor e-i dist: {mean_nn_ei:.4f}\n"
        f"Hull surface nodes: {n_hull_nodes} neutral dipole pairs ({charge_mode_label}); "
        f"internal neutral particles: {n_internal_particles}; total mass: {hull_mass:.4g}\n"
        f"Mean surface spacing: {avg_spacing:.4f} | "
        f"Bacterium displacement: ({displacement[0]:.4f}, {displacement[1]:.4f}); "
        f"vel: ({hull_vel[0]:.4g}, {hull_vel[1]:.4g}); "
        f"angle: {np.degrees(hull_angle):.2f} deg (omega: {hull_omega:.4g} rad/t)\n"
        f"Final medium momentum: ({final_medium_momentum[0]:.4g}, {final_medium_momentum[1]:.4g}) | "
        f"hull momentum: ({final_hull_momentum[0]:.4g}, {final_hull_momentum[1]:.4g})\n"
        f"Total system momentum (med + hull): ({final_total_momentum[0]:.4g}, {final_total_momentum[1]:.4g})\n"
        f"Mean estimated hull reaction: ({mean_force[0]:.4g}, {mean_force[1]:.4g}) | "
        f"Medium density: {density:.3g} | Surface density: {surface_density:.3g} | "
        f"Damping: {damping_coeff:.3g} | CFD viscosity tau: {cfd_tau:.3f} (visualization only)"
    )
    return tmp_video.name, report


def toggle_running_charges_btn(current_state: bool):
    new_state = not bool(current_state)
    label = "Charge On" if new_state else "Charge Off"
    return new_state, gr.update(value=label, variant="primary" if new_state else "secondary")


theme = gr.themes.Default(primary_hue="blue")

with gr.Blocks(title="Microbio Unified Medium — PBC Quantum-Classical", theme=theme) as demo:
    gr.Markdown(
        "# Microbio Unified Medium\n"
        "**Scope note:** Effective semiclassical particle approximation (not an exact many-body "
        "Schrödinger solver); parameters require calibration for target substance and temperature.\n\n"
        "- Periodic Boundary Conditions (PBC) with Minimum Image Convention (continuous infinite medium)\n"
        "- Fermi-Dirac statistical repulsion between identical fermions\n"
        "- Exact linear & angular momentum-conserving medium-surface action-reaction"
    )
    running_charges_state = gr.State(value=True)
    with gr.Row():
        mode = gr.Radio(
            ["Fish — traveling wave", "Jellyfish — symmetric oscillation"],
            value="Fish — traveling wave", label="Resonator kinematics"
        )
        density = gr.Slider(0.2, 3.0, value=0.2, step=0.1, label="Medium particle density")
        surface_density = gr.Slider(
            0.0, 10.0, value=0.5, step=0.1,
            label="Shell particle density (number of surface pairs)"
        )
        body_mass_scale = gr.Slider(
            0.0, 10.0, value=0.0, step=0.1,
            label="Bacterium internal mass (neutral particles)"
        )
    with gr.Row():
        frequency = gr.Slider(0.01, 1.0, value=0.11, step=0.01, label="Frequency")
        amplitude = gr.Slider(0.0, 3.0, value=0.5, step=0.05, label="Oscillation amplitude")
        damping_val = gr.Slider(0.0, 1.0, value=DAMPING, step=0.005, label="Damping / CFD viscosity (water default)")
        frames = gr.Slider(20, 500, value=100, step=10, label="Number of frames")
    with gr.Row():
        cfd_enabled = gr.Checkbox(value=False, label="Enable CFD visualization (visualization only)")
        cfd_vis_mode = gr.Radio(
            choices=["Vortices", "Pressure", "Pressure Deviation"],
            value="Vortices", label="CFD color mode"
        )
    with gr.Row():
        n_pairs = gr.Slider(10, 300, value=20, step=10, label="Number of medium pairs")
        seed = gr.Number(value=SEED, precision=0, label="Seed")
        running_charge_btn = gr.Button("Charge On", variant="primary")
    run = gr.Button("Run experiment", variant="primary")
    video_out = gr.Video(label="Medium state and diagnostics", autoplay=True)
    report = gr.Textbox(label="Numerical results", lines=9)

    running_charge_btn.click(
        toggle_running_charges_btn,
        inputs=[running_charges_state],
        outputs=[running_charges_state, running_charge_btn],
    )
    run.click(
        run_experiment,
        inputs=[
            mode, density, surface_density, body_mass_scale, damping_val,
            frequency, amplitude, frames, n_pairs, seed, running_charges_state,
            cfd_enabled, cfd_vis_mode,
        ],
        outputs=[video_out, report],
    )

if __name__ == "__main__":
    demo.launch()
