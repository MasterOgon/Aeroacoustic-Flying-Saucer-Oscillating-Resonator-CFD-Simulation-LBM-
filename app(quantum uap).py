
import tempfile
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
import gradio as gr


# ============================================================
# PART 1: MICROBIO (16:9) CONSTANTS, PBC & PARTICLE/HULL PHYSICS
# ============================================================

FPS = 30
SEED = 7

# 16:9 domain preserving exact physical scales of microbio.py
BOX_X = 6.0
BOX_Y = 3.375  # 6.0 * (9 / 16)
DT = 0.075
SUBSTEPS_PER_FRAME = 8  # 12 * 0.05 = 0.6 physical time units per frame

# Fundamental and calibrated constants
PARTICLE_MASS = 1.0
ELECTRON_MASS = 0.08
COULOMB_K = 0.0036
COULOMB_SOFTENING = 0.25
Q_KIN_EPS = 0.10
Q_KIN_SCALE = 0.00003
DAMPING = 0.015
MEDIUM_TEMP_DEFAULT = 0.005
MEDIUM_TEMP_MAX = 0.1
MEDIUM_THERMAL_COUPLING = 0.05

# Effective Fermi-Dirac statistical repulsion potential for identical fermions
FD_KT_ELEC = 0.018
FD_RF_ELEC = 0.095
FD_LAMBDA_ELEC = 0.025
FD_KT_CORE = 0.045
FD_RF_CORE = 0.190
FD_LAMBDA_CORE = 0.030
FD_CUTOFF_MULT = 8.0

# Hull geometry
HULL_LENGTH = 2.40
HULL_HALF_LEN = 0.5 * HULL_LENGTH
HULL_R_MAX = 0.15 #(2.0 / 3.0)

HULL_CAP_START = 0.80


HULL_INIT_X = BOX_X / 2.0
HULL_INIT_Y = BOX_Y / 2.0
HULL_NODES_BASE = 64
HULL_MASS_MULTIPLIER = 1.0
INTERNAL_PARTICLES_BASE = 40
SURFACE_DIPOLE_OFFSET = 0.035
INNER_WALL_CLEARANCE = SURFACE_DIPOLE_OFFSET * 0.5 + 0.045
EXTERIOR_PARTICLE_CLEARANCE = 0.00002
WALL_RESTITUTION = 1.0
FISH_SPATIAL_K = 1.8 * np.pi
CURVATURE_GAIN = 6.0

# Trail and LBM / field grid parameters
TRAIL_LEN = 0.0
CFD_NX, CFD_NY = 160, 90   # LBM & plasma field grid (16:9)
PHI_NX, PHI_NY = 120, 68   # Electric potential grid (16:9)
CAPSULE_LAYERS = 14        # Number of nested gradient layers for metallic capsule


# ----------------------------- Periodic Boundaries & Transforms -----------------------------

def minimum_image(d: np.ndarray) -> np.ndarray:
    """Minimum Image Convention (MIC) in a 2D periodic domain [0, BOX_X) x [0, BOX_Y)."""
    d_mic = d.copy()
    d_mic[..., 0] -= BOX_X * np.round(d_mic[..., 0] / BOX_X)
    d_mic[..., 1] -= BOX_Y * np.round(d_mic[..., 1] / BOX_Y)
    return d_mic


def rotate_2d(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2D points around the origin by `angle` radians."""
    if len(pts) == 0:
        return pts.copy()
    c, s = np.cos(angle), np.sin(angle)
    x = pts[:, 0] * c - pts[:, 1] * s
    y = pts[:, 0] * s + pts[:, 1] * c
    return np.column_stack((x, y))


def _point_in_polygon_mask(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    if len(poly) < 3 or len(points) == 0:
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


def _get_unwrapped_hull_poly(hull_pos: np.ndarray, hull_center: np.ndarray) -> np.ndarray:
    """Returns hull polygon in continuous coordinates relative to hull_center."""
    if len(hull_pos) == 0:
        return np.zeros((0, 2), dtype=float)
    rel = minimum_image(np.asarray(hull_pos, dtype=float) - np.asarray(hull_center, dtype=float)[None, :])
    return np.asarray(hull_center, dtype=float)[None, :] + rel


def mask_points_inside_hull_periodic(
    points: np.ndarray,
    hull_unwrapped: np.ndarray,
    hull_center: np.ndarray,
) -> np.ndarray:
    """Checks if points lie inside the polygon under periodic boundary conditions (PBC)."""
    if len(points) == 0 or len(hull_unwrapped) < 3:
        return np.zeros(len(points), dtype=bool)
    pts_in_hull_frame = hull_center[None, :] + minimum_image(points - hull_center[None, :])
    return _point_in_polygon_mask(pts_in_hull_frame, hull_unwrapped)


def enforce_exterior_hull_clearance(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    hull_pos: np.ndarray,
    hull_center: np.ndarray,
    hull_vel: np.ndarray,
    hull_omega: float,
    hull_mass: float,
    hull_inertia: float,
    hull_deform_vel: np.ndarray = None,
    clearance: float = EXTERIOR_PARTICLE_CLEARANCE,
    update_hull_momentum: bool = True,
    disable_pitch_rotation: bool = False,
):
    """
    Сплошная экранирующая оболочка: выталкивает частицы, пересекшие контур
    или вошедшие в пограничный слой, и передает корпусу полный упругий импульс
    с учетом поступательной, вращательной и деформационной скорости стенки.
    """
    if len(hull_pos) < 3 or len(pos) == 0:
        return hull_vel, (0.0 if disable_pitch_rotation else hull_omega)

    hull_uw = _get_unwrapped_hull_poly(hull_pos, hull_center)
    pts_frame = hull_center[None, :] + minimum_image(pos - hull_center[None, :])

    seg_a = hull_uw
    seg_b = np.roll(hull_uw, -1, axis=0)
    ab = seg_b - seg_a
    ab_len2 = np.sum(ab * ab, axis=1) + 1e-18

    seg_mid = 0.5 * (seg_a + seg_b)
    n_cand = np.column_stack((-ab[:, 1], ab[:, 0]))
    n_unit = n_cand / (np.linalg.norm(n_cand, axis=1, keepdims=True) + 1e-12)
    probe_out = _point_in_polygon_mask(seg_mid + 1e-4 * n_unit, hull_uw)
    outward_n = n_unit * np.where(probe_out[:, None], -1.0, 1.0)

    if hull_deform_vel is not None and len(hull_deform_vel) == len(hull_pos):
        seg_deform_vel = 0.5 * (hull_deform_vel + np.roll(hull_deform_vel, -1, axis=0))
    else:
        seg_deform_vel = np.zeros_like(hull_pos)

    for _ in range(2):
        inside = _point_in_polygon_mask(pts_frame, hull_uw)
        ap = pts_frame[:, None, :] - seg_a[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab_len2[None, :], 0.0, 1.0)
        closest = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
        diff = pts_frame[:, None, :] - closest
        dist = np.linalg.norm(diff, axis=2)

        min_seg_idx = np.argmin(dist, axis=1)
        min_dist = dist[np.arange(len(pts_frame)), min_seg_idx]
        nearest_pts = closest[np.arange(len(pts_frame)), min_seg_idx]
        seg_out_n = outward_n[min_seg_idx]

        violated = inside | (min_dist < clearance)
        if not np.any(violated):
            break

        ids = np.where(violated)[0]
        radial_vec = pts_frame[ids] - nearest_pts[ids]
        radial_norm = np.linalg.norm(radial_vec, axis=1, keepdims=True)
        use_radial = (~inside[ids])[:, None] & (radial_norm > 1e-9)
        nn = np.where(use_radial, radial_vec / np.maximum(radial_norm, 1e-12), seg_out_n[ids])

        # Выводим частицу строго на внешнюю поверхность экранирующего слоя
        pts_frame[ids] = nearest_pts[ids] + nn * clearance
        pos[ids] = pts_frame[ids] % np.array([BOX_X, BOX_Y])

        # Полная скорость точки поверхности: движение центра + вращение + изгиб стенки
        r_arm = nearest_pts[ids] - hull_center[None, :]
        eff_omega = 0.0 if disable_pitch_rotation else hull_omega
        v_rot_surf = np.column_stack((-eff_omega * r_arm[:, 1], eff_omega * r_arm[:, 0]))
        v_def_surf = seg_deform_vel[min_seg_idx[ids]]
        v_surf_contact = hull_vel[None, :] + v_rot_surf + v_def_surf

        rel_v = vel[ids] - v_surf_contact
        rel_vn = np.sum(rel_v * nn, axis=1)

        # Любая частица, оказавшаяся внутри (inside), гарантированно пересекла стенку
        collided = (rel_vn < 0.0) | inside[ids]

        if np.any(collided):
            app_ids = ids[collided]
            nn_app = nn[collided]
            r_app = r_arm[collided]
            is_inside_app = inside[app_ids]
            raw_vn = rel_vn[collided]

            # Если частица проскочила глубже середины и уже летит к противоположной стенке,
            # берем модуль нормальной скорости, чтобы корректно отразить её обратно
            vn_app = np.where(is_inside_app & (raw_vn > 0.0), -raw_vn, raw_vn)

            if update_hull_momentum and hull_mass > 1e-12:
                m_p = mass[app_ids]
                r_cross_n = r_app[:, 0] * nn_app[:, 1] - r_app[:, 1] * nn_app[:, 0]
                inv_rot = 0.0 if disable_pitch_rotation else (r_cross_n ** 2) / max(hull_inertia, 1e-6)
                inv_eff_mass = (1.0 / m_p) + (1.0 / hull_mass) + inv_rot
                j_imp = (1.0 + WALL_RESTITUTION) * (-vn_app) / inv_eff_mass
                imp_vec = j_imp[:, None] * nn_app

                vel[app_ids] += imp_vec / m_p[:, None]
                hull_vel = hull_vel - np.sum(imp_vec, axis=0) / hull_mass
                if not disable_pitch_rotation:
                    ang_imp = float(np.sum(-j_imp * r_cross_n))
                    hull_omega = float(hull_omega + ang_imp / max(hull_inertia, 1e-6))
                else:
                    hull_omega = 0.0
            else:
                vel[app_ids] -= (1.0 + WALL_RESTITUTION) * vn_app[:, None] * nn_app

    return hull_vel, (0.0 if disable_pitch_rotation else hull_omega)


# ----------------------------- Medium Heat Source & Pairwise Forces -----------------------------

def _medium_target_v_rms(medium_temp: float, mass: np.ndarray) -> np.ndarray:
    t_eff = max(0.0, float(medium_temp))
    return np.sqrt(t_eff / np.maximum(mass, 1e-12))[:, None]


def _apply_medium_heat_source(
    vel: np.ndarray,
    mass: np.ndarray,
    medium_temp: float,
    damping_coeff: float,
    rng: np.random.Generator,
    dt: float,
    flow_drift_vx: float = 0.0,
) -> None:
    """Точный термостат Ланжевена (шаг O в схеме BAOAB), связанный с вязким трением."""
    if len(vel) == 0:
        return
    t_target = max(0.0, float(medium_temp))

    # Общий коэффициент затухания: вязкое трение (gamma_damp = damping / m) + базовая связь
    gamma = MEDIUM_THERMAL_COUPLING + (float(damping_coeff) / np.maximum(mass, 1e-12))
    c = np.exp(-gamma * dt)[:, None]
    sigma = np.sqrt(np.maximum(0.0, 1.0 - c * c))

    # Переходим в систему отсчёта потока
    v_rel = vel.copy()
    v_rel[:, 0] -= flow_drift_vx
    p_before = np.sum(mass[:, None] * v_rel, axis=0)

    if t_target > 0.0:
        v_target = _medium_target_v_rms(t_target, mass)
        thermal_kick = rng.normal(0.0, 1.0, size=v_rel.shape) * v_target
        v_rel = c * v_rel + sigma * thermal_kick
    else:
        # При T = 0 термостат работает как чистое вязкое охлаждение
        v_rel = c * v_rel

    # Сохраняем суммарный импульс среды (чтобы случайный шум не создавал ложный дрейф)
    p_after = np.sum(mass[:, None] * v_rel, axis=0)
    v_rel -= ((p_after - p_before) / np.sum(mass))[None, :]

    # Возвращаемся в лабораторную систему отсчёта
    v_rel[:, 0] += flow_drift_vx
    vel[:] = v_rel


def pairwise_medium_forces(pos: np.ndarray, charge: np.ndarray, species: np.ndarray) -> np.ndarray:
    """Conservative forces between all free medium charges (Coulomb + Q_kin + Fermi-Dirac, MIC)."""
    if len(pos) < 2:
        return np.zeros_like(pos)
    d = minimum_image(pos[:, None, :] - pos[None, :, :])
    r2_raw = np.sum(d * d, axis=2)
    r_raw = np.sqrt(r2_raw + 1e-18)

    r2_soft = r2_raw + COULOMB_SOFTENING ** 2
    np.fill_diagonal(r2_soft, np.inf)
    invr3 = r2_soft ** (-1.5)
    qq = charge[:, None] * charge[None, :]
    f_coul = COULOMB_K * (qq * invr3)[:, :, None] * d

    opp_mask = species[:, None] != species[None, :]
    np.fill_diagonal(opp_mask, False)
    denom_q = (r2_raw + Q_KIN_EPS ** 2) ** 2
    qkin_coeff = (2.0 * Q_KIN_SCALE / denom_q) * opp_mask
    f_qkin = qkin_coeff[:, :, None] * d

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


def _compute_hull_ion_charge_weights(
    n_nodes: int,
    surface_charge_weight: float,
    running_charge_weights=None,
):
    if n_nodes <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
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
        return ion_w, elec_w
    return (
        np.full(n_nodes, surface_charge_weight, dtype=float),
        np.full(n_nodes, surface_charge_weight, dtype=float),
    )


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
    """Conservative interaction between medium charges and hull surface dipoles."""
    n_nodes = len(hull_ions)
    if n_nodes == 0 or len(pos) == 0:
        return np.zeros_like(pos), np.zeros(2), 0.0

    hull_pos_charges = np.vstack([hull_ions, hull_elecs])
    ion_w, elec_w = _compute_hull_ion_charge_weights(
        n_nodes, surface_charge_weight, running_charge_weights
    )
    hull_q = np.concatenate([ion_w, -elec_w])

    dr = minimum_image(pos[:, None, :] - hull_pos_charges[None, :, :])
    r2_raw = np.sum(dr * dr, axis=2)
    r2_soft = r2_raw + COULOMB_SOFTENING ** 2
    qq = charge[:, None] * hull_q[None, :]
    pair_f_coul = COULOMB_K * ((qq * (r2_soft ** (-1.5)))[:, :, None] * dr)

    pair_f_qkin = np.zeros_like(dr)
    pair_f_fd = np.zeros_like(dr)

    core_mask = species == 0
    elec_mask = species == 1

    if np.any(core_mask):
        p_core = pos[core_mask]
        dr_cc = minimum_image(p_core[:, None, :] - hull_ions[None, :, :])
        r_cc = np.linalg.norm(dr_cc, axis=2) + 1e-12
        arg_cc = np.clip((r_cc - FD_RF_CORE) / FD_LAMBDA_CORE, -40.0, 40.0)
        mag_cc = (FD_KT_CORE / FD_LAMBDA_CORE) * ion_w[None, :] / (1.0 + np.exp(arg_cc))
        mag_cc[r_cc > (FD_RF_CORE + FD_CUTOFF_MULT * FD_LAMBDA_CORE)] = 0.0
        pair_f_fd[core_mask, :n_nodes, :] += (mag_cc / r_cc)[:, :, None] * dr_cc

        dr_ce = minimum_image(p_core[:, None, :] - hull_elecs[None, :, :])
        r2_ce = np.sum(dr_ce * dr_ce, axis=2)
        coeff_ce = (2.0 * Q_KIN_SCALE * elec_w[None, :]) / ((r2_ce + Q_KIN_EPS ** 2) ** 2)
        pair_f_qkin[core_mask, n_nodes:, :] += coeff_ce[:, :, None] * dr_ce

    if np.any(elec_mask):
        p_elec = pos[elec_mask]
        dr_ee = minimum_image(p_elec[:, None, :] - hull_elecs[None, :, :])
        r_ee = np.linalg.norm(dr_ee, axis=2) + 1e-12
        arg_ee = np.clip((r_ee - FD_RF_ELEC) / FD_LAMBDA_ELEC, -40.0, 40.0)
        mag_ee = (FD_KT_ELEC / FD_LAMBDA_ELEC) * elec_w[None, :] / (1.0 + np.exp(arg_ee))
        mag_ee[r_ee > (FD_RF_ELEC + FD_CUTOFF_MULT * FD_LAMBDA_ELEC)] = 0.0
        pair_f_fd[elec_mask, n_nodes:, :] += (mag_ee / r_ee)[:, :, None] * dr_ee

        dr_ec = minimum_image(p_elec[:, None, :] - hull_ions[None, :, :])
        r2_ec = np.sum(dr_ec * dr_ec, axis=2)
        coeff_ec = (2.0 * Q_KIN_SCALE * ion_w[None, :]) / ((r2_ec + Q_KIN_EPS ** 2) ** 2)
        pair_f_qkin[elec_mask, :n_nodes, :] += coeff_ec[:, :, None] * dr_ec

    pair_total_on_med = pair_f_coul + pair_f_qkin + pair_f_fd
    total_force_on_medium = np.sum(pair_total_on_med, axis=1)

    force_on_hull_nodes = -np.sum(pair_total_on_med, axis=0)
    reaction_on_hull = np.sum(force_on_hull_nodes, axis=0)

    r_arm = minimum_image(hull_pos_charges - hull_center[None, :])
    torque_on_hull = float(
        np.sum(r_arm[:, 0] * force_on_hull_nodes[:, 1] - r_arm[:, 1] * force_on_hull_nodes[:, 0])
    )
    return total_force_on_medium, reaction_on_hull, torque_on_hull


# ----------------------------- Hull Geometry & Kinematics -----------------------------

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
    if isinstance(mode, str) and ("jellyfish" in mode.lower() or "медуза" in mode.lower()):
        return 0.0
    return 1.0

def _evaluate_bacterium_deformation(
    x_norm: np.ndarray,
    phase: float,
    amplitude: float,
    mode_fish: float,
    sharpness: float = 0.0,
):
    sh = float(np.clip(sharpness, 0.0, 1.0))

    # 1. Медуза: быстрый полутакт — когда середина прогибается вверх (sin(phase) растёт около phase = 0, 2*pi...).
    # Увеличиваем этот пик скорости до ~2.3x, а возврат замедляем, сохраняя монотонность (производная > 0).
    jelly_asym = (0.45 + 0.50 * sh) * np.sin(phase) + (0.18 * sh) * np.sin(2.0 * phase)
    asym_phase_jelly = phase + jelly_asym
        # 2. Рыба: сжатие каждого полупериода (pi) в быстрый импульс волны + пауза между полутактами.
    # Коэффициенты подобраны так, чтобы производная была строго >= 0 (без обратного хода волны),
    # но в момент импульса волна пробегала по всему телу в ~4.5 раза быстрее базовой частоты.
    phase_shifted = phase - 0.30 * np.pi
    fish_asym = sh * (0.48 * np.sin(2.0 * phase_shifted) + 0.08 * np.sin(4.0 * phase_shifted))
    asym_phase_fish = phase + fish_asym

    
        

    
    asym_phase = mode_fish * asym_phase_fish + (1.0 - mode_fish) * asym_phase_jelly

    s = 0.5 * (x_norm + 1.0)
    ds_dx = 1.0 / max(HULL_LENGTH, 1e-9)

    center_bow = np.maximum(0.0, 1.0 - x_norm ** 2)
    bow_amp = amplitude * 0.22
    bow_offset = bow_amp * np.sin(asym_phase) * center_bow * (1.0 - mode_fish)

    # Классическая плавная волна рыбы без искажения формы
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
    sharpness: float = 0.0,
):
    theta = 2.0 * np.pi * u
    s = 0.5 - 0.5 * np.cos(theta)
    x_norm = 2.0 * s - 1.0

    y_local_center, d2y_dx2, asym_phase, wave_arg = _evaluate_bacterium_deformation(
        x_norm, phase, amplitude, mode_fish, sharpness=sharpness
    )

    r_prof = _bacterium_radius_profile(x_norm, r_max=HULL_R_MAX, cap_start=HULL_CAP_START)
    side_sign = np.where(np.sin(theta) >= 0.0, 1.0, -1.0)
    x_raw = x_norm * HULL_HALF_LEN
    y_raw = y_local_center + side_sign * r_prof

    curvature = -d2y_dx2

    asymmetry_fish = 1.0 + 0.35 * x_norm
    phase_travel_fish = np.cos(wave_arg + 0.25 * asym_phase)
    q_fish = np.tanh(side_sign * curvature * CURVATURE_GAIN + 0.45 * phase_travel_fish) * asymmetry_fish

    abs_x = np.abs(x_norm)
    jelly_k = 1.2 * np.pi
    radial_flow = np.cos(asym_phase - jelly_k * abs_x)
    q_jelly = np.tanh(side_sign * curvature * CURVATURE_GAIN + 0.65 * radial_flow)

    q_running = mode_fish * q_fish + (1.0 - mode_fish) * q_jelly
    return x_raw, y_raw, q_running




def generate_internal_particles_local(n_internal: int) -> np.ndarray:
    n_internal = max(0, int(n_internal))
    if n_internal == 0:
        return np.zeros((0, 2), dtype=float)

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

    r_local_full = np.maximum(_bacterium_radius_profile(vx), 1e-6)
    y_frac = vy / r_local_full

    if len(vx) >= n_internal:
        idx = np.linspace(0, len(vx) - 1, n_internal, dtype=int)
        return np.column_stack((vx[idx], y_frac[idx]))

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


def _constrain_points_inside_hull(
    pts_local: np.ndarray,
    hull_local: np.ndarray,
    clearance: float = INNER_WALL_CLEARANCE,
) -> np.ndarray:
    """Keeps internal particles strictly inside the closed hull contour."""
    if len(pts_local) == 0 or len(hull_local) < 3:
        return pts_local

    constrained = pts_local.copy()
    seg_a = hull_local
    seg_b = np.roll(hull_local, -1, axis=0)
    ab = seg_b - seg_a
    ab_len2 = np.sum(ab * ab, axis=1) + 1e-18

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
    sharpness: float = 0.0,
) -> np.ndarray:
    if len(internal_norm) == 0:
        return np.zeros((0, 2), dtype=float)
    mode_fish = _mode_blend(mode)

    u_fine = np.linspace(0.0, 1.0, 400, endpoint=False)
    x_f, y_f, _ = _evaluate_bacterium_contour(u_fine, phase, float(amplitude), mode_fish, sharpness=sharpness)
    cx_f = float(np.mean(x_f))
    cy_f = float(np.mean(y_f))
    hull_local = np.column_stack(((x_f - cx_f) * length_scale, (y_f - cy_f) * length_scale))

    x_norm = internal_norm[:, 0]
    y_frac = internal_norm[:, 1]
    y_center_off, _, _, _ = _evaluate_bacterium_deformation(
        x_norm, phase, float(amplitude), mode_fish, sharpness=sharpness
    )

    eps_x = 1e-3
    x_plus = np.clip(x_norm + eps_x, -1.0, 1.0)
    x_minus = np.clip(x_norm - eps_x, -1.0, 1.0)
    y_plus, _, _, _ = _evaluate_bacterium_deformation(x_plus, phase, float(amplitude), mode_fish, sharpness=sharpness)
    y_minus, _, _, _ = _evaluate_bacterium_deformation(x_minus, phase, float(amplitude), mode_fish, sharpness=sharpness)
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
    local_pts = _constrain_points_inside_hull(local_pts, hull_local, clearance=INNER_WALL_CLEARANCE)

    rot_pts = rotate_2d(local_pts, hull_angle)
    return rot_pts + center_xy[None, :]


def make_hull_nodes(
    mode,
    phase: float,
    amplitude: float,
    n_nodes: int,
    running_charges: bool = False,
    center_xy=None,
    hull_angle: float = 0.0,
    sharpness: float = 0.0,
):
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
        u_fine, phase, float(amplitude), mode_fish, sharpness=sharpness
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

    x = np.interp(target_s, cum_s, np.append(x_s, x_s[0]))
    y = np.interp(target_s, cum_s, np.append(y_s, y_s[0]))
    q_nodes = np.interp(target_s, cum_s, np.append(q_f, q_f[0]))

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
    normals_local = np.column_stack((n1_x, n1_y)) if dot1 >= 0.0 else np.column_stack((n2_x, n2_y))

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
    i_shell = 0.0
    if len(hull_pos) > 0 and shell_mass > 0.0:
        r_s = hull_pos - hull_center[None, :]
        i_shell = (shell_mass / len(hull_pos)) * float(np.sum(np.sum(r_s * r_s, axis=1)))

    i_int = 0.0
    if len(internal_world) > 0 and internal_mass > 0.0:
        r_i = internal_world - hull_center[None, :]
        i_int = (internal_mass / len(internal_world)) * float(np.sum(np.sum(r_i * r_i, axis=1)))

    return max(i_shell + i_int, 1e-4)


# ============================================================
# PART 2: LBM CFD, POTENTIAL FIELD & CAPSULE VISUALIZATION
# ============================================================

CFD_CMAP = LinearSegmentedColormap.from_list(
    "cfd_multicolor",
    [
        (0.00, "#004d99"),
        (0.10, "#00b7ff"),
        (0.25, "#00ffd0"),
        (0.40, "#00ff66"),
        (0.60, "#dfff00"),
        (0.85, "#ff9a00"),
        (1.00, "#ff2200"),
    ],
    N=512,
)
CFD_CMAP.set_bad(color="#000000")

ION_FLOW_CMAP = LinearSegmentedColormap.from_list(
    "cosmic_ion_plasma",
    [
        "#000000",
        "#040814",
        "#091938",
        "#0f3a6e",
        "#1373a8",
        "#22b8cf",
        "#7ef9ff",
    ],
    N=256,
)
ION_FLOW_CMAP.set_bad(color="#000000")

ION_COCOON_CMAP = LinearSegmentedColormap.from_list(
    "ion_cocoon_plasma",
    [
        (0.00, "#000000"),
        (0.20, "#030b1c"),
        (0.42, "#0a2458"),
        (0.65, "#087ea4"),
        (0.85, "#22e6ff"),
        (1.00, "#e6fbff"),
    ],
    N=256,
)
ION_COCOON_CMAP.set_bad(color="#000000")

ION_WAVES_CMAP = LinearSegmentedColormap.from_list(
    "ion_waves_bipolar",
    [
        (0.00, "#6d28d9"),
        (0.28, "#1e1b4b"),
        (0.50, "#02040a"),
        (0.72, "#084c61"),
        (1.00, "#38f9ff"),
    ],
    N=256,
)
ION_WAVES_CMAP.set_bad(color="#000000")

CFD_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=int,
)
CFD_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=float)

_GRID_X = (np.arange(CFD_NX) + 0.5) * (BOX_X / CFD_NX)
_GRID_Y = (np.arange(CFD_NY) + 0.5) * (BOX_Y / CFD_NY)
_GRID_XX, _GRID_YY = np.meshgrid(_GRID_X, _GRID_Y)
_GRID_PTS = np.column_stack((_GRID_XX.ravel(), _GRID_YY.ravel()))

_PHI_X = (np.arange(PHI_NX) + 0.5) * (BOX_X / PHI_NX)
_PHI_Y = (np.arange(PHI_NY) + 0.5) * (BOX_Y / PHI_NY)
_PHI_XX, _PHI_YY = np.meshgrid(_PHI_X, _PHI_Y)
_PHI_PTS = np.column_stack((_PHI_XX.ravel(), _PHI_YY.ravel()))


def compute_electric_potential_grid(
    grid_pts: np.ndarray,
    grid_shape: tuple,
    hull_ions: np.ndarray,
    hull_elecs: np.ndarray,
    hull_ion_q: np.ndarray,
    hull_elec_q: np.ndarray,
    med_pos: np.ndarray,
    med_charge: np.ndarray,
    solid_mask: np.ndarray,
    softening: float = COULOMB_SOFTENING,
) -> np.ndarray:
    """Vectorized electric potential Phi(r) with hull masking."""
    pos_list = []
    q_list = []

    if len(hull_ions) > 0:
        pos_list.append(hull_ions)
        q_list.append(hull_ion_q)
    if len(hull_elecs) > 0:
        pos_list.append(hull_elecs)
        q_list.append(-hull_elec_q)
    if len(med_pos) > 0:
        pos_list.append(med_pos)
        q_list.append(med_charge * 1.15)

    if not pos_list:
        phi = np.zeros(grid_shape, dtype=float)
        phi[solid_mask] = np.nan
        return phi

    all_pos = np.vstack(pos_list)
    all_q = np.concatenate(q_list)

    d = grid_pts[:, None, :] - all_pos[None, :, :]
    d[..., 0] -= BOX_X * np.round(d[..., 0] / BOX_X)
    d[..., 1] -= BOX_Y * np.round(d[..., 1] / BOX_Y)

    r_soft = np.sqrt(np.sum(d * d, axis=2) + softening ** 2)
    phi_flat = (1.0 / r_soft) @ all_q
    phi = np.nan_to_num(phi_flat.reshape(grid_shape), nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(~solid_mask):
        phi = phi - float(np.mean(phi[~solid_mask]))
    phi[solid_mask] = np.nan
    return phi


def _compute_real_ion_flow_field(
    pos: np.ndarray,
    vel: np.ndarray,
    hull_ions: np.ndarray,
    hull_ions_vel: np.ndarray,
    hull_ion_charges: np.ndarray,
    solid: np.ndarray,
    glow_gain: float = 1.0,
):
    """Visualization of medium and shell ion flux density."""
    gain = max(float(glow_gain), 0.001)
    med_ion_pos = pos[0::2]
    med_ion_vel = vel[0::2]
    med_ion_q = np.ones(len(med_ion_pos), dtype=float)
    if len(hull_ions) > 0:
        all_ion_pos = np.vstack([med_ion_pos, hull_ions])
        all_ion_vel = np.vstack([med_ion_vel, hull_ions_vel])
        all_ion_q = np.concatenate([med_ion_q, hull_ion_charges])
    else:
        all_ion_pos = med_ion_pos
        all_ion_vel = med_ion_vel
        all_ion_q = med_ion_q

    if len(all_ion_pos) == 0:
        return np.zeros((CFD_NY, CFD_NX), dtype=float), 0.0, 0.95

    sigma = max(COULOMB_SOFTENING * 0.95, 0.14)
    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)
    norm_kernel = 1.0 / (2.0 * np.pi * sigma * sigma)
    dx = _GRID_PTS[:, 0:1] - all_ion_pos[None, :, 0]
    dy = _GRID_PTS[:, 1:2] - all_ion_pos[None, :, 1]
    dx -= BOX_X * np.round(dx / BOX_X)
    dy -= BOX_Y * np.round(dy / BOX_Y)
    r2 = dx * dx + dy * dy
    weights = norm_kernel * np.exp(-r2 * inv_2sig2)
    q_weights = weights * all_ion_q[None, :]
    jx = q_weights @ all_ion_vel[:, 0]
    jy = q_weights @ all_ion_vel[:, 1]
    flux_mag = np.hypot(jx, jy)
    ion_speed = np.hypot(all_ion_vel[:, 0], all_ion_vel[:, 1])
    local_agitation = q_weights @ ion_speed
    raw_intensity = (0.80 * flux_mag + 0.20 * local_agitation) * gain
    scale_ref = 1.15
    field = np.tanh((raw_intensity / scale_ref) * 1.15).reshape(CFD_NY, CFD_NX)
    field[solid] = 0.0
    return np.clip(np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0), 0.0, 0.95


def _compute_ion_cocoon_field(
    pos: np.ndarray,
    vel: np.ndarray,
    hull_pos: np.ndarray,
    hull_ions: np.ndarray,
    hull_elecs: np.ndarray,
    hull_ion_q: np.ndarray,
    hull_elec_q: np.ndarray,
    charge: np.ndarray,
    solid: np.ndarray,
    glow_gain: float = 1.0,
):
    """Ion Cocoon field mode."""
    gain = max(float(glow_gain), 0.01)
    if len(hull_pos) > 0:
        d_hull = _GRID_PTS[:, None, :] - hull_pos[None, :, :]
        d_hull[..., 0] -= BOX_X * np.round(d_hull[..., 0] / BOX_X)
        d_hull[..., 1] -= BOX_Y * np.round(d_hull[..., 1] / BOX_Y)
        min_dist = np.min(np.hypot(d_hull[..., 0], d_hull[..., 1]), axis=1).reshape(CFD_NY, CFD_NX)
    else:
        min_dist = np.full((CFD_NY, CFD_NX), 1.0, dtype=float)

    med_ion_pos = pos[0::2]
    med_ion_vel = vel[0::2]
    if len(med_ion_pos) > 0:
        sigma = 0.24
        dx = _GRID_PTS[:, 0:1] - med_ion_pos[None, :, 0]
        dy = _GRID_PTS[:, 1:2] - med_ion_pos[None, :, 1]
        dx -= BOX_X * np.round(dx / BOX_X)
        dy -= BOX_Y * np.round(dy / BOX_Y)
        w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) / (2.0 * np.pi * sigma * sigma)
        jx = (w @ med_ion_vel[:, 0]).reshape(CFD_NY, CFD_NX)
        jy = (w @ med_ion_vel[:, 1]).reshape(CFD_NY, CFD_NX)
        j_mag = np.hypot(jx, jy)
    else:
        j_mag = np.zeros((CFD_NY, CFD_NX), dtype=float)

    phi_grid = compute_electric_potential_grid(
        _GRID_PTS,
        (CFD_NY, CFD_NX),
        hull_ions,
        hull_elecs,
        hull_ion_q,
        hull_elec_q,
        pos,
        charge,
        np.zeros((CFD_NY, CFD_NX), dtype=bool),
        softening=COULOMB_SOFTENING,
    )
    phi_clean = np.nan_to_num(phi_grid, nan=0.0, posinf=0.0, neginf=0.0)
    dx_cell = BOX_X / CFD_NX
    dy_cell = BOX_Y / CFD_NY
    dphi_dy, dphi_dx = np.gradient(phi_clean, dy_cell, dx_cell)
    grad_phi_mag = np.hypot(dphi_dx, dphi_dy)

    lambda_cocoon = 0.34
    alpha = 0.55
    beta = 0.085
    envelope = np.exp(-min_dist / lambda_cocoon)
    raw = envelope * (1.0 + alpha * j_mag + beta * grad_phi_mag) * gain
    field = np.tanh(raw * 0.85)
    field[solid] = 0.0
    return np.clip(np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0), 0.0, 1.0


def _compute_ion_waves_field(
    pos: np.ndarray,
    cfd_f: np.ndarray,
    solid: np.ndarray,
    glow_gain: float = 1.0,
):
    """Ion Waves field mode."""
    gain = max(float(glow_gain), 0.01)
    sigma = 0.26
    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)

    med_ion_pos = pos[0::2]
    med_elec_pos = pos[1::2]

    rho_ion = np.zeros(len(_GRID_PTS), dtype=float)
    rho_elec = np.zeros(len(_GRID_PTS), dtype=float)

    if len(med_ion_pos) > 0:
        d_i = _GRID_PTS[:, None, :] - med_ion_pos[None, :, :]
        d_i[..., 0] -= BOX_X * np.round(d_i[..., 0] / BOX_X)
        d_i[..., 1] -= BOX_Y * np.round(d_i[..., 1] / BOX_Y)
        rho_ion = np.sum(np.exp(-np.sum(d_i * d_i, axis=2) * inv_2sig2), axis=1)

    if len(med_elec_pos) > 0:
        d_e = _GRID_PTS[:, None, :] - med_elec_pos[None, :, :]
        d_e[..., 0] -= BOX_X * np.round(d_e[..., 0] / BOX_X)
        d_e[..., 1] -= BOX_Y * np.round(d_e[..., 1] / BOX_Y)
        rho_elec = np.sum(np.exp(-np.sum(d_e * d_e, axis=2) * inv_2sig2), axis=1)

    gamma = 0.95
    charge_wave = (rho_ion - gamma * rho_elec).reshape(CFD_NY, CFD_NX)

    div_u = np.zeros((CFD_NY, CFD_NX), dtype=float)
    if cfd_f is not None:
        f_clean = np.nan_to_num(cfd_f, nan=0.0, posinf=0.0, neginf=0.0)
        rho_lbm = np.maximum(np.sum(f_clean, axis=0), 1e-6)
        ux = np.clip(np.sum(f_clean * CFD_C[:, 0, None, None], axis=0) / rho_lbm, -0.35, 0.35)
        uy = np.clip(np.sum(f_clean * CFD_C[:, 1, None, None], axis=0) / rho_lbm, -0.35, 0.35)
        dux_dx = 0.5 * (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (BOX_X / CFD_NX)
        duy_dy = 0.5 * (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (BOX_Y / CFD_NY)
        div_u = dux_dx + duy_dy

    kappa = 3.2
    w_raw = np.nan_to_num((charge_wave + kappa * div_u) * gain, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(~solid):
        w_raw = w_raw - float(np.mean(w_raw[~solid]))
        lim = max(float(np.percentile(np.abs(w_raw[~solid]), 98.5)), 0.15)
    else:
        lim = 1.0

    w_norm = np.tanh(w_raw / max(lim, 1e-6))
    w_norm[solid] = 0.0
    return np.clip(np.nan_to_num(w_norm, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0), -1.0, 1.0


def _cfd_equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    usq = ux * ux + uy * uy
    out = np.empty((9, CFD_NY, CFD_NX), dtype=float)
    for k, (cx, cy) in enumerate(CFD_C):
        cu = 3.0 * (cx * ux + cy * uy)
        out[k] = CFD_W[k] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    return out


def _cfd_init(flow_vx_world: float = 0.0) -> np.ndarray:
    rho = np.ones((CFD_NY, CFD_NX), dtype=float)
    u_lat = float(np.clip(flow_vx_world * DT * (CFD_NX / BOX_X) * 0.08, -0.14, 0.14))
    ux = np.full_like(rho, u_lat)
    uy = np.zeros_like(rho)
    return _cfd_equilibrium(rho, ux, uy)


def _grid_solid_mask(hull: np.ndarray, center: np.ndarray, pts: np.ndarray, shape: tuple) -> np.ndarray:
    if len(hull) < 3:
        return np.zeros(shape, dtype=bool)
    poly = _get_unwrapped_hull_poly(hull, center)
    mask = MplPath(poly, closed=True).contains_points(pts).reshape(shape)
    for sx in (-BOX_X, BOX_X):
        for sy in (-BOX_Y, BOX_Y):
            mask |= MplPath(poly + np.array([sx, sy]), closed=True).contains_points(pts).reshape(shape)
    return mask


def _cfd_solid_mask(hull: np.ndarray, center: np.ndarray) -> np.ndarray:
    return _grid_solid_mask(hull, center, _GRID_PTS, (CFD_NY, CFD_NX))


def _phi_solid_mask(hull: np.ndarray, center: np.ndarray) -> np.ndarray:
    return _grid_solid_mask(hull, center, _PHI_PTS, (PHI_NY, PHI_NX))


def _cfd_wall_velocity_field(solid, hull_prev, hull_curr, center_prev, center_curr, dt):
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
    nearest = np.empty(len(grid_pts), dtype=int)
    chunk = 2048
    for a in range(0, len(grid_pts), chunk):
        b = min(a + chunk, len(grid_pts))
        delta = grid_rel[a:b, None, :] - rel_curr[None, :, :]
        nearest[a:b] = np.argmin(np.sum(delta * delta, axis=2), axis=1)
    lattice_velocity = vertex_velocity[nearest] * DT * (CFD_NX / BOX_X) * 0.08
    lattice_velocity = np.clip(lattice_velocity, -0.22, 0.22)
    wall_ux[iy, ix] = lattice_velocity[:, 0]
    wall_uy[iy, ix] = lattice_velocity[:, 1]
    return wall_ux, wall_uy


def _cfd_step(f, solid, tau, wall_ux, wall_uy, p_damping=0.08, flow_vx_world: float = 0.0):
    f_clean = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    raw_sum = np.sum(f_clean, axis=0)
    if not np.all(np.isfinite(raw_sum)) or np.any(raw_sum <= 1e-6):
        f_clean = _cfd_init(flow_vx_world)
        raw_sum = np.sum(f_clean, axis=0)

    rho = np.clip(raw_sum, 0.25, 3.0)
    ux = np.clip(np.sum(f_clean * CFD_C[:, 0, None, None], axis=0) / rho, -0.25, 0.25)
    uy = np.clip(np.sum(f_clean * CFD_C[:, 1, None, None], axis=0) / rho, -0.25, 0.25)

    damp = float(np.clip(p_damping, 0.0, 0.95))
    if damp > 0.0:
        rho_damped = np.where(solid, rho, 1.0 + (rho - 1.0) * (1.0 - damp))
        scale = np.clip(rho_damped / rho, 0.1, 2.0)
        f_clean = f_clean * scale[None, :, :]
        rho = rho_damped

    feq = _cfd_equilibrium(rho, ux, uy)
    eff_tau = max(float(tau), 0.55)
    post = f_clean - (f_clean - feq) / eff_tau

    rho_w = np.ones_like(rho)
    ux_w = np.where(solid, np.clip(wall_ux, -0.22, 0.22), ux)
    uy_w = np.where(solid, np.clip(wall_uy, -0.22, 0.22), uy)
    feq_wall = _cfd_equilibrium(rho_w, ux_w, uy_w)
    post[:, solid] = feq_wall[:, solid]

    streamed = np.empty_like(post)
    for k, (cx, cy) in enumerate(CFD_C):
        streamed[k] = np.roll(post[k], shift=(cy, cx), axis=(0, 1))

    if abs(flow_vx_world) > 1e-6:
        u_lat = float(np.clip(flow_vx_world * DT * (CFD_NX / BOX_X) * 0.08, -0.14, 0.14))
        feq_in = _cfd_equilibrium(
            np.ones((CFD_NY, 1)),
            np.full((CFD_NY, 1), u_lat),
            np.zeros((CFD_NY, 1)),
        )
        streamed[:, :, 0] = feq_in[:, :, 0]
        streamed[:, :, -1] = streamed[:, :, -2]

    return np.nan_to_num(streamed, nan=0.0, posinf=0.0, neginf=0.0)


def _cfd_render_field(f, solid, vis_mode, glow_gain=1.0):
    gain = max(float(glow_gain), 0.01)
    f_clean = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    rho = np.maximum(np.sum(f_clean, axis=0), 1e-6)
    ux = np.sum(f_clean * CFD_C[:, 0, None, None], axis=0) / rho
    uy = np.sum(f_clean * CFD_C[:, 1, None, None], axis=0) / rho

    fluid_mask = ~solid
    if vis_mode == "Pressure":
        data = np.nan_to_num((rho - 1.0) * gain, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(fluid_mask):
            lim = float(np.percentile(np.abs(data[fluid_mask]), 99.5))
            if not np.isfinite(lim):
                lim = 1e-5
        else:
            lim = 1e-5
        bound = max(lim, 1e-7)
        vmin, vmax = -bound, bound
    else:  # Vortices
        vort = (
            np.roll(uy, -1, axis=1)
            - np.roll(uy, 1, axis=1)
            - np.roll(ux, -1, axis=0)
            + np.roll(ux, 1, axis=0)
        ) * 0.5
        data = np.nan_to_num(np.abs(vort) * gain, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(fluid_mask):
            p997 = float(np.percentile(data[fluid_mask], 99.7))
            if not np.isfinite(p997):
                p997 = 1e-5
            vmax = max(p997, 1e-7)
        else:
            vmax = 1e-5
        vmin = 0.0

    data = np.asarray(data, dtype=float)
    data[solid] = np.nan
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), float(vmin), float(vmax)


# ----------------------------- Metallic Capsule Graphics -----------------------------

_SHIP_SCALE = HULL_LENGTH / 9.2
_U_SHIP = np.linspace(-1.0, 1.0, 96)
_S_WARPED_SHIP = 0.5 * (np.tanh(1.65 * _U_SHIP) / np.tanh(1.65) + 1.0)
SHIP_X_LOCAL = -HULL_HALF_LEN + HULL_LENGTH * _S_WARPED_SHIP
SHIP_S = np.clip((SHIP_X_LOCAL - SHIP_X_LOCAL[0]) / max(HULL_LENGTH, 1e-9), 0.0, 1.0)
SHIP_X_NORM = 2.0 * SHIP_S - 1.0


def to_ship_world(pts_local: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    """Transforms local hull coordinates to world coordinates."""
    if len(pts_local) == 0:
        return pts_local.copy()
    return rotate_2d(pts_local, angle) + center[None, :]

def evaluate_capsule_local_shell(mode, phase: float, amplitude: float, sharpness: float = 0.0):
    """Evaluates local geometry of the metallic capsule with outer margin covering surface dipoles."""
    mode_fish = _mode_blend(mode)
    y_center, _, _, _ = _evaluate_bacterium_deformation(
        SHIP_X_NORM, phase, float(amplitude), mode_fish, sharpness=sharpness
    )
    visual_margin = SURFACE_DIPOLE_OFFSET * 0.5 + 0.012
    r_prof = _bacterium_radius_profile(
        SHIP_X_NORM,
        r_max=HULL_R_MAX + visual_margin,
        cap_start=HULL_CAP_START,
    )
    r_prof[0] = 0.0
    r_prof[-1] = 0.0
    upper = y_center + r_prof
    lower = y_center - r_prof
    return y_center, upper, lower


def build_metallic_capsule_layers_world(
    x_local: np.ndarray,
    y_center_local: np.ndarray,
    upper_local: np.ndarray,
    lower_local: np.ndarray,
    center: np.ndarray,
    angle: float,
    n_layers: int = CAPSULE_LAYERS,
    power_p: float = 1.6,
):
    t_vals = np.linspace(0.0, 0.94, n_layers)
    polys_world = []
    half_r_up = upper_local - y_center_local
    half_r_lo = y_center_local - lower_local
    x_margin_scale = 1.0 + (SURFACE_DIPOLE_OFFSET * 0.5 + 0.012) / max(HULL_HALF_LEN, 1e-6)

    for t in t_vals:
        tp = t ** power_p
        scale_y = max(0.02, 1.0 - tp)
        inset = HULL_R_MAX * tp
        scale_x = max(0.04, (1.0 - inset / max(HULL_HALF_LEN, 1e-6)) * x_margin_scale)

        x_t = x_local * scale_x
        yc_t = np.interp(x_t, x_local, y_center_local)
        y_up_t = yc_t + scale_y * half_r_up
        y_lo_t = yc_t - scale_y * half_r_lo

        poly_l = np.vstack([
            np.column_stack([x_t, y_up_t]),
            np.column_stack([x_t[::-1], y_lo_t[::-1]]),
        ])
        polys_world.append(to_ship_world(poly_l, center, angle))
    return polys_world



        

# ============================================================
# PART 3: PHYSICS ENGINE (VELOCITY-VERLET, PBC & WIND TUNNEL)
# ============================================================
def initialize_medium(
    n_pairs: int,
    seed: int,
    medium_temp: float,
    flow_vx: float = 0.0,
    hull_poly: np.ndarray = None,
):
    """Естественное равновесное распределение зарядов в невозмущенной среде (без кристаллизации и без стартового удара)."""
    rng = np.random.default_rng(int(seed))
    n = max(1, int(n_pairs))
    total = 2 * n

    # 1. Стратифицированная случайная расстановка (естественный вид без больших пустот)
    nx = max(2, int(np.ceil(np.sqrt(total * 1.25 * (BOX_X / BOX_Y)))))
    ny = max(2, int(np.ceil((total * 1.25) / nx)))
    dx, dy = BOX_X / nx, BOX_Y / ny
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))

    grid_pts = np.column_stack((
        (gx.ravel() + 0.5 + rng.uniform(-0.42, 0.42, nx * ny)) * dx,
        (gy.ravel() + 0.5 + rng.uniform(-0.42, 0.42, nx * ny)) * dy,
    )) % np.array([BOX_X, BOX_Y])

    if hull_poly is not None and len(hull_poly) >= 3:
        outside = ~_point_in_polygon_mask(grid_pts, hull_poly)
        grid_pts = grid_pts[outside]

    # Случайная выборка без повторений (если точек хватает)
    replace_flag = len(grid_pts) < total
    chosen = grid_pts[rng.choice(len(grid_pts), size=total, replace=replace_flag)]

    pos = chosen.copy()
    charge = np.tile([1.0, -1.0], n)
    species = np.tile([0, 1], n)
    mass = np.tile([PARTICLE_MASS, ELECTRON_MASS], n)

    # 2. Короткая релаксация координат (убирает случайные сближения, сохраняя естественный хаос)
    for _ in range(12):
        f_rel = pairwise_medium_forces(pos, charge, species)
        # Нормируем смещение, чтобы частицы мягко встали на равновесные дистанции без разлета
        f_norm = np.linalg.norm(f_rel, axis=1, keepdims=True)
        step_dir = np.where(f_norm > 1e-9, f_rel / np.maximum(f_norm, 1e-9), 0.0)
        step_size = np.minimum(0.03, 0.15 * f_norm)
        pos = (pos + step_dir * step_size) % np.array([BOX_X, BOX_Y])

        # Не пускаем частицы внутрь контура корабля при релаксации
        if hull_poly is not None and len(hull_poly) >= 3:
            inside = _point_in_polygon_mask(pos, hull_poly)
            if np.any(inside):
                pos[inside] = rng.uniform([0.0, 0.0], [BOX_X, BOX_Y], size=(int(np.sum(inside)), 2))

    # 3. Чисто тепловые скорости Максвелла с нулевым суммарным импульсом среды
    t_init = max(0.0, float(medium_temp))
    v_std = np.sqrt(t_init / np.maximum(mass, 1e-12))[:, None]
    vel = rng.normal(0.0, 1.0, pos.shape) * v_std
    p_cm = np.sum(mass[:, None] * vel, axis=0) / np.sum(mass)
    vel -= p_cm[None, :]
    vel[:, 0] += float(flow_vx)

    return pos, vel, charge, mass, species


def run_simulation(
    mode: str,
    density: float,
    scale_pairs_by_density: bool,
    surface_density: float,
    body_mass_scale: float,
    damping_val: float,
    medium_temp: float,
    frequency: float,
    amplitude: float,
    frames: int,
    n_pairs: int,
    seed: int,
    running_charges: bool,
    env_mode: str,
    wind_speed: float,
    cfd_vis_mode: str,
    cfd_glow_gain: float,
    show_dynamic_field: bool = True,
    disable_pitch_rotation: bool = False,
    sharpness: float = 0.0,
):
    """Integrates the coupled 'medium + ship/bacterium' trajectory."""
    no_pitch = bool(disable_pitch_rotation)
    n_hull_nodes = max(0, int(round(HULL_NODES_BASE * float(surface_density))))
    damping_coeff = float(np.clip(damping_val, 0.0, 1.0))
    medium_temp_val = float(np.clip(medium_temp, 0.0, MEDIUM_TEMP_MAX))
    glow_gain_val = max(0.001, float(cfd_glow_gain))
    use_running = bool(running_charges)
    n_frames = max(10, int(frames))

    is_wind_tunnel = isinstance(env_mode, str) and ("wind" in env_mode.lower() or "продувк" in env_mode.lower())
    flow_vx = float(wind_speed) if is_wind_tunnel else 0.0

    if bool(scale_pairs_by_density):
        effective_pairs = max(4, int(round(float(n_pairs) * (float(density) / 0.2))))
        effective_pairs = min(effective_pairs, 300)
    else:
        effective_pairs = max(1, min(int(n_pairs), 300))

    heat_rng = np.random.default_rng(int(seed) + 100003)
    spawn_rng = np.random.default_rng(int(seed) + 555)

    n_internal_particles = max(0, int(round(INTERNAL_PARTICLES_BASE * float(body_mass_scale))))
    internal_norm = generate_internal_particles_local(n_internal_particles)

    shell_mass = float(n_hull_nodes) * (PARTICLE_MASS + ELECTRON_MASS) * HULL_MASS_MULTIPLIER
    internal_mass = float(n_internal_particles) * PARTICLE_MASS
    hull_mass = shell_mass + internal_mass
        # --- РАЗДЕЛЕНИЕ СТАРТОВОЙ ПОЗИЦИИ ДЛЯ ПРОДУВКИ И СВОБОДНОГО ПОЛЁТА ---
    if is_wind_tunnel:
        # В аэродинамической трубе любой режим (Рыба или Медуза) строго по центру
        init_x = HULL_INIT_X
        init_y = HULL_INIT_Y
    else:
        # В свободном полёте:
        if _mode_blend(mode) >= 0.5:
            # Рыба плывёт влево: ставим у правого края с безопасным отступом от хвоста
            margin_x = 0.25
            init_x = BOX_X - HULL_HALF_LEN - margin_x
            init_y = HULL_INIT_Y
        else:
            # Медуза остаётся строго по центру
            init_x = HULL_INIT_X
            init_y = HULL_INIT_Y

    hull_center = np.array([init_x, init_y], dtype=float)
    # --------------------------------------------------------------------


    #hull_center = np.array([HULL_INIT_X, HULL_INIT_Y], dtype=float)
    hull_vel = np.zeros(2, dtype=float)
    hull_angle = 0.0
    hull_omega = 0.0
    phase = 0.0
    surf_weight = 1.0 if n_hull_nodes > 0 else 0.0

    collision_nodes = max(n_hull_nodes, 140)
    hull_pos0, hull_ions0, _, _, _, _ = make_hull_nodes(
        mode, phase, float(amplitude), collision_nodes,
        running_charges=use_running, center_xy=hull_center, hull_angle=hull_angle,
        sharpness=float(sharpness),
    )
    prev_geom_local = rotate_2d(minimum_image(hull_pos0 - hull_center[None, :]), -hull_angle)

    pos, vel, charge, mass, species = initialize_medium(
        effective_pairs, seed, medium_temp_val, flow_vx=flow_vx, hull_poly=hull_pos0
    )

    n_trails = min(24, len(pos))
    trails_buf = [[pos[k].copy()] for k in range(n_trails)]

    cfd_enabled = cfd_vis_mode != "Off"
    is_ion_flow_mode = cfd_vis_mode == "Ion Flow"
    is_ion_cocoon_mode = cfd_vis_mode == "Ion Cocoon"
    is_ion_waves_mode = cfd_vis_mode == "Ion Waves"
    use_lbm_cfd = cfd_enabled and (cfd_vis_mode in ("Vortices", "Pressure", "Ion Waves"))

    cfd_f = _cfd_init(flow_vx) if use_lbm_cfd else None
    cfd_tau = float(np.clip(0.58 + 0.08 * (damping_coeff / max(DAMPING, 1e-9) - 1.0), 0.55, 0.95))

    trajectory = []
    steps = n_frames * SUBSTEPS_PER_FRAME
    prev_hull_cfd = hull_pos0.copy()
    prev_center_cfd = hull_center.copy()
    prev_hull_ions_frame = hull_ions0.copy() if n_hull_nodes == collision_nodes else None

    for step in range(steps):
        if step % SUBSTEPS_PER_FRAME == 0 and n_hull_nodes > 0:
            if prev_hull_ions_frame is None:
                _, prev_hull_ions_frame, _, _, _, _ = make_hull_nodes(
                    mode, phase, float(amplitude), n_hull_nodes,
                    running_charges=use_running, center_xy=hull_center, hull_angle=hull_angle,
                    sharpness=float(sharpness),
                )

        phase += 2.0 * np.pi * float(frequency) * DT

        # 1. Один расчёт геометрии на весь подшаг + сохранение локальных координат для быстрого поворота
        hull_pos, hull_ions, hull_elecs, _, run_w, len_scale = make_hull_nodes(
            mode, phase, float(amplitude), n_hull_nodes,
            running_charges=use_running, center_xy=hull_center, hull_angle=hull_angle,
            sharpness=float(sharpness),
        )
        if n_hull_nodes > 0:
            base_hull_local = rotate_2d(hull_pos - hull_center[None, :], -hull_angle)
            base_ions_local = rotate_2d(hull_ions - hull_center[None, :], -hull_angle)
            base_elecs_local = rotate_2d(hull_elecs - hull_center[None, :], -hull_angle)

        if n_hull_nodes >= collision_nodes:
            geom_hull_pos = hull_pos
            curr_geom_local = base_hull_local
        else:
            geom_hull_pos = make_hull_nodes(
                mode, phase, float(amplitude), collision_nodes,
                running_charges=False, center_xy=hull_center, hull_angle=hull_angle,
                sharpness=float(sharpness),
            )[0]
            curr_geom_local = rotate_2d(minimum_image(geom_hull_pos - hull_center[None, :]), -hull_angle)

        v_deform_local = (curr_geom_local - prev_geom_local) / max(DT, 1e-12)
        prev_geom_local = curr_geom_local
        hull_deform_vel = rotate_2d(v_deform_local, hull_angle)

        internal_world = evaluate_internal_particles_world(
            internal_norm, mode, phase, float(amplitude), hull_center, hull_angle, len_scale,
            sharpness=float(sharpness),
        )
        hull_inertia = _compute_hull_moment_of_inertia(
            geom_hull_pos, hull_center, shell_mass, internal_world, internal_mass
        )

        force = pairwise_medium_forces(pos, charge, species)
        hull_force = np.zeros(2, dtype=float)
        hull_torque = 0.0
        if n_hull_nodes > 0 and surf_weight > 0.0:
            f_surf, f_react, t_react = hull_surface_forces(
                pos, species, charge, hull_ions, hull_elecs, hull_center, surf_weight, run_w
            )
            force += f_surf
            hull_force = f_react
            hull_torque = 0.0 if no_pitch else t_react

        # B -> A -> O -> A (интегратор BAOAB)
        vel += 0.5 * (force / mass[:, None]) * DT
        pos = pos + 0.5 * vel * DT
        _apply_medium_heat_source(
            vel, mass, medium_temp_val, damping_coeff, heat_rng, DT, flow_drift_vx=flow_vx
        )
        pos = pos + 0.5 * vel * DT
                
        if len(pos) > 0:
            if is_wind_tunnel:
                wrapped_x = (pos[:, 0] < 0.0) | (pos[:, 0] >= BOX_X)
                if np.any(wrapped_x):
                    n_resp = int(np.sum(wrapped_x))
                    pos[wrapped_x, 0] = 0.02 if flow_vx >= 0.0 else (BOX_X - 0.02)
                    pos[wrapped_x, 1] = spawn_rng.uniform(0.05, BOX_Y - 0.05, size=n_resp)
                    v_th_new = spawn_rng.normal(0.0, 1.0, size=(n_resp, 2)) * _medium_target_v_rms(
                        medium_temp_val, mass[wrapped_x]
                    )
                    vel[wrapped_x, 0] = flow_vx + v_th_new[:, 0]
                    vel[wrapped_x, 1] = v_th_new[:, 1]
                    for k_idx in np.where(wrapped_x)[0]:
                        if k_idx < n_trails:
                            trails_buf[k_idx].clear()
            else:
                # Режим Free flight: гасим реактивную струю на границах до тепловой скорости
                wrapped_any = (
                    (pos[:, 0] < 0.0) | (pos[:, 0] >= BOX_X) |
                    (pos[:, 1] < 0.0) | (pos[:, 1] >= BOX_Y)
                )
                if np.any(wrapped_any):
                    n_resp = int(np.sum(wrapped_any))
                    v_th_new = spawn_rng.normal(0.0, 1.0, size=(n_resp, 2)) * _medium_target_v_rms(
                        medium_temp_val, mass[wrapped_any]
                    )
                    vel[wrapped_any] = v_th_new
                    for k_idx in np.where(wrapped_any)[0]:
                        if k_idx < n_trails:
                            trails_buf[k_idx].clear()

        pos %= np.array([BOX_X, BOX_Y])


        


        


        
        # Перемещение корпуса и переиспользование локальной геометрии без повторного вызова make_hull_nodes
        if is_wind_tunnel:
            hull_vel = np.zeros(2, dtype=float)
            hull_omega = 0.0
            if no_pitch:
                hull_angle = 0.0
        elif hull_mass > 1e-12:
            hull_vel += 0.5 * (hull_force / hull_mass) * DT
            hull_center = (hull_center + hull_vel * DT) % np.array([BOX_X, BOX_Y])
            if no_pitch:
                hull_omega = 0.0
                hull_angle = 0.0
            else:
                hull_omega += 0.5 * (hull_torque / hull_inertia) * DT
                hull_angle += hull_omega * DT

            if n_hull_nodes > 0:
                hull_pos = hull_center[None, :] + rotate_2d(base_hull_local, hull_angle)
                hull_ions = hull_center[None, :] + rotate_2d(base_ions_local, hull_angle)
                hull_elecs = hull_center[None, :] + rotate_2d(base_elecs_local, hull_angle)
            geom_hull_pos = hull_center[None, :] + rotate_2d(curr_geom_local, hull_angle)
            hull_deform_vel = rotate_2d(v_deform_local, hull_angle)

        hull_vel, hull_omega = enforce_exterior_hull_clearance(
            pos=pos,
            vel=vel,
            mass=mass,
            hull_pos=geom_hull_pos,
            hull_center=hull_center,
            hull_vel=hull_vel,
            hull_omega=hull_omega,
            hull_mass=hull_mass,
            hull_inertia=hull_inertia,
            hull_deform_vel=hull_deform_vel,
            clearance=EXTERIOR_PARTICLE_CLEARANCE,
            update_hull_momentum=(not is_wind_tunnel),
            disable_pitch_rotation=no_pitch,
        )
        if is_wind_tunnel:
            hull_vel = np.zeros(2, dtype=float)
            hull_omega = 0.0
        if no_pitch:
            hull_angle = 0.0
            hull_omega = 0.0

        # Вторая половина шага сил (используем уже повёрнутые hull_ions / hull_elecs)
        force2 = pairwise_medium_forces(pos, charge, species)
        hull_force2 = np.zeros(2, dtype=float)
        hull_torque2 = 0.0
        if n_hull_nodes > 0 and surf_weight > 0.0:
            f_surf2, f_react2, t_react2 = hull_surface_forces(
                pos, species, charge, hull_ions, hull_elecs, hull_center, surf_weight, run_w
            )
            force2 += f_surf2
            hull_force2 = f_react2
            hull_torque2 = 0.0 if no_pitch else t_react2

        vel += 0.5 * (force2 / mass[:, None]) * DT

        if is_wind_tunnel:
            hull_vel = np.zeros(2, dtype=float)
            hull_omega = 0.0
        elif hull_mass > 1e-12:
            hull_vel += 0.5 * (hull_force2 / hull_mass) * DT
            if no_pitch:
                hull_omega = 0.0
                hull_angle = 0.0
            else:
                internal_world = evaluate_internal_particles_world(
                    internal_norm, mode, phase, float(amplitude), hull_center, hull_angle, len_scale
                )
                hull_inertia2 = _compute_hull_moment_of_inertia(
                    geom_hull_pos, hull_center, shell_mass, internal_world, internal_mass
                )
                hull_omega += 0.5 * (hull_torque2 / hull_inertia2) * DT

        if use_lbm_cfd:
            curr_hull_cfd = geom_hull_pos % np.array([BOX_X, BOX_Y])
            solid_step = _cfd_solid_mask(curr_hull_cfd, hull_center)
            wall_ux, wall_uy = _cfd_wall_velocity_field(
                solid_step, prev_hull_cfd, curr_hull_cfd, prev_center_cfd, hull_center, DT
            )
            cfd_f = _cfd_step(
                cfd_f, solid_step, cfd_tau, wall_ux, wall_uy, p_damping=0.10, flow_vx_world=flow_vx
            )
            prev_hull_cfd = curr_hull_cfd
            prev_center_cfd = hull_center.copy()

        if step % SUBSTEPS_PER_FRAME == 0:
            hull_vel, hull_omega = enforce_exterior_hull_clearance(
                pos=pos,
                vel=vel,
                mass=mass,
                hull_pos=geom_hull_pos,
                hull_center=hull_center,
                hull_vel=hull_vel,
                hull_omega=hull_omega,
                hull_mass=hull_mass,
                hull_inertia=hull_inertia,
                hull_deform_vel=hull_deform_vel,
                clearance=EXTERIOR_PARTICLE_CLEARANCE,
                update_hull_momentum=(not is_wind_tunnel),
                disable_pitch_rotation=no_pitch,
            )
            if is_wind_tunnel:
                hull_vel = np.zeros(2, dtype=float)
                hull_omega = 0.0
            if no_pitch:
                hull_angle = 0.0
                hull_omega = 0.0

            internal_world = evaluate_internal_particles_world(
                internal_norm, mode, phase, float(amplitude), hull_center, hull_angle, len_scale
            )

            for k in range(n_trails):
                if len(trails_buf[k]) > 0:
                    if np.hypot(pos[k, 0] - trails_buf[k][-1][0], pos[k, 1] - trails_buf[k][-1][1]) > 0.5 * BOX_Y:
                        trails_buf[k].clear()
                trails_buf[k].append(pos[k].copy())
                if len(trails_buf[k]) > TRAIL_LEN:
                    trails_buf[k].pop(0)

            hull_ion_q, hull_elec_q = _compute_hull_ion_charge_weights(n_hull_nodes, surf_weight, run_w)

            phi_frame = None
            if show_dynamic_field:
                solid_phi = _phi_solid_mask(geom_hull_pos % np.array([BOX_X, BOX_Y]), hull_center)
                phi_frame = compute_electric_potential_grid(
                    _PHI_PTS,
                    (PHI_NY, PHI_NX),
                    hull_ions,
                    hull_elecs,
                    hull_ion_q,
                    hull_elec_q,
                    pos,
                    charge,
                    solid_phi,
                    softening=COULOMB_SOFTENING,
                )

            field_frame, vmin_f, vmax_f = None, 0.0, 1.0
            if cfd_enabled:
                solid_i = _cfd_solid_mask(geom_hull_pos % np.array([BOX_X, BOX_Y]), hull_center)
                if is_ion_flow_mode:
                    if n_hull_nodes > 0:
                        if prev_hull_ions_frame is not None and len(prev_hull_ions_frame) == len(hull_ions):
                            hull_ions_v = minimum_image(hull_ions - prev_hull_ions_frame) / max(DT, 1e-12)
                        else:
                            hull_ions_v = np.broadcast_to(hull_vel, hull_ions.shape)
                    else:
                        hull_ions_v = np.zeros((0, 2))

                    field_frame, vmin_f, vmax_f = _compute_real_ion_flow_field(
                        pos, vel, hull_ions, hull_ions_v, hull_ion_q, solid_i, glow_gain=glow_gain_val
                    )
                elif is_ion_cocoon_mode:
                    field_frame, vmin_f, vmax_f = _compute_ion_cocoon_field(
                        pos, vel, geom_hull_pos, hull_ions, hull_elecs,
                        hull_ion_q, hull_elec_q, charge, solid_i, glow_gain=glow_gain_val,
                    )
                elif is_ion_waves_mode:
                    field_frame, vmin_f, vmax_f = _compute_ion_waves_field(
                        pos, cfd_f, solid_i, glow_gain=glow_gain_val
                    )
                else:
                    field_frame, vmin_f, vmax_f = _cfd_render_field(
                        cfd_f, solid_i, cfd_vis_mode, glow_gain=glow_gain_val
                    )

            prev_hull_ions_frame = hull_ions.copy() if n_hull_nodes > 0 else None

            trajectory.append({
                "phase": phase,
                "hull_center": hull_center.copy(),
                "hull_angle": 0.0 if no_pitch else float(hull_angle),
                "hull_pos": (geom_hull_pos % np.array([BOX_X, BOX_Y])).copy(),
                "hull_ions": (hull_ions % np.array([BOX_X, BOX_Y])).copy(),
                "hull_elecs": (hull_elecs % np.array([BOX_X, BOX_Y])).copy(),
                "running_q": None if run_w is None else run_w.copy(),
                "internal": (internal_world % np.array([BOX_X, BOX_Y])).copy(),
                "pos": pos.copy(),
                "trails": [np.array(tr) for tr in trails_buf],
                "phi": phi_frame,
                "field": field_frame,
                "vmin": vmin_f,
                "vmax": vmax_f,
            })

    return trajectory


# ============================================================
# PART 4: 16:9 RENDERING & GRADIO INTERFACE
# ============================================================

def _clear_contour_set(cs):
    """Removes contour/contourf collections from the previous frame."""
    if cs is None:
        return
    try:
        for coll in getattr(cs, "collections", []):
            coll.remove()
    except Exception:
        pass
    try:
        cs.remove()
    except Exception:
        pass


def generate_simulation_video(
    env_mode: str,
    wind_speed: float,
    disable_pitch_rotation: bool,
    mode: str,
    frequency: float,
    amplitude: float,
    sharpness: float,
    running_charges: bool,
    surface_density: float,
    body_mass_scale: float,
    medium_temp: float,
    density: float,
    scale_pairs_by_density: bool,
    damping_val: float,
    show_capsule: bool,
    show_microbio_particles: bool,
    show_stars: bool,
    cfd_vis_mode: str,
    cfd_glow_gain: float,
    show_dynamic_field: bool,
    show_particles_and_trails: bool,
    frames: int,
    n_pairs: int,
):
    no_pitch = bool(disable_pitch_rotation)
    show_cap = bool(show_capsule)
    show_microbio = bool(show_microbio_particles)
    show_stars_flag = bool(show_stars)
    show_dyn_field = bool(show_dynamic_field)
    show_med = bool(show_particles_and_trails)

    trajectory = run_simulation(
        mode=mode,
        density=density,
        scale_pairs_by_density=scale_pairs_by_density,
        surface_density=surface_density,
        body_mass_scale=body_mass_scale,
        damping_val=damping_val,
        medium_temp=medium_temp,
        frequency=frequency,
        amplitude=amplitude,
        frames=frames,
        n_pairs=n_pairs,
        seed=SEED,
        running_charges=running_charges,
        env_mode=env_mode,
        wind_speed=wind_speed,
        cfd_vis_mode=cfd_vis_mode,
        cfd_glow_gain=cfd_glow_gain,
        show_dynamic_field=show_dyn_field,
        disable_pitch_rotation=no_pitch,
        sharpness=float(sharpness),
    )
    n_rendered = len(trajectory)

    cfd_enabled = cfd_vis_mode != "Off"
    is_wind = isinstance(env_mode, str) and ("wind" in env_mode.lower() or "продувк" in env_mode.lower())
    flow_vx = float(wind_speed) if is_wind else 0.0
    any_hull_visible = show_cap or show_microbio

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=80)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_xlim(0.0, BOX_X)
    ax.set_ylim(0.0, BOX_Y)
    ax.set_aspect("equal")
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#000000")
    ax.set_axis_off()

    # Background stars (optional; pure black background when off)
    star_rng = np.random.default_rng(777)
    n_stars = 90
    star_x0 = star_rng.uniform(0.0, BOX_X, size=n_stars)
    star_y0 = star_rng.uniform(0.05, BOX_Y - 0.05, size=n_stars)
    star_sizes = star_rng.choice([1.5, 3.0, 5.5, 9.5], size=n_stars, p=[0.45, 0.30, 0.18, 0.07]) * 12.0
    star_alphas = star_rng.uniform(0.35, 0.90, size=n_stars)
    star_rgba = np.ones((n_stars, 4), dtype=float)
    star_rgba[:, 3] = star_alphas
    star_scatter = ax.scatter(
        star_x0, star_y0, s=star_sizes, c=star_rgba, edgecolors="none", zorder=0, visible=show_stars_flag
    )

    # Background CFD / Ion Flow / Ion Cocoon / Ion Waves field
    cfd_im = None
    if cfd_enabled and trajectory[0]["field"] is not None:
        if cfd_vis_mode == "Ion Flow":
            active_cmap = ION_FLOW_CMAP
            alpha_val = 0.92
        elif cfd_vis_mode == "Ion Cocoon":
            active_cmap = ION_COCOON_CMAP
            alpha_val = 0.95
        elif cfd_vis_mode == "Ion Waves":
            active_cmap = ION_WAVES_CMAP
            alpha_val = 0.88
        else:
            active_cmap = CFD_CMAP
            alpha_val = 0.72

        cfd_im = ax.imshow(
            trajectory[0]["field"],
            extent=[0.0, BOX_X, 0.0, BOX_Y],
            origin="lower",
            cmap=active_cmap,
            vmin=trajectory[0]["vmin"],
            vmax=trajectory[0]["vmax"],
            alpha=alpha_val,
            interpolation="bicubic",
            aspect="auto",
            zorder=1,
        )

    contour_state = {"fill_pos": None, "fill_neg": None, "line_pos": None, "line_neg": None}

    # Metallic ship capsule (gradient layers)
    edge_rgb = np.array(mcolors.to_rgb("#ffffff"))
    core_rgb = np.array(mcolors.to_rgb("#374151"))
    t_layers = np.linspace(0.0, 1.0, CAPSULE_LAYERS)
    capsule_patches = []
    for idx_l, t_val in enumerate(t_layers):
        rgb_layer = (1.0 - t_val) * edge_rgb + t_val * core_rgb
        poly_layer = Polygon(
            np.zeros((10, 2)),
            closed=True,
            facecolor=rgb_layer,
            edgecolor="#ffffff" if idx_l == 0 else "none",
            linewidth=1.5 if idx_l == 0 else 0.0,
            alpha=0.98,
            zorder=4 + idx_l * 0.1,
            visible=show_cap,
        )
        ax.add_patch(poly_layer)
        capsule_patches.append(poly_layer)

    # Microbio physical particles assembly (body + internal neutral particles + surface dipole cores/electrons)
    microbio_hull_patch = Polygon(
        np.zeros((4, 2)),
        closed=True,
        facecolor="#2b3340",
        edgecolor="#64748b",
        linewidth=1.0,
        alpha=0.85,
        zorder=4,
        visible=show_microbio,
    )
    ax.add_patch(microbio_hull_patch)
    microbio_internal_scatter = ax.scatter(
        [], [], s=72, c="#b8c4d4", alpha=0.90, zorder=6, visible=show_microbio
    )
    microbio_hull_ions = ax.scatter(
        [], [], s=80, c="#ff9e8f", alpha=0.95, zorder=8, visible=show_microbio
    )
    microbio_hull_elecs = ax.scatter(
        [], [], s=60, c="#99d3ff", alpha=0.95, zorder=9, visible=show_microbio
    )

    # Free medium physical particles and their trails
    med_ions_scatter = ax.scatter([], [], s=44, c="#ff5555", alpha=0.90, zorder=13, visible=show_med)
    med_elecs_scatter = ax.scatter([], [], s=28, c="#3399ff", alpha=0.85, zorder=13, visible=show_med)
    trail_lines = [ax.plot([], [], linewidth=0.9, alpha=0.50, zorder=5, visible=show_med)[0] for _ in range(24)]

    def update(frame_idx: int):
        st = trajectory[frame_idx]
        center = st["hull_center"]
        angle = 0.0 if no_pitch else st["hull_angle"]
        phase_val = st["phase"]

        hull_unwrapped = _get_unwrapped_hull_poly(st["hull_pos"], center)

        if show_stars_flag:
            cur_star_x = np.mod(star_x0 + flow_vx * 0.02 * frame_idx * DT, BOX_X)
            cur_star_rgba = star_rgba.copy()
            if any_hull_visible and len(hull_unwrapped) >= 3:
                inside_stars = mask_points_inside_hull_periodic(
                    np.column_stack([cur_star_x, star_y0]), hull_unwrapped, center
                )
                cur_star_rgba[inside_stars, 3] = 0.0
            star_scatter.set_offsets(np.column_stack([cur_star_x, star_y0]))
            star_scatter.set_facecolor(cur_star_rgba)
            star_scatter.set_visible(True)
        else:
            star_scatter.set_visible(False)

        if cfd_im is not None and st["field"] is not None:
            cfd_im.set_data(st["field"])
            cfd_im.set_clim(st["vmin"], st["vmax"])

        # Dynamic electric potential isolines Phi(x, y)
        for key in ("fill_pos", "fill_neg", "line_pos", "line_neg"):
            _clear_contour_set(contour_state[key])
            contour_state[key] = None

        dynamic_contour_artists = []
        if show_dyn_field and st["phi"] is not None:
            phi = st["phi"]
            valid_phi = phi[~np.isnan(phi)]
            if len(valid_phi) > 0:
                p_scale = max(float(np.percentile(np.abs(valid_phi), 97.5)), 0.25)
                pos_levels = np.linspace(0.08 * p_scale, 0.95 * p_scale, 6)
                neg_levels = np.linspace(-0.95 * p_scale, -0.08 * p_scale, 6)

                cf_pos = ax.contourf(
                    _PHI_X,
                    _PHI_Y,
                    phi,
                    levels=pos_levels,
                    cmap="Reds",
                    alpha=0.22,
                    extend="max",
                    zorder=2,
                )
                cf_neg = ax.contourf(
                    _PHI_X,
                    _PHI_Y,
                    phi,
                    levels=neg_levels,
                    cmap="Blues_r",
                    alpha=0.22,
                    extend="min",
                    zorder=2,
                )
                cl_pos = ax.contour(
                    _PHI_X,
                    _PHI_Y,
                    phi,
                    levels=pos_levels,
                    colors="#ff5a79",
                    linewidths=0.95,
                    alpha=0.72,
                    zorder=3,
                )
                cl_neg = ax.contour(
                    _PHI_X,
                    _PHI_Y,
                    phi,
                    levels=neg_levels,
                    colors="#38bdf8",
                    linewidths=0.95,
                    alpha=0.72,
                    zorder=3,
                )
                contour_state["fill_pos"] = cf_pos
                contour_state["fill_neg"] = cf_neg
                contour_state["line_pos"] = cl_pos
                contour_state["line_neg"] = cl_neg
                for cs in (cf_pos, cf_neg, cl_pos, cl_neg):
                    dynamic_contour_artists.extend(getattr(cs, "collections", []))

        # Metallic ship capsule
        if show_cap:
            y_center_l, upper_l, lower_l = evaluate_capsule_local_shell(
                mode, phase_val, float(amplitude), sharpness=float(sharpness)
            )
            polys_w = build_metallic_capsule_layers_world(
                SHIP_X_LOCAL,
                y_center_l,
                upper_l,
                lower_l,
                center,
                angle,
                n_layers=CAPSULE_LAYERS,
                power_p=1.6,
            )
            for poly_patch, poly_pts in zip(capsule_patches, polys_w):
                poly_patch.set_xy(poly_pts)
                poly_patch.set_visible(True)
        else:
            for poly_patch in capsule_patches:
                poly_patch.set_visible(False)

        # Microbio physical particles (hull contour + internal neutral particles + shell dipole particles)
        if show_microbio:
            microbio_hull_patch.set_visible(True)
            microbio_internal_scatter.set_visible(True)
            microbio_hull_ions.set_visible(True)
            microbio_hull_elecs.set_visible(True)

            if len(hull_unwrapped) >= 3:
                microbio_hull_patch.set_xy(hull_unwrapped)

            if len(st["internal"]) > 0:
                int_uw = center[None, :] + minimum_image(st["internal"] - center[None, :])
                microbio_internal_scatter.set_offsets(int_uw)
            else:
                microbio_internal_scatter.set_offsets(np.empty((0, 2)))

            if len(st["hull_ions"]) > 0:
                ions_uw = center[None, :] + minimum_image(st["hull_ions"] - center[None, :])
                elecs_uw = center[None, :] + minimum_image(st["hull_elecs"] - center[None, :])
                microbio_hull_ions.set_offsets(ions_uw)
                microbio_hull_elecs.set_offsets(elecs_uw)
                if st["running_q"] is not None:
                    rq = st["running_q"]
                    microbio_hull_ions.set_color(["#ff3333" if v >= 0 else "#ff9e8f" for v in rq])
                    microbio_hull_elecs.set_color(["#238cff" if v <= 0 else "#99d3ff" for v in rq])
                else:
                    microbio_hull_ions.set_color("#ff9e8f")
                    microbio_hull_elecs.set_color("#99d3ff")
            else:
                microbio_hull_ions.set_offsets(np.empty((0, 2)))
                microbio_hull_elecs.set_offsets(np.empty((0, 2)))
        else:
            microbio_hull_patch.set_visible(False)
            microbio_internal_scatter.set_visible(False)
            microbio_hull_ions.set_visible(False)
            microbio_hull_elecs.set_visible(False)

        # Free medium particles and trails
        if show_med and len(st["pos"]) > 0:
            p_all = st["pos"]
            ions_pos = p_all[0::2]
            elecs_pos = p_all[1::2]

            if any_hull_visible and len(hull_unwrapped) >= 3:
                ions_in = mask_points_inside_hull_periodic(ions_pos, hull_unwrapped, center)
                elecs_in = mask_points_inside_hull_periodic(elecs_pos, hull_unwrapped, center)
                ions_vis = ions_pos[~ions_in]
                elecs_vis = elecs_pos[~elecs_in]
            else:
                ions_vis = ions_pos
                elecs_vis = elecs_pos

            med_ions_scatter.set_offsets(ions_vis if len(ions_vis) > 0 else np.empty((0, 2)))
            med_elecs_scatter.set_offsets(elecs_vis if len(elecs_vis) > 0 else np.empty((0, 2)))
            med_ions_scatter.set_visible(True)
            med_elecs_scatter.set_visible(True)

            trails = st["trails"]
            for idx_t, line in enumerate(trail_lines):
                if idx_t < len(trails) and len(trails[idx_t]) > 1:
                    tr = trails[idx_t]
                    if any_hull_visible and len(hull_unwrapped) >= 3:
                        tr_in = mask_points_inside_hull_periodic(tr, hull_unwrapped, center)
                        tr_masked = tr.copy()
                        tr_masked[tr_in] = np.nan
                    else:
                        tr_masked = tr
                    line.set_data(tr_masked[:, 0], tr_masked[:, 1])
                    line.set_color("#ff6666" if (idx_t % 2 == 0) else "#55aaff")
                    line.set_visible(True)
                else:
                    line.set_data([], [])
                    line.set_visible(False)
        else:
            med_ions_scatter.set_visible(False)
            med_elecs_scatter.set_visible(False)
            for line in trail_lines:
                line.set_visible(False)

        artists = [
            star_scatter,
            *dynamic_contour_artists,
            *capsule_patches,
            microbio_hull_patch,
            microbio_internal_scatter,
            microbio_hull_ions,
            microbio_hull_elecs,
            med_ions_scatter,
            med_elecs_scatter,
            *trail_lines,
        ]
        if cfd_im is not None:
            artists.insert(0, cfd_im)
        return tuple(artists)

    ani = FuncAnimation(fig, update, frames=n_rendered, interval=1000 / FPS, blit=False)
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.close()
    
    writer = FFMpegWriter(
        fps=FPS,
        bitrate=2400,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    ani.save(tmp_video.name, writer=writer, dpi=80)
    plt.close(fig)
    return tmp_video.name


# ============================================================
# GRADIO INTERFACE (ENGLISH, MOBILE-FRIENDLY VERTICAL SPACING)
# ============================================================
MOBILE_SPACING_CSS = """
.gradio-container,
.gradio-container .main,
.gradio-container .block,
.gradio-container .form,
.gradio-container .group,
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-panel {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container {
    display: flex !important;
    flex-direction: column !important;
    max-width: 100% !important;
    width: 100% !important;
    gap: 2.7px !important;
    padding-left: 1.3px !important;
    padding-right: 1.3px !important;
}
.control-block {
    padding: 1.3px 0.7px !important;
    margin-bottom: 2px !important;
    border-radius: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.control-row {
    gap: 2.7px !important;
    margin-top: 1.3px !important;
    margin-bottom: 1.3px !important;
    background: transparent !important;
}
.control-row > div {
    min-width: 220px !important;
    margin-bottom: 1.3px !important;
    background: transparent !important;
}
.gradio-container label.wrap,
.gradio-container .gr-check-label {
    padding-top: 5px !important;
    padding-bottom: 5px !important;
    line-height: 1.35 !important;
}
"""


with gr.Blocks(title="Microbio 16:9") as demo:

    with gr.Group(elem_classes=["control-block"]):
        gr.Markdown("**Block 1 — Flight & Flow**")
        with gr.Row(elem_classes=["control-row"]):
            env_mode_radio = gr.Radio(
                choices=["Free flight", "Wind tunnel"],
                value="Free flight",
                label="Motion mode",
            )
            wind_speed_slider = gr.Slider(
                0.0, 4.0, value=1.5, step=0.05,
                label="Flow speed",
            )
        with gr.Row(elem_classes=["control-row"]):
            disable_pitch_chk = gr.Checkbox(
                value=False,
                label="Disable pitch rotation",
            )

    with gr.Group(elem_classes=["control-block"]):
        gr.Markdown("**Block 2 — Kinematics & Hull**")
        with gr.Row(elem_classes=["control-row"]):
            mode_radio = gr.Radio(
                choices=["Fish — traveling wave", "Jellyfish — symmetric oscillation"],
                value="Fish — traveling wave",
                label="Kinematics",
            )
            frequency_slider = gr.Slider(0.01, 1.0, value=0.11, step=0.01, label="Frequency")
            amplitude_slider = gr.Slider(0.0, 3.0, value=0.5, step=0.05, label="Amplitude")
            sharpness_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Sharpness (shock pulse)")
        with gr.Row(elem_classes=["control-row"]):
            running_charges_chk = gr.Checkbox(value=True, label="Running shell charges")
            surface_density_slider = gr.Slider(0.0, 10.0, value=0.5, step=0.1, label="Shell node density")
            body_mass_slider = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Ship internal mass")

    with gr.Group(elem_classes=["control-block"]):
        gr.Markdown("**Block 3 — Medium & CFD**")
        with gr.Row(elem_classes=["control-row"]):
            medium_temp_slider = gr.Slider(0.0, MEDIUM_TEMP_MAX, value=MEDIUM_TEMP_DEFAULT, step=0.001, label="Medium temperature")
            density_slider = gr.Slider(0.2, 3.0, value=0.2, step=0.1, label="Medium density")
            damping_slider = gr.Slider(0.0, 1.0, value=DAMPING, step=0.005, label="CFD damping")
        with gr.Row(elem_classes=["control-row"]):
            scale_pairs_by_density_chk = gr.Checkbox(
                value=False,
                label="Scale particle count by medium density",
            )

    with gr.Group(elem_classes=["control-block"]):
        gr.Markdown("**Block 4 — Visualization & Fields**")
        with gr.Row(elem_classes=["control-row"]):
            show_capsule_chk = gr.Checkbox(value=True, label="Metallic ship capsule")
            show_microbio_chk = gr.Checkbox(value=False, label="Microbio physical particles")
        with gr.Row(elem_classes=["control-row"]):
            show_dyn_field_chk = gr.Checkbox(value=True, label="Dynamic field")
            show_particles_trails_chk = gr.Checkbox(value=True, label="Particles and trails")
            show_stars_chk = gr.Checkbox(value=True, label="Background stars")
        with gr.Row(elem_classes=["control-row"]):
            cfd_vis_radio = gr.Radio(
                choices=["Ion Flow", "Ion Cocoon", "Ion Waves", "Vortices", "Pressure", "Off"],
                value="Ion Flow",
                label="Background field",
            )
            cfd_glow_slider = gr.Slider(0.01, 1.0, value=0.6, step=0.01, label="Brightness")

    with gr.Group(elem_classes=["control-block"]):
        gr.Markdown("**Block 5 — Simulation Scale**")
        with gr.Row(elem_classes=["control-row"]):
            frames_slider = gr.Slider(40, 1000, value=120, step=10, label="Number of frames")
            n_pairs_slider = gr.Slider(10, 300, value=40, step=10, label="Number of particles (pairs)")

    run_btn = gr.Button("Generate Video", variant="primary")
    video_out = gr.Video(label="16:9 Visualization (.mp4)", autoplay=True)

    run_btn.click(
        generate_simulation_video,
        inputs=[
            env_mode_radio,
            wind_speed_slider,
            disable_pitch_chk,
            mode_radio,
            frequency_slider,
            amplitude_slider,
            sharpness_slider,
            running_charges_chk,
            surface_density_slider,
            body_mass_slider,
            medium_temp_slider,
            density_slider,
            scale_pairs_by_density_chk,
            damping_slider,
            show_capsule_chk,
            show_microbio_chk,
            show_stars_chk,
            cfd_vis_radio,
            cfd_glow_slider,
            show_dyn_field_chk,
            show_particles_trails_chk,
            frames_slider,
            n_pairs_slider,
        ],
        outputs=[video_out],
    )

if __name__ == "__main__":
    demo.launch(css=MOBILE_SPACING_CSS)
