
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon
from IPython.display import Video, display

# ============================================================
# WAVE RESONATOR + DELAYED CHARGE WAKE (PARTICLE VISUALIZATION)
# Tuned for rarefied-medium scaling at INFLOW_SPEED = 1.45
# ============================================================

FPS = 30
FRAMES = 1000
DT = 1.0 / FPS

# Wave frequency & reference relaxation frequency tuned for rarefied flow
F_AVG = 0.30
WAKE_FREQ_REF = 0.45

# Resonator geometry ("bacterium" + "fish" wave)
X = np.linspace(-3.0, 3.0, 60)
S = np.clip((X - X[0]) / (X[-1] - X[0]), 0.0, 1.0)
X_NORM = 2.0 * S - 1.0
DS_DX = 1.0 / (X[-1] - X[0])

CAP_START = 0.80
R_MAX = 0.55
FISH_SPATIAL_K = 1.8 * np.pi
FISH_WAVE_AMP = 0.22
CURVATURE_GAIN = 6.0

# Rarefied-medium plasma / charge-wake parameters (dimensionless)
# Expanded Debye scale relative to hull radius (0.55) and softening (0.25)
BASE_DEBYE_LENGTH = 1.85
SOFTENING = 0.25
# Scaled to keep smooth deflection during faster transit at INFLOW_SPEED = 1.45
COULOMB = 0.022
DAMPING = 0.015
PARTICLE_MASS = 0.10

# Freestream / inflow parameters
INFLOW_SPEED = 1.45
BOX_X_MIN, BOX_X_MAX = -4.2, 4.2
BOX_Y_MIN, BOX_Y_MAX = -2.7, 2.7
INFLOW_X = BOX_X_MIN + 0.1

# Elastic normal contact with moving hull (no artificial sticking)
RESTITUTION = 0.75

# Particle emission interval and random generator
EMIT_EVERY = 5
MAX_PARTICLES = 2000
rng = np.random.default_rng(42)


def resonator(phase, dphase_dt=0.0):
    """Centerline wave, vertical velocity, slopes/curvatures, and bacterium profile."""
    envelope_fish = S ** 1.35
    wave_arg = phase - FISH_SPATIAL_K * S

    y_center = FISH_WAVE_AMP * envelope_fish * np.sin(wave_arg)
    vy_center = FISH_WAVE_AMP * envelope_fish * dphase_dt * np.cos(wave_arg)

    denv_dx = 1.35 * (S ** 0.35) * DS_DX
    dy_dx = FISH_WAVE_AMP * (
        denv_dx * np.sin(wave_arg)
        - envelope_fish * FISH_SPATIAL_K * DS_DX * np.cos(wave_arg)
    )

    d2y_dx2 = FISH_WAVE_AMP * (
        -2.0 * denv_dx * FISH_SPATIAL_K * DS_DX * np.cos(wave_arg)
        - envelope_fish * (FISH_SPATIAL_K * DS_DX) ** 2 * np.sin(wave_arg)
    )

    abs_x = np.abs(X_NORM)
    cap_t = np.clip((abs_x - CAP_START) / (1.0 - CAP_START), 0.0, 1.0)
    radius_profile = np.where(
        abs_x < CAP_START,
        R_MAX,
        R_MAX * np.sqrt(np.maximum(0.0, 1.0 - cap_t ** 2)),
    )

    upper = y_center + radius_profile
    lower = y_center - radius_profile

    return upper, lower, y_center, vy_center, dy_dx, d2y_dx2


def surface_state(phase, dphase_dt=0.0):
    """
    Wall geometry, outward normals, wall velocity, and curvature-coupled charge.
    """
    upper, lower, y_center, vy_center, dy_dx, d2y_dx2 = resonator(phase, dphase_dt)

    norm_scale = np.sqrt(1.0 + dy_dx ** 2) + 1e-12
    nx_up = -dy_dx / norm_scale
    ny_up = 1.0 / norm_scale

    nx_lo = dy_dx / norm_scale
    ny_lo = -1.0 / norm_scale

    curvature = -d2y_dx2
    asymmetry = 1.0 + 0.35 * X_NORM

    q_upper = np.tanh(ny_up * curvature * CURVATURE_GAIN) * asymmetry
    q_lower = np.tanh(ny_lo * curvature * CURVATURE_GAIN) * asymmetry

    surface_pos = np.vstack([
        np.column_stack([X, upper]),
        np.column_stack([X, lower]),
    ])

    surface_q = np.concatenate([q_upper, q_lower])

    surface_normals = np.vstack([
        np.column_stack([nx_up, ny_up]),
        np.column_stack([nx_lo, ny_lo]),
    ])

    surface_vel = np.vstack([
        np.column_stack([np.zeros_like(X), vy_center]),
        np.column_stack([np.zeros_like(X), vy_center]),
    ])

    return surface_pos, surface_q, surface_normals, surface_vel, upper, lower


def get_wake_response_params(freq, inflow_speed):
    """
    Computes wake relaxation rate and effective Debye length for a rarefied medium
    taking into account wave frequency vs. wake response frequency and flow advection.
    """
    freq_ratio = freq / max(WAKE_FREQ_REF, 1e-6)
    advection_factor = 1.0 + 0.25 * (inflow_speed / 1.45)

    relax_rate = float(
        np.clip(0.32 / (1.0 + 1.4 * (freq_ratio ** 2)), 0.04, 0.32)
    )
    eff_debye = (
        BASE_DEBYE_LENGTH
        * (1.0 + 0.30 * np.tanh(freq_ratio))
        * advection_factor
    )
    return relax_rate, eff_debye


def plasma_force(particles, wake_pos, wake_q, debye_len):
    """Screened Coulomb force between lagged charge wake and particles."""
    if len(particles) == 0:
        return np.zeros((0, 2))

    positions = np.array([p["pos"] for p in particles])
    charges = np.array([p["q"] for p in particles])

    r = positions[:, None, :] - wake_pos[None, :, :]
    r2 = np.sum(r * r, axis=2) + SOFTENING ** 2
    distance = np.sqrt(r2)

    screening = np.exp(-distance / debye_len)
    kernel = screening * (
        1.0 / (distance ** 3)
        + 1.0 / (debye_len * (distance ** 2))
    )

    field = np.sum(
        wake_q[None, :, None] * kernel[:, :, None] * r,
        axis=1,
    )

    return COULOMB * charges[:, None] * field


def interact_with_hull(particle, surface_pos, surface_normals, surface_vel, dt):
    """
    Elastic normal reflection relative to the moving hull boundary + Newton's 3rd law.
    """
    pos = particle["pos"]
    vel = particle["vel"]

    diff = surface_pos - pos[None, :]
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))

    surf_pt = surface_pos[idx]
    n = surface_normals[idx]
    v_wall = surface_vel[idx]

    signed_dist = float(np.dot(pos - surf_pt, n))
    reaction_force_on_hull = np.zeros(2, dtype=float)

    if signed_dist < 0.0:
        pos = pos - (signed_dist - 1e-3) * n

        v_old = vel.copy()
        v_rel = vel - v_wall
        v_rel_n = float(np.dot(v_rel, n))
        v_rel_t = v_rel - v_rel_n * n

        if v_rel_n < 0.0:
            v_rel_n_new = -RESTITUTION * v_rel_n
        else:
            v_rel_n_new = v_rel_n

        vel = v_wall + v_rel_n_new * n + v_rel_t

        dp_particle = PARTICLE_MASS * (vel - v_old)
        reaction_force_on_hull = -dp_particle / max(dt, 1e-12)

        particle["pos"] = pos
        particle["vel"] = vel

    return reaction_force_on_hull


# ============================================================
# Simulation
# ============================================================

particles = []
history = []
phase = 0.0
previous_force = 0.0
trajectory = []

wake_pos = None
wake_q = None

for frame in range(FRAMES):
    t = frame * DT
    frequency = F_AVG
    dphase_dt = 2.0 * np.pi * frequency
    phase += dphase_dt * DT

    surface_pos, surface_q, surface_normals, surface_vel, upper, lower = surface_state(
        phase, dphase_dt
    )

    relax_rate, eff_debye = get_wake_response_params(frequency, INFLOW_SPEED)

    # Initialize and relax the charge wake (phase-lagged + downstream drift in rarefied flow)
    if wake_pos is None or wake_q is None:
        wake_pos = surface_pos.copy()
        wake_q = surface_q.copy()
    else:
        drift_x = 0.06 * INFLOW_SPEED * DT
        target_wake_pos = surface_pos.copy()
        target_wake_pos[:, 0] += drift_x
        wake_pos = wake_pos + relax_rate * (target_wake_pos - wake_pos)
        wake_q = wake_q + relax_rate * (surface_q - wake_q)

    # Sparse freestream emission
    if frame % EMIT_EVERY == 0:
        n_spawn = int(rng.integers(3, 7))
        y_positions = rng.uniform(BOX_Y_MIN + 0.25, BOX_Y_MAX - 0.25, size=n_spawn)
        q_values = rng.choice([-1.0, 1.0], size=n_spawn)

        for y_spawn, q_p in zip(y_positions, q_values):
            particles.append({
                "pos": np.array([INFLOW_X, float(y_spawn)], dtype=float),
                "vel": np.array([INFLOW_SPEED, 0.0], dtype=float),
                "q": float(q_p),
                "trail": [],
            })

    forces = plasma_force(particles, wake_pos, wake_q, eff_debye)

    total_particle_force = np.zeros(2)
    total_contact_reaction = np.zeros(2)

    for i, p in enumerate(particles):
        force = forces[i] - DAMPING * p["vel"]
        acceleration = force / PARTICLE_MASS

        p["vel"] += acceleration * DT
        p["pos"] += p["vel"] * DT

        contact_reaction = interact_with_hull(
            p,
            surface_pos,
            surface_normals,
            surface_vel,
            DT,
        )
        total_contact_reaction += contact_reaction

        p["trail"].append(p["pos"].copy())
        if len(p["trail"]) > 24:
            p["trail"].pop(0)

        total_particle_force += forces[i]

    particles = [
        p for p in particles
        if (BOX_X_MIN <= p["pos"][0] <= BOX_X_MAX)
        and (BOX_Y_MIN <= p["pos"][1] <= BOX_Y_MAX)
    ]

    if len(particles) > MAX_PARTICLES:
        particles = particles[-MAX_PARTICLES:]

    hull_force = -total_particle_force + total_contact_reaction
    previous_force = 0.85 * previous_force + 0.15 * hull_force[0]
    history.append(previous_force)

    trajectory.append({
        "phase": phase,
        "upper": upper.copy(),
        "lower": lower.copy(),
        "surface_pos": surface_pos.copy(),
        "surface_q": surface_q.copy(),
        "wake_pos": wake_pos.copy(),
        "wake_q": wake_q.copy(),
        "particles": [
            {
                "pos": p["pos"].copy(),
                "q": p["q"],
                "trail": np.array(p["trail"]),
            }
            for p in particles
        ],
        "frequency": frequency,
        "force": previous_force,
    })

history = np.array(history)
mean_force = float(np.mean(history))

# ============================================================
# Animation
# ============================================================
# ... (код расчёта до блока Animation остаётся без изменений)

# ============================================================
# Animation (telemetry text removed)
# ============================================================

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(BOX_X_MIN, BOX_X_MAX)
ax.set_ylim(BOX_Y_MIN, BOX_Y_MAX)
ax.set_aspect("equal")
ax.set_facecolor("#050b16")
fig.patch.set_facecolor("#050b16")

ax.set_title(
    "PLASMA WAVE PROPULSION",
    fontsize=13,
    color="white",
    pad=18,
)
ax.set_xlabel("X — direction of thrust")
ax.set_ylabel("Y")

body_patch = Polygon(
    np.zeros((120, 2)),
    closed=True,
    facecolor="#718096",
    edgecolor="white",
    linewidth=1.8,
    alpha=0.85,
)
ax.add_patch(body_patch)

surface_scatter = ax.scatter([], [], s=35, zorder=5)
wake_scatter = ax.scatter([], [], s=18, marker="x", alpha=0.65, zorder=5)
plasma_scatter = ax.scatter([], [], s=30, zorder=6)

trail_lines = []
for _ in range(12):
    line, = ax.plot([], [], linewidth=1.0, alpha=0.55)
    trail_lines.append(line)

ax.axhline(0, color="white", alpha=0.12, linewidth=0.8)
ax.axvline(0, color="white", alpha=0.08, linewidth=0.8)


def update(frame):
    state = trajectory[frame]
    upper = state["upper"]
    lower = state["lower"]

    polygon = np.vstack([
        np.column_stack([X, upper]),
        np.column_stack([X[::-1], lower[::-1]]),
    ])
    body_patch.set_xy(polygon)

    surface_pos = state["surface_pos"]
    surface_q = state["surface_q"]
    surface_colors = ["#ff3333" if q >= 0 else "#238cff" for q in surface_q]
    surface_scatter.set_offsets(surface_pos)
    surface_scatter.set_color(surface_colors)

    wpos = state["wake_pos"]
    wq = state["wake_q"]
    wcolors = ["#ff9999" if q >= 0 else "#77bbff" for q in wq]
    wake_scatter.set_offsets(wpos)
    wake_scatter.set_color(wcolors)

    plist = state["particles"]
    if plist:
        ppos = np.array([p["pos"] for p in plist])
        pcolors = ["#ff5555" if p["q"] >= 0 else "#3399ff" for p in plist]
        plasma_scatter.set_offsets(ppos)
        plasma_scatter.set_color(pcolors)
    else:
        plasma_scatter.set_offsets(np.empty((0, 2)))

    for i, line in enumerate(trail_lines):
        if i < len(plist) and len(plist[i]["trail"]) > 1:
            trail = plist[i]["trail"]
            line.set_data(trail[:, 0], trail[:, 1])
            line.set_color("#ff6666" if plist[i]["q"] >= 0 else "#55aaff")
        else:
            line.set_data([], [])

    ax.set_title(
    "PLASMA WAVE PROPULSION",
    
        fontsize=13,
        color="white",
    )

    return (
        body_patch,
        surface_scatter,
        wake_scatter,
        plasma_scatter,
        *trail_lines,
    )








ani = FuncAnimation(
    fig,
    update,
    frames=FRAMES,
    interval=1000 / FPS,
    blit=False,
)

try:
    writer = FFMpegWriter(fps=FPS, bitrate=1800)
    ani.save("plasma_wave_propulsion.mp4", writer=writer, dpi=120)
    output_file = "plasma_wave_propulsion.mp4"
except Exception:
    ani.save("plasma_wave_propulsion.gif", writer=PillowWriter(fps=FPS), dpi=100)
    output_file = "plasma_wave_propulsion.gif"

plt.close(fig)
print("Video generated:", output_file)

if output_file.endswith(".mp4"):
    display(Video(output_file, embed=True))
else:
    from IPython.display import Image
    display(Image(filename=output_file))
