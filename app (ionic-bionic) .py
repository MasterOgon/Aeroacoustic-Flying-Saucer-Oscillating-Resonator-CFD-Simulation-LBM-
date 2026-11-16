# ==========================================
# КОСМЕТИЧЕСКИ ОБНОВЛЕННЫЙ ФАЙЛ 14
# ==========================================
import io
import tempfile
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 1. D2Q9 LBM LATTICE CONSTANTS
# ==========================================
NX, NY = 280, 110
w = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float32,
)
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
spec_y = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=np.int32)
X_GRID, Y_GRID = np.meshgrid(
    np.arange(NX, dtype=np.float32),
    np.arange(NY, dtype=np.float32),
    indexing="ij",
)

# Продвинутая цветовая палитра из файла «бионик» для давления/вихрей CFD
cfd_multicolor_cmap = LinearSegmentedColormap.from_list("cfd_multicolor",
    [(0.00, "#004d99"), (0.10, "#00b7ff"), (0.25, "#00ffd0"), (0.40, "#00ff66"),
     (0.60, "#dfff00"), (0.85, "#ff9a00"), (1.00, "#ff2200")], N=512)
cfd_multicolor_cmap.set_bad(color='white')


# ==========================================
# 2. 5-REGIME ENVIRONMENT MODEL (EARTH TO ISM)
# ==========================================
def get_mission_environment(stage_pct: float, craft_speed_kms: float):
  if stage_pct <= 20.0:
    t = stage_pct / 20.0
    regime = f"Dense Atmosphere ({t*40:.0f} km) — Vortices + EGD Ion Wind"
    slip_factor = 1.0 - 0.15 * t
    ehd_coupling = 0.025 * (1.0 + 0.8 * t)
    ion_coupling = 0.0
    debye_radius = 2.5
    lambda_d_phys = "10⁻⁷ m (EGD layer)"
    f_pi_hz = 1e6
    f_cutoff_lbm = 1.0
    density_factor = 1.0 * (0.1**t)
    tau = 0.55 + 0.08 * t

  elif stage_pct <= 40.0:
    t = (stage_pct - 20.0) / 20.0
    regime = f"2. Mesosphere ({40 + t*60:.0f} km) — EGD Peak + Slip"
    slip_factor = 0.85 - 0.75 * t
    ehd_coupling = 0.045 * (1.0 - 0.7 * t)
    ion_coupling = 0.015 * t
    debye_radius = 2.5 + 3.5 * t
    lambda_d_phys = f"{0.01 + 0.04*t:.2f} m"
    f_pi_hz = 1e5
    f_cutoff_lbm = 0.5
    density_factor = 0.1 * (0.05**t)
    tau = 0.63 + 0.17 * t

  elif stage_pct <= 65.0:
    t = (stage_pct - 40.0) / 25.0
    regime = f"3. Ionosphere ({100 + t*700:.0f} km) — Debye Fin"
    slip_factor = 0.10 * (1.0 - t)
    ehd_coupling = 0.013 * (1.0 - t)
    ion_coupling = 0.015 + 0.035 * t
    debye_radius = 6.0 + 6.0 * t
    lambda_d_phys = f"{0.05 + 0.95*t:.2f} m (with potential)"
    f_pi_hz = 30000.0
    f_cutoff_lbm = 0.095
    density_factor = 0.005 * (0.2**t)
    tau = 0.80 + 0.10 * t

  elif stage_pct <= 85.0:
    t = (stage_pct - 65.0) / 20.0
    regime = "4. Interplanetary Plasma (Solar Wind, >2nd Cosmic)"
    slip_factor = 0.0
    ehd_coupling = 0.0
    ion_coupling = 0.055
    debye_radius = 12.0 + 10.0 * t
    lambda_d_phys = "~10 m (Virtual Cocoon)"
    f_pi_hz = 470.0
    f_cutoff_lbm = 0.055
    density_factor = 0.001 * (0.15**t)
    tau = 0.92

  else:
    t = (stage_pct - 85.0) / 15.0
    regime = "5. Interstellar Medium (ISM — Ultra-low Density)"
    slip_factor = 0.0
    ehd_coupling = 0.0
    ion_coupling = 0.060
    debye_radius = 22.0 + 6.0 * t
    lambda_d_phys = "~15 - 20 m"
    f_pi_hz = 50.0
    f_cutoff_lbm = 0.028
    density_factor = 0.00015 * (0.3**t)
    tau = 0.95

  u_lbm_inflow = np.clip(craft_speed_kms / 160.0, 0.0, 0.15)
  if 65.0 < stage_pct <= 85.0:
    u_lbm_inflow = max(u_lbm_inflow, 0.06)

  return {
      "regime": regime,
      "slip_factor": slip_factor,
      "ehd_coupling": ehd_coupling,
      "ion_coupling": ion_coupling,
      "debye_radius": debye_radius,
      "lambda_d_phys": lambda_d_phys,
      "f_pi_hz": f_pi_hz,
      "f_cutoff_lbm": f_cutoff_lbm,
      "density_factor": density_factor,
      "tau": tau,
      "u_lbm_inflow": u_lbm_inflow,
  }


# ==========================================
# 3. "BACTERIUM" GEOMETRY
# ==========================================
def get_resonator_state(
    t: int,
    freq: float,
    amp: float,
    debye_radius: float,
    voltage_gain: float,
    mode_fish: float,
):
  x0 = 55.0
  length = 95.0
  y0 = NY / 2.0

  s = np.clip((X_GRID - x0) / length, 0.0, 1.0)
  x_norm = 2.0 * s - 1.0
  in_body_x = (X_GRID >= x0) & (X_GRID <= x0 + length)

  raw_phase = 2.0 * np.pi * freq * t
  asym_strength = (1.0 - mode_fish) * 0.45
  asym_phase = raw_phase + asym_strength * np.sin(raw_phase)
  dphase_dt = 2.0 * np.pi * freq * (1.0 + asym_strength * np.cos(raw_phase))

  center_bow = np.maximum(0.0, 1.0 - x_norm**2)
  bow_amp = amp * 1.65
  bow_offset = bow_amp * np.sin(asym_phase) * center_bow * (1.0 - mode_fish)

  spatial_k = 1.8 * np.pi
  envelope_fish = s**1.35
  wave_amp = amp * 1.15
  wave_offset = (
      wave_amp
      * envelope_fish
      * np.sin(asym_phase - spatial_k * s)
      * mode_fish
  )

  total_offset = bow_offset + wave_offset
  y_center = y0 + total_offset

  d_bow_dt = (
      bow_amp
      * dphase_dt
      * np.cos(asym_phase)
      * center_bow
      * (1.0 - mode_fish)
  )
  d_wave_dt = (
      wave_amp
      * envelope_fish
      * dphase_dt
      * np.cos(asym_phase - spatial_k * s)
      * mode_fish
  )
  vy_axis = d_bow_dt + d_wave_dt

  ds_dx = 1.0 / length
  d_center_bow_dx = -2.0 * x_norm * ds_dx * 2.0
  dy_dx_bow = (
      bow_amp * np.sin(asym_phase) * d_center_bow_dx * (1.0 - mode_fish)
  )

  denv_dx = 1.35 * (s**0.35) * ds_dx
  dy_dx_wave = (
      wave_amp
      * (
          denv_dx * np.sin(asym_phase - spatial_k * s)
          - envelope_fish * spatial_k * ds_dx * np.cos(asym_phase - spatial_k * s)
      )
      * mode_fish
  )
  dy_dx = dy_dx_bow + dy_dx_wave

  d2y_dx2_bow = (
      bow_amp * np.sin(asym_phase) * (-4.0 * ds_dx**2) * (1.0 - mode_fish)
  )
  d2y_dx2_wave = (
      wave_amp
      * (
          -2.0 * denv_dx * spatial_k * ds_dx * np.cos(asym_phase - spatial_k * s)
          - envelope_fish * (spatial_k * ds_dx) ** 2 * np.sin(asym_phase - spatial_k * s)
      )
      * mode_fish
  )
  d2y_dx2 = d2y_dx2_bow + d2y_dx2_wave

  dvy_dx = (
      bow_amp
      * dphase_dt
      * np.cos(asym_phase)
      * d_center_bow_dx
      * (1.0 - mode_fish)
      + wave_amp
      * (
          denv_dx * dphase_dt * np.cos(asym_phase - spatial_k * s)
          + envelope_fish
          * dphase_dt
          * spatial_k
          * ds_dx
          * np.sin(asym_phase - spatial_k * s)
      )
      * mode_fish
  )
  ux_wall = -(Y_GRID - y_center) * dvy_dx
  vy_wall = vy_axis

  abs_x = np.abs(x_norm)
  cap_start = 0.80
  r_max = 7.0

  radius_profile = np.where(
      abs_x < cap_start,
      r_max,
      r_max
      * np.sqrt(
          np.maximum(0.0, 1.0 - ((abs_x - cap_start) / (1.0 - cap_start)) ** 2)
      ),
  )

  dx_outside = np.maximum(0.0, np.maximum(x0 - X_GRID, X_GRID - (x0 + length)))
  dy_local = np.abs(Y_GRID - y_center)

  radial_dist = np.where(
      in_body_x, dy_local - radius_profile, np.sqrt(dx_outside**2 + dy_local**2)
  )
  solid_mask = in_body_x & (radial_dist <= 0.0)

  ny_surf = np.sign(Y_GRID - y_center)
  nx_surf = -ny_surf * dy_dx
  norm_len = np.sqrt(nx_surf**2 + ny_surf**2) + 1e-6
  nx_surf /= norm_len
  ny_surf /= norm_len

  curvature = -d2y_dx2
  curvature_polarity = np.tanh(ny_surf * curvature * 950.0)
    
  phi = np.zeros((NX, NY), dtype=np.float32)
  if debye_radius > 0.2 and voltage_gain > 0.0:
    effective_debye_radius = debye_radius * (0.65 + 0.55 * np.sqrt(voltage_gain))
    
    # Считаем расстояние не по сложной капсуле, а от центрального эллипса/тела,
    # чтобы внешние изолинии стремились к правильной гладкой форме (окружностям/эллипсам)
    dx_field = np.maximum(0.0, np.maximum(x0 - X_GRID, X_GRID - (x0 + length)))
    dy_field = np.abs(Y_GRID - y0)
    smooth_external_dist = np.sqrt(dx_field**2 + dy_field**2)

    phi = np.where(
        radial_dist > 0.0,
        voltage_gain
        * curvature_polarity
        * np.exp(-smooth_external_dist / effective_debye_radius),
        0.0,
    )

  

  c_wave_lbm = (freq * length) / max(0.5, mode_fish * 0.9 + 0.1)

  return (
      solid_mask,
      radial_dist,
      phi,
      ux_wall,
      vy_wall,
      nx_surf,
      ny_surf,
      c_wave_lbm,
  )


# ==========================================
# 4. MAIN LBM SIMULATION & ANIMATION LOOP
# ==========================================
def run_simulation(
    stage_pct,
    craft_speed_kms,
    freq,
    amp,
    voltage_gain,
    mode_fish,
    vis_mode,
    show_streamlines,  # Новый параметр для опционального отображения линий потока
):
  env = get_mission_environment(stage_pct, craft_speed_kms)
  tau = env["tau"]
  omega = 1.0 / tau
  slip_factor = env["slip_factor"]
  ehd_coupling = env["ehd_coupling"]
  ion_coupling = env["ion_coupling"]
  debye_radius = env["debye_radius"]
  u_inflow = env["u_lbm_inflow"]

  freq_efficiency = 1.0
  if ion_coupling > 0.0 and freq > env["f_cutoff_lbm"]:
    freq_efficiency = float(
        np.exp(-(((freq - env["f_cutoff_lbm"]) / 0.025) ** 2))
    )

  rho = np.ones((NX, NY), dtype=np.float32)
  ux = np.full((NX, NY), u_inflow, dtype=np.float32)
  uy = np.zeros((NX, NY), dtype=np.float32)

  def equilibrium(r, u, v):
    usqr = u**2 + v**2
    feq = np.zeros((9, NX, NY), dtype=np.float32)
    for i in range(9):
      cu = cx[i] * u + cy[i] * v
      feq[i] = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr)
    return feq

  f = equilibrium(rho, ux, uy)
  steps = 140
  skip_frames = 3
  f_hydro_ehd_hist = []
  f_debye_hist = []
  f_total_hist = []
  pil_frames = []
  prev_solid_mask = np.zeros((NX, NY), dtype=bool)
  solid_mask = np.zeros((NX, NY), dtype=bool)
  phi = np.zeros((NX, NY), dtype=np.float32)
  c_wave_lbm = 1.0

  for t in range(steps):
    (
        solid_mask,
        signed_dist,
        phi,
        ux_wall,
        vy_wall,
        nx_surf,
        ny_surf,
        c_wave_lbm,
    ) = get_resonator_state(
        t, freq, amp, debye_radius, voltage_gain, mode_fish
    )

    if t > 0:
      uncovered = prev_solid_mask & (~solid_mask)
      if np.any(uncovered):
        rho_neighbors = 0.25 * (
            np.roll(rho, 1, axis=0)
            + np.roll(rho, -1, axis=0)
            + np.roll(rho, 1, axis=1)
            + np.roll(rho, -1, axis=1)
        )
        rho[uncovered] = rho_neighbors[uncovered] * (1.0 - 0.02 * slip_factor)
        ux[uncovered] = ux_wall[uncovered] * slip_factor + u_inflow * (
            1.0 - slip_factor
        )
        uy[uncovered] = vy_wall[uncovered] * slip_factor
        f[:, uncovered] = equilibrium(rho, ux, uy)[:, uncovered]

    prev_solid_mask = solid_mask.copy()

    rho = np.sum(f, axis=0)
    ux = np.sum(f * cx[:, None, None], axis=0) / (rho + 1e-12)
    uy = np.sum(f * cy[:, None, None], axis=0) / (rho + 1e-12)

    fx_em_reaction = 0.0
    if voltage_gain > 0.0:
      Ex, Ey = np.gradient(-phi)
      total_em_coupling = ehd_coupling + ion_coupling * freq_efficiency
      if total_em_coupling > 0.0:
        fx_em = total_em_coupling * rho * Ex
        fy_em = total_em_coupling * rho * Ey
        ux += fx_em / (rho + 1e-12)
        uy += fy_em / (rho + 1e-12)
        fx_em_reaction = -float(np.sum(fx_em)) * 25.0

    feq = equilibrium(rho, ux, uy)
    f = f - omega * (f - feq)

    mom_x_before = np.sum(f[:, solid_mask] * cx[:, None], axis=0)
    mom_y_before = np.sum(f[:, solid_mask] * cy[:, None], axis=0)

    f_bb = f[opp]
    f_sp = f[spec_y]
    wall_momentum = np.zeros_like(f)
    for i in range(9):
      cu_wall = cx[i] * ux_wall + cy[i] * vy_wall
      wall_momentum[i] = 6.0 * w[i] * rho * cu_wall * slip_factor
    f_boundary = slip_factor * (f_bb + wall_momentum) + (1.0 - slip_factor) * f_sp
    f[:, solid_mask] = f_boundary[:, solid_mask]

    mom_x_after = ux_wall[solid_mask] * rho[solid_mask]
    mom_y_after = vy_wall[solid_mask] * rho[solid_mask]

    Fx_local = -(mom_x_after - mom_x_before)
    Fy_local = -(mom_y_after - mom_y_before)
    fx_momentum = float(np.sum(Fx_local)) * 12.0
    fx_hydro = fx_momentum

    for i in range(9):
      f[i] = np.roll(f[i], shift=(cx[i], cy[i]), axis=(0, 1))

    ux[0, :] = u_inflow
    uy[0, :] = 0.0
    rho[0, :] = 1.0
    f[:, 0, :] = equilibrium(rho[0:1, :], ux[0:1, :], uy[0:1, :])[:, 0, :]
    f[:, -1, :] = f[:, -2, :]

    w_plasma = ion_coupling / (ehd_coupling + ion_coupling + 1e-9)
    fx_ehd_part = fx_em_reaction * (1.0 - w_plasma)
    fx_plasma_part = fx_em_reaction * w_plasma
    f_hydro_ehd_hist.append(fx_hydro + fx_ehd_part)
    f_debye_hist.append(fx_plasma_part)
    f_total_hist.append(fx_hydro + fx_em_reaction)

    sync_ratio = u_inflow / max(0.01, c_wave_lbm * 0.04)
    sync_warning = (
        "⚠️ SYNCHRONISM LIMIT (U ≥ c_wave)"
        
      if sync_ratio > 1.0
        else f"Sync U/c_wave: {sync_ratio:.2f}"
    )

      
    if t % skip_frames == 0:
      fig, (ax1, ax2) = plt.subplots(
          2, 1, figsize=(9.5, 6.0), gridspec_kw={"height_ratios": [2.2, 1]}
      )
      plt.style.use("dark_background")

      # 1. Давление: многоцветный градиент и резкость
      if vis_mode == "Pressure Inversion (ΔP — Cavities & Collapse)":
        dp = (rho - 1.0) * 3.0
        im = ax1.imshow(
            dp.T,
            origin="lower",
            cmap=cfd_multicolor_cmap,
            vmin=-0.06,
            vmax=0.06,
            aspect="auto",
            alpha=0.9,
            interpolation='bilinear',
        )
        cbar_label = "Pressure Deviation ΔP (Blue=Cavity, Red=Collapse)"
      
      # 2. Вихри: синий фон (начало палитры) + резкие горячие цвета (как в «бионик»)
      elif vis_mode == "Vorticity + Φ":
        duy_dx = np.gradient(uy, axis=0)
        dux_dy = np.gradient(ux, axis=1)
        vorticity = duy_dx - dux_dy
        vort_data = np.abs(np.nan_to_num(vorticity, nan=0.0))
        im = ax1.imshow(
            vort_data.T,
            origin="lower",
            cmap=cfd_multicolor_cmap,
            vmin=0.0,
            vmax=0.04,
            aspect="auto",
            alpha=0.9,
            interpolation='bilinear',
        )
        cbar_label = "Medium / Ion Cloud Vorticity"
      
      # 3. Поток: АБСОЛЮТНО НЕ ТРОГАЕМ (оригинальный вид из файла 14)
      else:
        speed = np.sqrt(ux**2 + uy**2)
        im = ax1.imshow(
            speed.T,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=0.20,
            aspect="auto",
            alpha=0.9,
        )
        cbar_label = "Flow / Ion Speed |U|"
      
    
      

    
      # Опциональное отображение линий набегающего потока (включается кнопкой)
      if show_streamlines and vis_mode != "Pressure Inversion (ΔP — Cavities & Collapse)":
        try:
          skip_stream = 4
          Y_stream, X_stream = np.mgrid[
              0 : NY : skip_stream, 0 : NX : skip_stream
          ]
          U_sub = ux[::skip_stream, ::skip_stream].T
          V_sub = uy[::skip_stream, ::skip_stream].T
          ax1.streamplot(
              X_stream,
              Y_stream,
              U_sub,
              V_sub,
              color="white",
              linewidth=0.6,
              density=1.2,
              arrowsize=0.8,
          )
        except Exception:
          pass

      if debye_radius > 0.2 and voltage_gain > 0.0:
        if float(np.max(np.abs(phi))) > 0.12:
          ax1.contour(
              phi.T,
              levels=[0.12, 0.55],
              colors=["#ff9f43"],
              linewidths=1.2,
              linestyles="--",
          )
          ax1.contour(
              phi.T,
              levels=[-0.55, -0.12],
              colors=["#00d2d3"],
              linewidths=1.4,
              linestyles="-",
          )

      ax1.contour(solid_mask.T, levels=[0.5], colors="white", linewidths=1.8)
      ax1.set_title(
          f"{env['regime']} | Step: {t}/{steps}\nSpeed:"
          f" {craft_speed_kms:.1f} km/s | λ_D: {env['lambda_d_phys']} | Freq"
          f" Eff: {freq_efficiency*100:.0f}% | {sync_warning}",
          fontsize=9.0,
          color="white",
          pad=6,
      )
      ax1.set_ylabel("Y (grid)")

      t_axis = np.arange(len(f_total_hist))
      ax2.plot(
          t_axis,
          f_hydro_ehd_hist,
          color="#ff6b6b",
          alpha=0.85,
          linewidth=1.2,
          label="Gas / EGD Thrust",
      )
      ax2.plot(
          t_axis,
          f_debye_hist,
          color="#00d2d3",
          alpha=0.9,
          linewidth=1.2,
          label="Debye Thrust (Plasma)",
      )
      ax2.plot(
          t_axis,
          f_total_hist,
          color="#feca57",
          linewidth=1.6,
          label="Total Thrust",
      )
      ax2.axhline(0, color="gray", linewidth=0.8)
      ax2.set_xlim(0, steps)
      ax2.set_ylim(
          -2.0,
          max(
              5.0, np.max(f_total_hist) * 1.2 if len(f_total_hist) > 0 else 3.0
          ),
      )
      mean_norm = np.mean(f_total_hist[40:]) if len(f_total_hist) > 40 else 0.0
      real_thrust_index = mean_norm * env["density_factor"]
      ax2.set_title(
          f"Norm Thrust: {mean_norm:.3f} | Real Scale (w/ ρ):"
          f" {real_thrust_index:.2e} N (rel.)",
          fontsize=8.5,
      )
      ax2.set_xlabel("SimulationSteps", fontsize=8.5)
      ax2.set_ylabel("Thrust Fx", fontsize=8.5)
      ax2.legend(loc="upper right", fontsize=7.0, ncol=3)
      ax2.grid(alpha=0.2)

      plt.tight_layout()
      buf = io.BytesIO()
      plt.savefig(buf, format="png", dpi=110)
      plt.close(fig)
      buf.seek(0)
      pil_frames.append(Image.open(buf).copy())

  gif_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
  pil_frames[0].save(
      gif_file.name,
      format="GIF",
      save_all=True,
      append_images=pil_frames[1:],
      duration=65,
      loop=0,
  )
  return gif_file.name


# ==========================================
# 5. GRADIO WEB INTERFACE
# ==========================================
with gr.Blocks(title="Multi-Regime Resonator: Earth to Interstellar Space") as demo:
  gr.Markdown(
      """
    # 🌌 Multi-Regime Wave Resonator: Earth to Interstellar Space (D2Q9 LBM Animation)
    - 0–20% (Troposphere): Charge creates an EGD ion wind, boosting vortices and cavity collapse.
    - 40–65% (Ionosphere, ~7.8 km/s): Neutrals slip; Debye sheath forms around concave areas (−).
    - 65–100% (2nd Cosmic >11.2 km/s → ISM): Debye cocoon expands across the domain, with strict low-frequency ion cutoffs ($f_{pi}$).

    **Kinematics:**
    Smooth blend of jellyfish arc bend (mode=0) and fish traveling wave (mode=1). Charges are formed from local compression/extension of the wall in both modes.
    """
  )
  with gr.Row():
    with gr.Column(scale=1):
      stage_slider = gr.Slider(
          0,
          100,
          value=50,
          step=2,
          label="Mission Trajectory (0=Earth, 50=Ionosphere, 75=Interplanetary, 95=Interstellar)",
      )
      speed_slider = gr.Slider(
          0.0,
          20.0,
          value=7.8,
          step=0.5,
          label="Craft Speed (km/s) [7.8 = 1st Cosmic, 11.2 = 2nd Cosmic]",
      )
      voltage_slider = gr.Slider(
          0.0,
          2.5,
          value=1.4,
          step=0.1,
          label="Curvature Potential (EGD Discharge / Debye Cloud)",
      )
      freq_slider = gr.Slider(
          0.01,
          0.12,
          value=0.04,
          step=0.005,
          label="Oscillation Frequency (ISM requires min < 0.03!)",
      )
      amp_slider = gr.Slider(
          0.5, 12.0, value=1.5, step=0.25, label="Tail Bending Amplitude (cells)"
      )
      mode_slider = gr.Slider(
          0.0,
          1.0,
          value=0.85,
          step=0.05,
          label="Kinematics: 0 = Jellyfish (arc bend), 1 = Fish (wave)",
      )
      vis_radio = gr.Radio(
          [
              "Pressure Inversion (ΔP — Cavities & Collapse)",
              "Vorticity + Φ",
              "Flow / Ion Speed |U|",
          ],
          value="Pressure Inversion (ΔP — Cavities & Collapse)",
          label="Visualization Mode",
      )
      # Добавлена кнопка (чекбокс) для опционального отображения линий потока
      streamlines_chk = gr.Checkbox(
          value=True,
          label="Отображать линии набегающего потока (Streamlines)"
      )
      run_btn = gr.Button("🚀 Generate Simulation Animation", variant="primary")
    with gr.Column(scale=2):
      output_img = gr.Image(
          label="Live CFD Field Animation & Environment Telemetry (GIF)"
      )

  inputs = [
      stage_slider,
      speed_slider,
      freq_slider,
      amp_slider,
      voltage_slider,
      mode_slider,
      vis_radio,
      streamlines_chk,
  ]
  run_btn.click(fn=run_simulation, inputs=inputs, outputs=output_img)
  demo.load(fn=run_simulation, inputs=inputs, outputs=output_img)

if __name__ == "__main__":
  demo.launch()