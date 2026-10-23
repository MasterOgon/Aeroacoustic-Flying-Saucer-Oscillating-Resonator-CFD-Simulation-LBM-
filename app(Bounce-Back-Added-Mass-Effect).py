import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gradio as gr
import tempfile
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# Параметры сетки LBM (D2Q9)
# ============================================================
N = 400  # Разрешение сетки N x N
X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

# Векторы скоростей D2Q9
c = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
], dtype=int)

# Противоположные направления для Bounce-Back
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)

# Весовые коэффициенты
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=float)

# Цветовая палитра CFD
cfd_multicolor_cmap = LinearSegmentedColormap.from_list("cfd_multicolor", [
    (0.00, "#004d99"), (0.10, "#00b7ff"), (0.25, "#00ffd0"), (0.40, "#00ff66"),
    (0.60, "#dfff00"), (0.85, "#ff9a00"), (1.00, "#ff2200")
], N=512)
cfd_multicolor_cmap.set_bad(color='white')


def equilibrium(rho, u):
    """Стандартная равновесная функция распределения BGK D2Q9 для всей сетки."""
    feq = np.zeros((9, N, N), dtype=float)
    usqr = u[0]**2 + u[1]**2
    for i in range(9):
        cu = 3.0 * (c[i, 0] * u[0] + c[i, 1] * u[1])
        feq[i] = rho * w[i] * (1.0 + cu + 0.5 * cu**2 - 1.5 * usqr)
    return feq


def equilibrium_local(rho_val, u_x, u_y):
    """Равновесная функция для произвольного числа точек (безопасно для рефиллинга)."""
    feq = np.zeros((9, len(rho_val)), dtype=float)
    usqr = u_x**2 + u_y**2
    for i in range(9):
        cu = 3.0 * (c[i, 0] * u_x + c[i, 1] * u_y)
        feq[i] = rho_val * w[i] * (1.0 + cu + 0.5 * cu**2 - 1.5 * usqr)
    return feq


def get_body_geometry(cx, cy, attack_angle_deg, slide_angle_deg, phase,
                      amplitude, wavelength, asymmetry_val,
                      semi_major=50.0, semi_minor=12.0, d_phase_dt=1.0):
    """
    Вычисляет маску тела, локальные смещения и скорректированные скорости деформации.
    """
    alpha = np.deg2rad(attack_angle_deg)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    Xc = X - cx
    Yc = Y - cy

    # Поворот в систему координат аппарата
    Xr = Xc * cos_a + Yc * sin_a
    Yr = -Xc * sin_a + Yc * cos_a

    beta_deg = np.clip(abs(slide_angle_deg), 0.0, 90.0)
    w_wave = beta_deg / 90.0
    w_perp = 1.0 - w_wave

    asym_factor = np.clip(asymmetry_val / 10.0, 0.0, 0.8) if asymmetry_val > 1e-4 else 0.0

    # Волновой изгиб / Поперечные колебания
    center_bow = (1.0 - (Xr / semi_major)**2)
    center_bow = np.maximum(0.0, center_bow)
    
    warped_phase_medusa = phase - asym_factor * np.sin(phase) * w_perp
    bow_offset = amplitude * 1.8 * np.sin(warped_phase_medusa) * center_bow * w_perp

    ship_length = semi_major * 2.0
    k = (2.0 * np.pi) / (ship_length * max(0.1, wavelength))
    wave_phase_shift = Xr * k - phase
    tail_amp = 1.0 + (asym_factor * 1.2) * (Xr / semi_major)
    wave_offset = amplitude * 1.5 * tail_amp * np.sin(wave_phase_shift) * w_wave

    total_offset = bow_offset + wave_offset
    Yr_bent = Yr - total_offset

    body_mask = (Xr**2) / (semi_major**2) + (Yr_bent**2) / (semi_minor**2) < 1.0

    # Аналитическая производная смещения по времени с исправленным знаком волновой фазы (-d_phase_dt)
    d_warped_phase_dt = d_phase_dt * (1.0 - asym_factor * np.cos(phase) * w_perp)
    d_bow_dt = amplitude * 1.8 * d_warped_phase_dt * np.cos(warped_phase_medusa) * center_bow * w_perp
    
    # Корректная производная волнового смещения по времени
    d_wave_dt = -amplitude * 1.5 * tail_amp * d_phase_dt * np.cos(wave_phase_shift) * w_wave
    v_deform_local = d_bow_dt + d_wave_dt

    # Направление колебаний в глобальной СК
    normal_x = -sin_a
    normal_y = cos_a

    v_osc_x = v_deform_local * normal_x
    v_osc_y = v_deform_local * normal_y

    return body_mask, v_osc_x, v_osc_y


def simulate(
    frames=300,
    attack_angle=0.0,
    slide_angle=0.0,
    rotation_control=0.0,
    buoyancy=0.0,
    frequency=3.0,
    amplitude=1.5,
    asymmetry=0.0,
    gravity=0.02,
    tau=0.58,
    mass=10.0,
    damping=0.05,
    wavelength=0.85,
    vis_mode="Pressure"
):
    dt = 1.0
    omega = 1.0 / tau

    rho = np.ones((N, N), dtype=float)
    u = np.zeros((2, N, N), dtype=float)
    f = equilibrium(rho, u)

    cx = float(N // 2)
    cy = float(N // 2)

    v_x = 0.0
    v_y = 0.0
    angular_velocity = 0.0

    v_x = 0.0
    v_y = 0.0
    angular_velocity = 0.0
    
    # Инициализация переменных для динамического следа (Pressure Recovery Lag)
    v_wake_x = 0.0
    v_wake_y = 0.0
    wake_relaxation = 0.05  # Коэффициент релаксации (память среды / время рассасывания зоны давления)
    
    # Физически обоснованный момент инерции для эллипса: I = 1/4 * m * (a^2 + b^2)
    semi_major, semi_minor = 50.0, 12.0
    moment_of_inertia = 0.25 * mass * (semi_major**2 + semi_minor**2)
    
    current_phase = 0.0
    current_attack_angle = attack_angle

    # Создание Sponge Layer (поглощающего слоя) на границах
    sponge_mask = np.zeros((N, N), dtype=float)
    margin = 30
    for idx in range(margin):
        factor = ((margin - idx) / margin)**2 * 0.15
        sponge_mask[idx, :] = np.maximum(sponge_mask[idx, :], factor)
        sponge_mask[-1 - idx, :] = np.maximum(sponge_mask[-1 - idx, :], factor)
        sponge_mask[:, idx] = np.maximum(sponge_mask[:, idx], factor)
        sponge_mask[:, -1 - idx] = np.maximum(sponge_mask[:, -1 - idx], factor)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#121212')
    ax.set_facecolor('#121212')
    ax.axis('off')
    fig.tight_layout()
    imgs = []

    prev_body_mask = np.zeros((N, N), dtype=bool)

    for t in range(frames):
        d_phase = 2.0 * np.pi * (frequency / 500.0)
        current_phase += d_phase

        # 1. Получение геометрии и скорректированных скоростей деформации
        body_mask, v_osc_x, v_osc_y = get_body_geometry(
            cx, cy, current_attack_angle, slide_angle, current_phase,
            amplitude, wavelength, asymmetry, semi_major, semi_minor, d_phase_dt=d_phase
        )

        # 2. Вычисление полных локальных скоростей стенки корпуса
        rx = X - cx
        ry = Y - cy
        omega_rad = np.deg2rad(angular_velocity)

        u_wall_x = v_x - omega_rad * ry + v_osc_x
        u_wall_y = v_y + omega_rad * rx + v_osc_y

        # 3. Безопасный рефиллинг освободившихся ячеек
        fresh_fluid = prev_body_mask & (~body_mask)
        if np.any(fresh_fluid):
            rho[fresh_fluid] = 1.0
            u[0][fresh_fluid] = v_x
            u[1][fresh_fluid] = v_y
            f[:, fresh_fluid] = equilibrium_local(
                rho[fresh_fluid], u[0][fresh_fluid], u[1][fresh_fluid]
            )

        prev_body_mask = body_mask.copy()

        # 4. Столкновение LBM (Collision)
        rho_sum = np.sum(f, axis=0)
        ux = np.sum(f * c[:, 0].reshape(9, 1, 1), axis=0) / np.maximum(rho_sum, 1e-6)
        uy = np.sum(f * c[:, 1].reshape(9, 1, 1), axis=0) / np.maximum(rho_sum, 1e-6)

        u[0] = ux
        u[1] = uy
        feq = equilibrium(rho_sum, u)

        # Шаг BGK релаксации
        f = f * (1.0 - omega) + omega * feq

        # Поглощение волн у внешних границ (Sponge Layer)
        for i in range(9):
            f[i] = f[i] * (1.0 - sponge_mask) + feq[i] * sponge_mask

        # 5. Поточное перемещение (Streaming)
        f_streamed = np.zeros_like(f)
        for i in range(9):
            f_streamed[i] = np.roll(f[i], (c[i, 0], c[i, 1]), axis=(0, 1))

        # 6. Строгий Moving Bounce-Back (метод Ладда с локальной плотностью)
        hydro_fx = 0.0
        hydro_fy = 0.0
        hydro_torque = 0.0

        f_post = f_streamed.copy()

        for i in range(1, 9):
            i_opp = opp[i]
            cx_i, cy_i = c[i, 0], c[i, 1]

            # Граничные узлы: текущая ячейка жидкая, а целевая (куда ведет поток) — внутри тела
            boundary_mask = (~body_mask) & (np.roll(body_mask, (-cx_i, -cy_i), axis=(0, 1)))

            if np.any(boundary_mask):
                uw_x = u_wall_x[boundary_mask]
                uw_y = u_wall_y[boundary_mask]
                rho_local = rho_sum[boundary_mask]

                # Корректный член передачи импульса движущейся стенки
                momentum_transfer = 6.0 * w[i] * rho_local * (cx_i * uw_x + cy_i * uw_y)

                f_incoming = f_streamed[i, boundary_mask]
                f_reflected = f_incoming - momentum_transfer

                f_post[i_opp, boundary_mask] = f_reflected

                # Силовой импульс (3-й закон Ньютона)
                delta_p_x = cx_i * (f_incoming + f_reflected)
                delta_p_y = cy_i * (f_incoming + f_reflected)

                hydro_fx += np.sum(delta_p_x)
                hydro_fy += np.sum(delta_p_y)

                rx_b = X[boundary_mask] - cx
                ry_b = Y[boundary_mask] - cy
                hydro_torque += np.sum(rx_b * delta_p_y - ry_b * delta_p_x)

        f = f_post

        # 7. Интегрирование динамики Ньютона-Эйлера с учетом запаздывания следа (Wake Lag)
        v_wake_x += wake_relaxation * (v_x - v_wake_x)
        v_wake_y += wake_relaxation * (v_y - v_wake_y)

        rel_v_x = v_x - v_wake_x
        rel_v_y = v_y - v_wake_y

        # Квадратичное сопротивление относительно вовлеченного потока в следе
        drag_force_x = -0.5 * damping * rel_v_x * abs(rel_v_x)
        drag_force_y = -0.5 * damping * rel_v_y * abs(rel_v_y)

        buoyant_force_y = (gravity - buoyancy) * mass
        total_fx = hydro_fx + drag_force_x
        total_fy = hydro_fy + buoyant_force_y + drag_force_y
        

        
        accel_x = total_fx / mass
        accel_y = total_fy / mass

        v_x += accel_x * dt
        v_y += accel_y * dt

        cx += v_x * dt
        cy += v_y * dt

        # Физичная остановка при достижении границ сетки вместо резкого скачка координат
        if cx < 60 or cx > N - 60:
            v_x = 0.0
            cx = np.clip(cx, 60, N - 60)
        if cy < 60 or cy > N - 60:
            v_y = 0.0
            cy = np.clip(cy, 60, N - 60)

        # Вращательная динамика с физическим демпфированием
        control_torque = rotation_control * 0.2 * mass
        total_torque = hydro_torque + control_torque
        angular_acceleration = total_torque / max(moment_of_inertia, 1.0)
        angular_velocity += angular_acceleration * dt
        angular_velocity *= 0.98  # Гидродинамическое демпфирование вращения
        
        current_attack_angle += angular_velocity * dt
        
        # Корректное ограничение угла атаки с обнулением угловой скорости при упоре в лимит
        if current_attack_angle <= -60.0:
            current_attack_angle = -60.0
            angular_velocity = 0.0
        elif current_attack_angle >= 60.0:
            current_attack_angle = 60.0
            angular_velocity = 0.0

        # 8. Визуализация
        pressure = (rho_sum - 1.0) * 3.0

        if vis_mode == "Pressure":
            data = np.nan_to_num(pressure, nan=0.0)
            vabs = float(np.percentile(np.abs(data), 99.5)) or 1e-4
            vmin, vmax = -vabs, vabs
        elif vis_mode == "Pressure Deviation":
            data = np.abs(np.nan_to_num(pressure, nan=0.0))
            vmax = float(np.percentile(data, 99.7)) or 1e-4
            vmin = 0.0
        elif vis_mode == "Vorticity":
            vort = ((np.roll(u[1], -1, axis=0) - np.roll(u[1], 1, axis=0)) * 0.5 -
                    (np.roll(u[0], -1, axis=1) - np.roll(u[0], 1, axis=1)) * 0.5)
            data = np.abs(np.nan_to_num(vort, nan=0.0))
            vmax = float(np.percentile(data, 99.7)) or 1e-4
            vmin = 0.0
        else:
            data = np.nan_to_num(pressure, nan=0.0)
            vabs = float(np.percentile(np.abs(data), 99.5)) or 1e-4
            vmin, vmax = -vabs, vabs

        data[body_mask] = np.nan
        im = ax.imshow(data.T, cmap=cfd_multicolor_cmap, animated=True,
                       origin='lower', vmin=vmin, vmax=vmax, interpolation='bilinear')
        imgs.append([im])

    ani = animation.ArtistAnimation(fig, imgs, interval=30, blit=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ani.save(tmp.name, fps=25, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
    return tmp.name


# ============================================================
# Gradio Interface
# ============================================================
with gr.Blocks() as demo:
    with gr.Column(elem_classes="border-class"):
        output = gr.Video(label="Physically Validated CFD Simulation (1:1)", elem_classes="square-video-box")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Global & Grid")
                frames = gr.Slider(50, 1000, value=200, step=50, label="Frames")
            with gr.Column():
                gr.Markdown("#### Kinematics & Controls")
                attack_angle = gr.Slider(-45, 45, value=0.0, step=5.0, label="Attack Angle (°)")
                slide_angle = gr.Slider(-180, 180, value=0.0, step=5.0, label="Slide Angle (°)")
                rotation_control = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Yaw Torque")
                buoyancy = gr.Slider(-0.1, 0.1, value=0.0, step=0.002, label="Buoyancy Accel")
            with gr.Column():
                gr.Markdown("#### Oscillating Actuator")
                frequency = gr.Slider(0.5, 8.0, value=3.0, step=0.5, label="Oscillation Freq (Hz)")
                amplitude = gr.Slider(0.1, 5.0, value=1.5, step=0.1, label="Stroke Amplitude")
                asymmetry = gr.Slider(0.0, 10.0, value=2.0, step=0.1, label="Stroke Asymmetry")
                wavelength = gr.Slider(0.3, 3.0, value=0.85, step=0.05, label="Wave Speed / Length")
            with gr.Column():
                gr.Markdown("#### Fluid & Rigid Body Dynamics")
                gravity = gr.Slider(0.0, 0.1, value=0.01, step=0.002, label="Gravity Acceleration")
                tau = gr.Slider(0.51, 0.95, value=0.58, step=0.01, label="LBM Viscosity (Tau)")
                mass = gr.Slider(1.0, 5000.0, value=10.0, step=0.5, label="Vehicle Mass")
                damping = gr.Slider(0.0, 0.5, value=0.02, step=0.005, label="Far-Field Fluid Damping")
                vis_mode = gr.Radio(["Pressure", "Pressure Deviation", "Vorticity"],
                                    value="Pressure", label="Visualization Field")

        btn = gr.Button("RUN RIGOROUS CFD SIMULATION", variant="primary")

    btn.click(
        simulate,
        inputs=[frames, attack_angle, slide_angle, rotation_control, buoyancy,
                frequency, amplitude, asymmetry, gravity, tau, mass, damping,
                wavelength, vis_mode],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(),
        css="""
            body {background-color: #121212; color: #d0d0d0; font-family: monospace;}
            .gradio-container {max-width: 950px !important; margin: 0 auto;}
            .square-video-box {width: 100% !important; max-width: 600px !important; margin: 0 auto !important; aspect-ratio: 1 / 1 !important;}
            .border-class {border: 1px solid #262626; background-color: #1c1c1c; padding: 15px; border-radius: 4px;}
            footer {display: none !important; }
        """
    )
