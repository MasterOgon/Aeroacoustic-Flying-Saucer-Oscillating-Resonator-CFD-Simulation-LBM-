import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gradio as gr
import tempfile
import copy
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# УВЕЛИЧЕННОЕ РАЗВЕРШЕНИЕ СЕТКИ ДО N=500
# ============================================================
N = 500 
X, Y = np.meshgrid(np.arange(N), np.arange(N))

c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
w = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=float)

cfd_multicolor_cmap = LinearSegmentedColormap.from_list("cfd_multicolor",
    [(0.00,"#004d99"), (0.10, "#00b7ff"), (0.25, "#00ffd0"), (0.40, "#00ff66"),
     (0.60, "#dfff00"), (0.85, "#ff9a00"), (1.00, "#ff2200")], N=512)
cfd_multicolor_cmap.set_bad(color='white')

def equilibrium(rho, u):
    feq = np.zeros((9, N, N), dtype=float)
    usqr = u[0]**2 + u[1]**2
    for i in range(9):
        cu = 3 * (c[i,0] * u[0] + c[i,1] * u[1])
        feq[i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*usqr)
    return feq

# ============================================================
# ДИНАМИЧЕСКАЯ МАСКА ГИБКОГО УПРУГОГО РЕЗОНАТОРА (ПЛАВНЫЙ ПЕРЕХОД)
# ============================================================
# ============================================================
# ДИНАМИЧЕСКАЯ МАСКА ГИБКОГО УПРУГОГО РЕЗОНАТОРА (ИСПРАВЛЕНАЯ ЛОГИКА)
# ============================================================
def get_dynamic_resonator_mask(
    cx, cy, attack_angle_deg, slide_angle_deg, phase, 
    amplitude, dyn_freq, wavelength, asymmetry_val,
    semi_major=64.0,  
    semi_minor=13.5   
):
    alpha = np.deg2rad(attack_angle_deg)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    # Локальные координаты относительно центра
    Xc = X - cx
    Yc = Y - cy

    Xr = Xc * cos_a + Yc * sin_a
    Yr = -Xc * sin_a + Yc * cos_a

    # Нормализованный угол скольжения от 0 (медуза) до 1 (рыба при slide_angle = 90)
    # Используем abs/нормировку на 90 градусов
    beta_deg = np.clip(abs(slide_angle_deg), 0.0, 90.0)
    w_wave = beta_deg / 90.0          # Вес режима рыбы (бегущая волна)
    w_perp = 1.0 - w_wave             # Вес режима медузы (поперечный изгиб)

    x_norm = np.clip(Xr / semi_major, -1.0, 1.0)
    
    # Если асимметрия нулевая, делаем коэффициент строго 0, чтобы колебания были гармоничными
    asym_factor = np.clip(asymmetry_val / 10.0, 0.0, 0.8) if asymmetry_val > 1e-4 else 0.0

    # 1. Поперечный дуговой изгиб (Режим медузы, slide_angle = 0)
    center_bow = (1.0 - x_norm**2)
    
    # Тайм-ворпинг фазы работает ТОЛЬКО в режиме медузы (при w_perp > 0) и если есть асимметрия
    warped_phase_medusa = phase - asym_factor * np.sin(phase) * w_perp
    
    bow_amplitude = amplitude * 1.8 
    bow_offset = bow_amplitude * np.sin(warped_phase_medusa) * center_bow

    # 2. Волновой изгиб (Режим рыбы, slide_angle = 90)
    ship_length = semi_major * 2.0
    k = (2.0 * np.pi) / (ship_length * max(0.1, wavelength))
    
    wave_phase_shift = Xr * k - phase
    
    # В режиме рыбы асимметрия управляет нарастанием амплитуды к хвосту (хвост машет шире носа)
    # При asymmetry = 0 профиль гармоничен, при росте асимметрии хвост получает эффективный килевой взмах
    tail_amplification = 1.0 + (asym_factor * 1.2) * (Xr / semi_major) # хвост сзади (Xr > 0 или в зависимости от ориентации)
    
    wave_amplitude = amplitude * 1.5 * tail_amplification
    wave_offset = wave_amplitude * np.sin(wave_phase_shift) * w_wave

    # Суммарное плавное совмещение режимов колебания в зависимости от slide_angle
    total_offset = (bow_offset * w_perp) + (wave_offset * w_wave)
    Yr_bent = Yr - total_offset

    body_mask = (Xr**2) / (semi_major**2) + (Yr_bent**2) / (semi_minor**2) < 1.0
    return body_mask, total_offset, w_wave
    

# ============================================================
def simulate(
    frames=400,
    attack_angle=0.0,
    slide_angle=0.0,
    rotation_control=0.0,
    buoyancy=0.0,
    frequency=3.0,
    amplitude=1.5,
    asymmetry=0.0,
    gravity=0.06,
    tau=0.58,
    mass=1.5,
    damping=0.5,
    wavelength=0.85,
    vis_mode="Pressure"
):
    omega = 1.0 / tau
    rho = np.ones((N, N), dtype=float)
    u = np.zeros((2, N, N), dtype=float)
    f = equilibrium(rho, u)

    cx = float(N // 2)
    cy = float(N * 0.5)

    v_x = 0.0
    v_y = 0.0
    
    angular_velocity = 0.0
    moment_of_inertia = 2.5
    rotational_damping = 0.15
    resistance_accumulator = 0.0
    release_force = 0.0
    smoothed_thrust = 0.0
    wake_pos = 0.0
    wake_neg = 0.0
    last_wave_dir = 0.0

    current_phase = 0.0
    current_attack_angle = attack_angle

    # Физический момент (emergent)
    natural_torque = 0.0
    torque_scale = 0.0018
    control_torque_gain = 0.35

    amplitude = min(amplitude, 10.0)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#121212')
    ax.set_facecolor('#121212')
    ax.axis('off')
    fig.tight_layout()

    imgs = []

    beta_deg = np.clip(abs(slide_angle), 0.0, 90.0)
    w_wave = beta_deg / 90.0
    w_perp = 1.0 - w_wave


    for t in range(frames):
        # ----------------------------------------------------
        # ВЛИЯНИЕ СРЕДЫ НА РАБОТУ ДВИГАТЕЛЯ (ИЗ ФАЙЛА ДВИЖ)
        # ----------------------------------------------------
        feedback_strength = resistance_accumulator * 0.05 * damping
        dyn_freq = max(0.5, frequency - feedback_strength * frequency)
        dyn_asym = min(10.8, asymmetry + feedback_strength * 0.6)

        # 1. Шаг фазы с учетом асимметрии времени полутактов
        d_phase = 2.0 * np.pi * dyn_freq / 60.0
        warped_phase = current_phase + dyn_asym * 0.4 * np.cos(current_phase)
        current_phase += d_phase
        
        # 2. Целевая скорость
        v_scale = amplitude * (dyn_freq / 30.0)
        # Если асимметрия нулевая, убираем варпинг и делаем колебания чистым синусом (без движения)
        has_asymmetry = asymmetry > 1e-4
        asym_factor_val = min(1.0, asymmetry / 10.0) if has_asymmetry else 0.0

        d_phase = 2.0 * np.pi * dyn_freq / 60.0
        
        # Фазовый сдвиг для полутактов работает плавно
        if w_perp > 0.0 and has_asymmetry:
            warped_phase = current_phase + asym_factor_val * 0.4 * np.cos(current_phase)
        else:
            warped_phase = current_phase
            
        current_phase += d_phase
        
        # Целевая скорость: если асимметрия равна 0, синус симметричен за период -> средняя тяга 0
        v_scale = amplitude * (dyn_freq / 30.0)
        if has_asymmetry:
            v_target = -v_scale * np.sin(warped_phase) * (1.0 + asym_factor_val * np.sign(np.cos(current_phase)))
        else:
            v_target = -v_scale * np.sin(current_phase)
            
        
        
        solid_mask, local_offset, asym_tilt = get_dynamic_resonator_mask(
            cx, cy, current_attack_angle, slide_angle, current_phase,
            amplitude, dyn_freq, wavelength, dyn_asym
        )

        alpha = np.deg2rad(current_attack_angle)
        normal_x = -np.sin(alpha)
        normal_y = np.cos(alpha)
        
        beta = np.deg2rad(-slide_angle)
        thrust_dir_x = normal_x * np.cos(beta) + normal_y * np.sin(beta) 
        thrust_dir_y = -normal_x * np.sin(beta) + normal_y * np.cos(beta)

        drive_x = v_target * thrust_dir_x
        drive_y = v_target * thrust_dir_y

        rho_sum = np.sum(f, axis=0)
        rho_sum = np.clip(rho_sum, 0.1, 8.0)
        ux = np.sum(f * c[:,0].reshape(9,1,1), axis=0) / (rho_sum + 1e-5)
        uy = np.sum(f * c[:,1].reshape(9,1,1), axis=0) / (rho_sum + 1e-5)

        u[0] = np.clip(ux, -0.32, 0.32)
        u[1] = np.clip(uy, -0.32, 0.32)

        f = f * (1 - omega) + omega * equilibrium(rho_sum, u)

        # ---------- Расчёт силы и момента через обмен импульсом ----------
        # 1. Импульс жидкости на ячейках тела ДО задания скорости
        mom_x_before = np.sum(f[:, solid_mask] * c[:, 0:1], axis=0)
        mom_y_before = np.sum(f[:, solid_mask] * c[:, 1:2], axis=0)

        # 2. Задаём скорость и равновесие (как было)
        u[0][solid_mask] = drive_x
        u[1][solid_mask] = drive_y
        rho_sum[solid_mask] = 1.0

        feq = equilibrium(rho_sum, u)
        for i in range(9):
            f[i, solid_mask] = feq[i, solid_mask]

        # 3. Импульс ПОСЛЕ задания
        mom_x_after = drive_x * 1.0          # rho=1
        mom_y_after = drive_y * 1.0

        # 4. Сила, с которой жидкость действует на тело (противоположна изменению импульса жидкости)
        Fx_local = -(mom_x_after - mom_x_before)   # shape = (num_solid,)
        Fy_local = -(mom_y_after - mom_y_before)

        # 5. Координаты ячеек тела относительно центра
        ys, xs = np.where(solid_mask)          # ys — строки (Y), xs — столбцы (X)
        rx = xs - cx
        ry = ys - cy

        # 6. Крутящий момент (в 2D: τ = rx*Fy - ry*Fx)
        natural_torque = np.sum(rx * Fy_local - ry * Fx_local) * torque_scale
        # -----------------------------------------------------------------

        for i in range(9):
            f[i] = np.roll(f[i], c[i].astype(int), axis=(1,0))

        # ----------------------------------------------------
        # МОДЕЛЬ КАВЕРНЫ
        # ----------------------------------------------------
        effective_area = float(np.sum(solid_mask))
        
        rho_near_body = rho_sum[solid_mask]
        actual_delta_P = max(0.05, 1.0 - float(np.mean(rho_near_body)))

        current_energy_step = actual_delta_P * effective_area * abs(v_target)
        resistance_accumulator += current_energy_step
        resistance_accumulator *= 0.92

        current_dir = np.sign(v_target) if abs(v_target) > 1e-5 else 0.0

        
        # Нормируем перепад давления и размах (амплитуду) работы двигателя
        delta_P_norm   = np.clip(actual_delta_P / 0.45, 0.0, 1.8)      
        amplitude_norm = np.clip(amplitude / 5.0, 0.0, 2.0)        # Энергия зависит от масштаба колебаний

        # КПД коллапса определяется термодинамическим потенциалом и размахом, 
        # но не обнуляется ложно в точке смены знака скорости
        eta_col = 0.35 * delta_P_norm * (amplitude_norm / (0.3 + amplitude_norm)) 
        k_shock = 1.0 + 0.8 * delta_P_norm                         # Ударный гидромолот каверны

        
        # При смене полутакта сброшенная энергия формирует вихревое кольцо
        if current_dir != last_wave_dir and last_wave_dir != 0:
            boost_magnitude = eta_col * resistance_accumulator * k_shock
            
            # Направляем порцию энергии в нужный вихревой шлейф без изменения знака
            if last_wave_dir > 0:
                wake_pos = boost_magnitude
            else:
                wake_neg = boost_magnitude
                
            resistance_accumulator *= 0.35
            last_wave_dir = current_dir

        # Каждый вихревой шлейф независимо затухает, сохраняя инерцию в своем направлении
        wake_pos *= 0.88
        wake_neg *= 0.88
        
        # Результирующая сила — это честная сумма инерционных следов вихрей в среде
        release_force = wake_pos - wake_neg
        
        

        

        if last_wave_dir == 0.0:
            last_wave_dir = current_dir

        smoothed_thrust = smoothed_thrust * 0.82 + release_force * 0.18
        thrust = smoothed_thrust
        
        # ----------------------------------------------------
        # ДИНАМИКА И ВЕКТОР ТЯГИ С УЧЕТОМ УГЛОВОЙ ИНЕРЦИИ
        # ----------------------------------------------------
        # Старый управляющий момент (оставляем слабым)
        fish_wave_asymmetry = np.sin(current_phase) * rotation_control
        turn_force_magnitude = (thrust + release_force * 0.5) * rotation_control
        effective_lever_arm = 0.35 + 0.25 * abs(fish_wave_asymmetry) + 0.15 * abs(asym_tilt)
        control_torque = turn_force_magnitude * effective_lever_arm * control_torque_gain

        # Суммарный момент = физический + (ослабленный) управляющий
        total_torque = natural_torque + control_torque

        angular_acceleration = (total_torque / moment_of_inertia) - (angular_velocity * rotational_damping)
        angular_velocity += angular_acceleration
        current_attack_angle += angular_velocity
        current_attack_angle = np.clip(current_attack_angle, -360.0, 360.0)
        
        # ----------------------------------------------------
        # ПОЛНЫЙ РАСЧЕТ СИЛ И СРЫВА ПОТОКА (ИЗ ФАЙЛА ДВИЖ)
        # ----------------------------------------------------
        thrust_vec_x = thrust * thrust_dir_x
        thrust_vec_y = thrust * thrust_dir_y
        thrust_mag = np.sqrt(thrust_vec_x**2 + thrust_vec_y**2) + 1e-6
        
        unit_thrust_x = thrust_vec_x / thrust_mag
        unit_thrust_y = thrust_vec_y / thrust_mag

        fluid_local_x = float(np.mean(u[0][solid_mask])) if np.any(solid_mask) else 0.0
        fluid_local_y = float(np.mean(u[1][solid_mask])) if np.any(solid_mask) else 0.0

        v_rel_x = v_x - fluid_local_x
        v_rel_y = v_y - fluid_local_y

        # Продольная составляющая с экранированием
        v_rel_parallel = v_rel_x * unit_thrust_x + v_rel_y * unit_thrust_y
        thrust_activity = min(1.0, abs(thrust) / 0.02 + abs(release_force) / 0.025)
        parallel_shield = max(0.05, 1.0 - 0.95 * thrust_activity)
        
        fx_parallel = thrust_vec_x - 0.04 * v_rel_parallel * unit_thrust_x * parallel_shield
        fy_parallel = thrust_vec_y - 0.04 * v_rel_parallel * unit_thrust_y * parallel_shield

        # Поперечная и вертикальная составляющие (сопротивление и срыв потока)
        unit_perp_x = -unit_thrust_y
        unit_perp_y = unit_thrust_x
        
        v_rel_perp = v_rel_x * unit_perp_x + v_rel_y * unit_perp_y
        v_rel_perp_mag = abs(v_rel_perp)

        effective_angle = current_attack_angle - np.degrees(np.arctan2(v_rel_y, v_rel_x))
        effective_angle = (effective_angle + 180.0) % 360.0 - 180.0
        
        alpha_crit = 18.0
        if abs(effective_angle) < alpha_crit:
            f_stall = 1.0
        else:
            excess = abs(effective_angle) - alpha_crit
            f_stall = max(0.2, 1.0 - 2.5 * excess)

        alpha_rad = np.deg2rad(current_attack_angle)
        lift_coefficient = np.sin(2.0 * alpha_rad)
        
        fx_perp = -0.08 * v_rel_perp * v_rel_perp_mag * unit_perp_x * f_stall + lift_coefficient * unit_perp_x * 0.02
        fy_perp = -0.08 * v_rel_perp * v_rel_perp_mag * unit_perp_y * f_stall + lift_coefficient * unit_perp_y * 0.02

        total_fx = fx_parallel + fx_perp
        total_fy = fy_parallel + fy_perp + gravity * mass - buoyancy * mass
        
        accel_x = total_fx / mass * 0.24
        accel_y = total_fy / mass * 0.24

        v_x += accel_x
        v_y += accel_y

        move_speed = 0.52
        cx += v_x * move_speed
        cy += v_y * move_speed
        
        cx = np.clip(cx, 0, N - 1)
        cy = np.clip(cy, 0, N - 1)
        
        # ----------------------------------------------------
        # ВИЗУАЛИЗАЦИЯ
        # ---------------------------------------------------- ь
        pressure = rho_sum - 1.0

        if vis_mode == "Pressure":
            data = np.nan_to_num(pressure, nan=0.0)
            vabs = float(np.percentile(np.abs(data), 99.5)) or 1e-5
            vmin, vmax = -vabs, vabs
        elif vis_mode == "Pressure Deviation":
            data = np.abs(np.nan_to_num(pressure, nan=0.0))
            vmax = float(np.percentile(data, 99.7)) or 1e-5
            vmin = 0.0
        elif vis_mode == "Vorticity":
            vort = ((np.roll(u[1], -1, axis=1) - np.roll(u[1], 1, axis=1)) * 0.5 -
                    (np.roll(u[0], -1, axis=0) - np.roll(u[0], 1, axis=0)) * 0.5)
            data = np.abs(np.nan_to_num(vort, nan=0.0))
            vmax = float(np.percentile(data, 99.7)) or 1e-5
            vmin = 0.0
        else:
            data = np.nan_to_num(pressure, nan=0.0)
            vabs = float(np.percentile(np.abs(data), 99.5)) or 1e-5
            vmin, vmax = -vabs, vabs

        data[solid_mask] = np.nan
        im = ax.imshow(data, cmap=cfd_multicolor_cmap, animated=True, vmin=vmin, vmax=vmax, interpolation='bilinear')
        
        imgs.append([im])

    ani = animation.ArtistAnimation(fig, imgs, interval=28, blit=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ani.save(tmp.name, fps=25, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)

    return tmp.name

# ============================================================
with gr.Blocks() as demo:
    
    with gr.Column(elem_classes="border-class"):
        output = gr.Video(label="Simulation Video (1:1)", elem_classes="square-video-box")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Global")
                frames = gr.Slider(minimum=50, maximum=1000, value=150, step=50, label="Frames")
            
            with gr.Column():
                gr.Markdown("#### Flight Control")
                attack_angle = gr.Slider(minimum=-45, maximum=45, value=0.0, step=22.5, label="Attack Angle (°)")
                slide_angle = gr.Slider(minimum=-180, maximum=180, value=0.0, step=1.0, label="Slide Angle (°)")
                rotation_control = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.05, label="Dynamic Yaw / Turn Rate")
                buoyancy = gr.Slider(minimum=-0.25, maximum=0.25, value=0.0, step=0.005, label="Buoyancy")

            with gr.Column():
                gr.Markdown("#### Engine")
                frequency = gr.Slider(minimum=0.5, maximum=8.0, value=3.0, step=0.5, label="Base Frequency (Hz)")
                amplitude = gr.Slider(minimum=0.1, maximum=10.0, value=2.25, step=0.05, label="Amplitude")
                asymmetry = gr.Slider(minimum=0.0, maximum=10.0, value=2.5, step=0.05, label="Base Asymmetry")
                wavelength = gr.Slider(minimum=0.3, maximum=3.0, value=0.85, step=0.05, label="Wavelength (L)")
                
            with gr.Column():
                gr.Markdown("#### Physics & Vis")
                gravity = gr.Slider(minimum=0.0, maximum=0.25, value=0.06, step=0.002, label="Gravity")
                tau = gr.Slider(minimum=0.51, maximum=0.95, value=0.58, step=0.01, label="Viscosity (Tau)")
                mass = gr.Slider(minimum=0.5, maximum=30.0, value=1.5, step=0.1, label="Mass")
                damping = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Feedback")
                vis_mode = gr.Radio(
                    choices=["Pressure", "Pressure Deviation", "Vorticity"],
                    value="Pressure Deviation",
                    label="Visualization Mode"
                )

        btn = gr.Button("RUN SIMULATION", variant="primary")

    btn.click(
        simulate, 
        inputs=[
            frames, attack_angle, slide_angle, rotation_control, buoyancy, 
            frequency, amplitude, asymmetry, gravity, tau, mass, damping, wavelength, vis_mode
        ], 
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
