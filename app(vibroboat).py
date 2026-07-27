import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import gradio as gr
import tempfile

# ============================================================
# ПАРАМЕТРЫ СЕТКИ LBM И ЦВЕТОВАЯ ПАЛИТРА
# ============================================================
N = 250
X, Y = np.meshgrid(np.arange(N), np.arange(N))

c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
w = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=float)
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)

cfd_multicolor_cmap = LinearSegmentedColormap.from_list("cfd_multicolor",
    [(0.00,"#004d99"), (0.10, "#00b7ff"), (0.25, "#00ffd0"), (0.40, "#00ff66"), 
     (0.60, "#dfff00"), (0.85, "#ff9a00"), (1.00, "#ff2200")], N=512)
cfd_multicolor_cmap.set_bad(color='#1c1c1c')

def equilibrium(rho, u):
    feq = np.zeros((9, N, N), dtype=float)
    usqr = u[0]**2 + u[1]**2
    for i in range(9):
        cu = 3.0 * (c[i,0] * u[0] + c[i,1] * u[1])
        feq[i] = rho * w[i] * (1.0 + cu + 0.5*cu**2 - 1.5*usqr)
    return feq

def equilibrium_1d(rho_1d, u_1d):
    feq = np.zeros((9, len(rho_1d)), dtype=float)
    usqr = u_1d[0]**2 + u_1d[1]**2
    for i in range(9):
        cu = 3.0 * (c[i,0] * u_1d[0] + c[i,1] * u_1d[1])
        feq[i] = rho_1d * w[i] * (1.0 + cu + 0.5*cu**2 - 1.5*usqr)
    return feq

def get_boundary_mask(cx, cy, radius):
    solid = (X - cx)**2 + (Y - cy)**2 < radius**2
    eroded = ndimage.binary_erosion(solid, structure=np.ones((3, 3), dtype=bool))
    boundary = solid & ~eroded
    return solid, boundary

# ============================================================
# ОСНОВНОЙ ДВИЖОК
# ============================================================
def simulate(frames=400, frequency=3.0, relative_density=1.0, k_attract=150.0, 
             motor_stiffness=5.0, tau=0.58, vis_mode="Pressure"):
    
    omega_lbm = 1.0 / tau
    rho = np.ones((N, N), dtype=float)
    u = np.zeros((2, N, N), dtype=float)
    f = equilibrium(rho, u)
    
    # Геометрия и массы
    R_boat = 22.0
    R_p = 10.0            
    R_stator = 18.0       
    
    M_displaced = np.pi * R_boat**2 * 1.0
    m_h = relative_density * M_displaced
    m_p = 0.1 * m_h
    M_tot = m_h + m_p
    
    I_p = m_p * R_p**2
    I_boat = 0.5 * m_h * R_boat**2
    I_added_rot = 0.0 * M_displaced * R_boat**2
    I_eff = I_boat + I_added_rot
    M_added = 0.0 * M_displaced
    M_eff = M_tot + M_added
    
    # Начальные параметры
    cm_x, cm_y = float(N)*0.35, float(N)*0.5
    v_cm_x, v_cm_y = 0.0, 0.0
    theta, omega_h, phi = 0.0, 0.0, 0.0
    
    dt_lbm = 1.0
    LBM_DT = 0.05 
    lbm_frequency = frequency * LBM_DT
    lbm_k_attract = k_attract * (LBM_DT**2)
    lbm_motor_stiffness = motor_stiffness * (LBM_DT**2)
    omega_p = lbm_frequency * 2 * np.pi 
    prev_omega_rel = omega_p - omega_h
    
    tau_mag_p_filtered, tau_mag_h_filtered = 0.0, 0.0
    Ff_x, Ff_y, Tf = 0.0, 0.0, 0.0
    
    cx_init = cm_x - (m_p / M_tot) * R_p * np.cos(phi)
    cy_init = cm_y - (m_p / M_tot) * R_p * np.sin(phi)
    solid_old, _ = get_boundary_mask(cx_init, cy_init, R_boat)
    
    # Sponge Layer
    sponge_thickness = 20
    sponge_sigma = np.zeros((N, N), dtype=float)
    for i in range(sponge_thickness):
        damping = ((sponge_thickness - i) / sponge_thickness)**2 * 0.15 
        sponge_sigma[i, :] = np.maximum(sponge_sigma[i, :], damping)
        sponge_sigma[-1-i, :] = np.maximum(sponge_sigma[-1-i, :], damping)
        sponge_sigma[:, i] = np.maximum(sponge_sigma[:, i], damping)
        sponge_sigma[:, -1-i] = np.maximum(sponge_sigma[:, -1-i], damping)
        
    f_rest = equilibrium(np.ones((N, N)), np.zeros((2, N, N)))
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#121212')
    ax.set_facecolor('#121212')
    ax.axis('off')
    fig.tight_layout()
    imgs = []
    
    pressure_old = np.zeros((N, N), dtype=float)
    stored_energy_x = np.zeros((N, N), dtype=float)
    stored_energy_y = np.zeros((N, N), dtype=float)
    energy_decay = 1.0
    vortex_efficiency = 1.0
    
    for t in range(frames):
        # 1. КИНЕМАТИКА
        rho_x, rho_y = R_p * np.cos(phi), R_p * np.sin(phi)
        cx = cm_x - (m_p / M_tot) * rho_x
        cy = cm_y - (m_p / M_tot) * rho_y
        drho_x = -omega_p * R_p * np.sin(phi)
        drho_y =  omega_p * R_p * np.cos(phi)
        vx_h = v_cm_x - (m_p / M_tot) * drho_x
        vy_h = v_cm_y - (m_p / M_tot) * drho_y
        
        # 2. ВНУТРЕННИЕ СИЛЫ
        P_x, P_y = cx + rho_x, cy + rho_y
        S_x, S_y = cx + R_stator * np.cos(theta), cy + R_stator * np.sin(theta)
        
        dist_x, dist_y = S_x - P_x, S_y - P_y
        dist_sq = dist_x**2 + dist_y**2
        F_mag_mag = (lbm_k_attract * 5000.0) / (dist_sq + 16.0) 
        F_mag_x, F_mag_y = F_mag_mag * dist_x, F_mag_mag * dist_y
        
        tau_mag_p_raw = rho_x * F_mag_y - rho_y * F_mag_x
        S_rel_x, S_rel_y = R_stator * np.cos(theta), R_stator * np.sin(theta)
        tau_mag_h_raw = S_rel_x * (-F_mag_y) - S_rel_y * (-F_mag_x)
        
        tau_mag_p_filtered = 0.90 * tau_mag_p_filtered + 0.10 * tau_mag_p_raw
        tau_mag_h_filtered = 0.90 * tau_mag_h_filtered + 0.10 * tau_mag_h_raw
        
        target_omega_rel = lbm_frequency * 2 * np.pi
        omega_rel = omega_p - omega_h
        d_omega_rel = (omega_rel - prev_omega_rel) / dt_lbm
        prev_omega_rel = omega_rel
        
        Kp = lbm_motor_stiffness * 5.0
        Kd = Kp * 0.15
        tau_motor_raw = Kp * (target_omega_rel - omega_rel) - Kd * d_omega_rel
        max_motor_torque = lbm_motor_stiffness * 2.0 
        tau_motor = np.clip(tau_motor_raw, -max_motor_torque, max_motor_torque)
        
        # 3. LBM: СТОЛКНОВЕНИЕ
        mask_fluid = ~solid_old
        rho_sum = np.sum(f, axis=0)
        u[0] = np.sum(f * c[:,0].reshape(9,1,1), axis=0) / (rho_sum + 1e-5)
        u[1] = np.sum(f * c[:,1].reshape(9,1,1), axis=0) / (rho_sum + 1e-5)
        
        max_fluid_u = 0.25 
        u[0] = np.clip(u[0], -max_fluid_u, max_fluid_u)
        u[1] = np.clip(u[1], -max_fluid_u, max_fluid_u)
        u[0, solid_old] = 0.0
        u[1, solid_old] = 0.0
        
        feq = equilibrium(rho_sum, u)
        f[:, mask_fluid] = f[:, mask_fluid] * (1 - omega_lbm) + omega_lbm * feq[:, mask_fluid]
        f = f * (1.0 - sponge_sigma) + f_rest * sponge_sigma
        
        # 4. LBM: ПЕРЕНОС
        for i in range(9):
            f[i] = np.roll(f[i], c[i].astype(int), axis=(1,0))
            
        # 5. LBM: ОБНОВЛЕНИЕ ГРАНИЦ И REFILL
        solid_new, boundary_new = get_boundary_mask(cx, cy, R_boat)
        uncovered = solid_old & ~solid_new 
        r_vec_x, r_vec_y = X - cx, Y - cy
        
        if np.any(uncovered):
            alpha = 0.25
            ux_unc = alpha * (vx_h - omega_h * r_vec_y[uncovered]) + (1-alpha) * u[0][uncovered]
            uy_unc = alpha * (vy_h + omega_h * r_vec_x[uncovered]) + (1-alpha) * u[1][uncovered]
            rho_unc = np.ones_like(ux_unc)
            feq_refill = equilibrium_1d(rho_unc, np.array([ux_unc, uy_unc]))
            for i in range(9):
                f[i, uncovered] = feq_refill[i]
                
        # 6. LBM: BOUNCE-BACK И РАСЧЕТ ЛОКАЛЬНОЙ РАБОТЫ
        F_fluid_x_raw, F_fluid_y_raw, tau_fluid_raw = 0.0, 0.0, 0.0
        f_new = np.copy(f)
        y_coords, x_coords = np.where(boundary_new)
        
        # <<< НОВОЕ: сетка для аккумулирования локальной работы стенки >>>
        local_work_grid = np.zeros((N, N), dtype=float)
        
        for i in range(1, 9):
            o = opp[i]
            src_y = np.clip(y_coords - int(c[i, 1]), 0, N-1)
            src_x = np.clip(x_coords - int(c[i, 0]), 0, N-1)
            was_fluid = ~solid_old[src_y, src_x]
            is_fluid = ~solid_new[src_y, src_x]
            is_fluid_source = was_fluid | is_fluid
            
            valid_y = y_coords[is_fluid_source] # узлы корпуса
            valid_x = x_coords[is_fluid_source]
            fluid_y = src_y[is_fluid_source]    # соседние узлы жидкости
            fluid_x = src_x[is_fluid_source]
            
            if len(valid_y) == 0:
                continue
                
            vw_lin_x, vw_lin_y = vx_h, vy_h
            rx_bb, ry_bb = r_vec_x[valid_y, valid_x], r_vec_y[valid_y, valid_x]
            vw_rot_x, vw_rot_y = -omega_h * ry_bb, omega_h * rx_bb
            
            v_wall_x = vw_lin_x + vw_rot_x
            v_wall_y = vw_lin_y + vw_rot_y
            
            cu_lin = 3.0 * (c[i,0] * vw_lin_x + c[i,1] * vw_lin_y)
            cu_rot = 3.0 * (c[i,0] * vw_rot_x + c[i,1] * vw_rot_y)
            rho_wall = rho_sum[valid_y, valid_x]
            f_in = f[i, valid_y, valid_x]
            
            f_out = f_in - 2.0 * w[i] * rho_wall * cu_lin - 2.0 * w[i] * rho_wall * cu_rot
            f_new[o, valid_y, valid_x] = f_out
            
            dp_x = (f_in + f_out) * c[i,0]
            dp_y = (f_in + f_out) * c[i,1]
            
            F_fluid_x_raw += np.sum(dp_x)
            F_fluid_y_raw += np.sum(dp_y)
            tau_fluid_raw += np.sum(rx_bb * dp_y - ry_bb * dp_x)

            # ... (предыдущий код bounce-back, расчет dp_x, dp_y остаются без изменений, 
            # чтобы LBM физика работала с полной скоростью vx_h, vy_h) ...

            # 1. Выделяем чистую локальную скорость колебаний от внутреннего маятника (без v_cm)
            v_osc_lin_x = -(m_p / M_tot) * drho_x
            v_osc_lin_y = -(m_p / M_tot) * drho_y
            
            # 2. Формируем скорость ИСКЛЮЧИТЕЛЬНО для расчета энергии (колебания + вращение)
            
            v_energy_x = v_osc_lin_x + vw_rot_x
            v_energy_y = v_osc_lin_y + vw_rot_y
            # 3. Считаем работу (dW), используя только "внутреннюю" энергию движителя
            dW = ((-dp_x * v_energy_x) + (-dp_y * v_energy_y)) * dt_lbm
            
            
            
            # Аккумулируем только ту работу, которая "накачивает" жидкость энергией
            dW_pos = np.maximum(dW, 0.0)
            
            # Точечно плюсуем энергию прямо в прилегающие пиксели жидкости
            np.add.at(local_work_grid, (fluid_y, fluid_x), dW_pos)
            
        f = f_new
        solid_old = np.copy(solid_new)
        
        rho_sum = np.sum(f, axis=0)
        u[0] = np.sum(f * c[:,0].reshape(9,1,1), axis=0) / (rho_sum + 1e-5)
        u[1] = np.sum(f * c[:,1].reshape(9,1,1), axis=0) / (rho_sum + 1e-5)
        pressure = rho_sum - 1.0
        
        # 7. ДИНАМИКА НЬЮТОНА + ИЗОТРОПНЫЙ НАКОПИТЕЛЬ
        alpha_force = 0.5 
        Ff_x += alpha_force * (F_fluid_x_raw - Ff_x)
        Ff_y += alpha_force * (F_fluid_y_raw - Ff_y)
        Tf   += alpha_force * (tau_fluid_raw - Tf)
        
        v_old_x, v_old_y = v_cm_x, v_cm_y
        v_cm_x += (Ff_x / M_eff) * dt_lbm
        v_cm_y += (Ff_y / M_eff) * dt_lbm
        
        # Выделяем "скорлупу" жидкости прямо вокруг корпуса
        shell = ndimage.binary_dilation(solid_new, iterations=1) & (~solid_new)
        
        p_shell = pressure[shell] / 3.0 
        p_abs = np.abs(p_shell)
        p_abs_old = np.abs(pressure_old[shell] / 3.0)
        dp_abs = p_abs - p_abs_old # >0 если растет, <0 если падает
        
        vec_x = cm_x - X[shell]
        vec_y = cm_y - Y[shell]
        vec_mag = np.hypot(vec_x, vec_y) + 1e-12
        dir_to_cm_x = vec_x / vec_mag
        dir_to_cm_y = vec_y / vec_mag
        
        v_rot_edge = np.abs(omega_h) * R_boat 
        v_trans = np.hypot(v_cm_x, v_cm_y)    
        
        flapping_ratio = v_rot_edge / (v_rot_edge + v_trans + 1e-6)
        # <<< НОВОЕ: Отказ от глобального распределения >>>
        # Энергия уже распределена в массиве local_work_grid именно там, где нужно.
        # Мы просто применяем множитель виляния (flapping ratio), чтобы изолировать колебательную часть.
        local_energy_input = local_work_grid[shell] * flapping_ratio
        
        stored_energy_x[shell] += local_energy_input * dir_to_cm_x
        stored_energy_y[shell] += local_energy_input * dir_to_cm_y
        
        stored_energy_x *= energy_decay
        stored_energy_y *= energy_decay
        
        # ОТДАЧА: Пропорционально проценту падения давления
        release_fraction = np.clip(-dp_abs / (p_abs_old + 1e-8), 0.0, 1.0)
        Ex_release = stored_energy_x[shell] * release_fraction
        Ey_release = stored_energy_y[shell] * release_fraction
        
        stored_energy_x[shell] -= Ex_release
        stored_energy_y[shell] -= Ey_release
        
        thrust_multiplier = 6.5  
        W_net_x = -np.sum(Ex_release) * vortex_efficiency * thrust_multiplier
        W_net_y = -np.sum(Ey_release) * vortex_efficiency * thrust_multiplier
        
        if abs(W_net_x) > 0 or abs(W_net_y) > 0:
            Ek_x = np.sign(v_cm_x) * (0.5 * M_eff * v_cm_x**2)
            Ek_y = np.sign(v_cm_y) * (0.5 * M_eff * v_cm_y**2)
            Ek_x_new = Ek_x + W_net_x
            Ek_y_new = Ek_y + W_net_y
            v_cm_x = np.sign(Ek_x_new) * np.sqrt(2.0 * np.abs(Ek_x_new) / M_eff)
            v_cm_y = np.sign(Ek_y_new) * np.sqrt(2.0 * np.abs(Ek_y_new) / M_eff)
            
        max_v = 0.25
        v_cm_x = np.clip(v_cm_x, -max_v, max_v)
        v_cm_y = np.clip(v_cm_y, -max_v, max_v)
        
        cm_x += v_cm_x * dt_lbm
        cm_y += v_cm_y * dt_lbm
        
        margin = R_boat + 2.0
        cm_x = np.clip(cm_x, margin, N - margin)
        cm_y = np.clip(cm_y, margin, N - margin)
        
        a_x = (v_cm_x - v_old_x) / dt_lbm
        a_y = (v_cm_y - v_old_y) / dt_lbm
        tau_inertial = m_p * (a_x * rho_y - a_y * rho_x)
        
        omega_p += ((tau_motor + tau_mag_p_filtered + tau_inertial) / I_p) * dt_lbm
        omega_max = lbm_frequency * 2.0 * np.pi
        if abs(omega_p) > omega_max:
            excess = abs(omega_p) - omega_max
            omega_p -= np.sign(omega_p) * 0.25 * excess
            
        omega_h += ((Tf - tau_motor + tau_mag_h_filtered) / I_eff) * dt_lbm
        phi += omega_p * dt_lbm
        theta += omega_h * dt_lbm
        
        pressure_old = np.copy(pressure)
        
        # ============================================================
        # ВИЗУАЛИЗАЦИЯ
        # ============================================================
        if t % 2 == 0: 
            if vis_mode == "Pressure":
                data = np.nan_to_num(pressure, nan=0.0)
                vabs = float(np.percentile(np.abs(data), 99.5)) or 1e-5
                vmin, vmax = -vabs, vabs
                cmap = cfd_multicolor_cmap
            elif vis_mode == "Pressure Deviation":
                data = np.nan_to_num(np.abs(pressure), nan=0.0)
                vmax = float(np.percentile(data, 99.7)) or 1e-5
                vmin = 0.0
                cmap = cfd_multicolor_cmap
            elif vis_mode == "Vorticity":
                vort = np.gradient(u[1], axis=0) - np.gradient(u[0], axis=1)
                data = np.nan_to_num(np.abs(vort), nan=0.0)   
                vmax = float(np.percentile(data, 99.7)) or 1e-5
                vmin = 0.0
                cmap = cfd_multicolor_cmap
                
            data[solid_new] = np.nan
            im = ax.imshow(data, cmap=cmap, animated=True, vmin=vmin, vmax=vmax, interpolation='bilinear')
            frame_artists = [im]
            
            c1_patch = Circle((cx, cy), R_boat, color='none', ec='white', ls='solid', lw=1.5, animated=True)
            axle = Circle((cx, cy), 1.5, color='gray', animated=True, zorder=5)
            stator_mag = Circle((S_x, S_y), 3.5, color='#ff3333', animated=True, zorder=6) 
            nose_x, nose_y = cx + R_boat * np.cos(theta), cy + R_boat * np.sin(theta)
            nose, = ax.plot([cx, nose_x], [cy, nose_y], color='white', lw=1, ls='--', animated=True, zorder=3)
            pend = Circle((P_x, P_y), 3.5, color='#33ccff', animated=True, zorder=5) 
            rod, = ax.plot([cx, P_x], [cy, P_y], color='gray', lw=2, animated=True, zorder=4)
            
            ax.add_patch(c1_patch)
            ax.add_patch(axle)
            ax.add_patch(stator_mag)
            ax.add_patch(pend)
            frame_artists.extend([c1_patch, axle, stator_mag, nose, pend, rod])
            
            imgs.append(frame_artists)
            
    ani = animation.ArtistAnimation(fig, imgs, interval=40, blit=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ani.save(tmp.name, fps=25, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
    return tmp.name

# ============================================================
# Gradio интерфейс
# ============================================================
with gr.Blocks() as demo:
    with gr.Column(elem_classes="border-class"):
        output = gr.Video(label="Proper CFD-Grade LBM Boat Simulation", elem_classes="square-video-box")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Global Params")
                frames = gr.Slider(minimum=50, maximum=2000, value=400, step=50, label="Frames")
                frequency = gr.Slider(minimum=0.1, maximum=10.0, value=0.5, step=0.1, label="Base Freq (Hz)")
                relative_density = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Hull Density")
            with gr.Column():
                gr.Markdown("#### System Dynamics")
                k_attract = gr.Slider(minimum=0.0, maximum=500.0, value=10.0, step=10.0, label="Magnetic Attraction")
                motor_stiffness = gr.Slider(minimum=0.1, maximum=10.0, value=0.1, step=0.1, label="Motor PD-Stiffness")
                tau = gr.Slider(minimum=0.51, maximum=0.95, value=0.58, step=0.01, label="Viscosity (Tau)")
            with gr.Column():
                gr.Markdown("#### Visualization")
                vis_mode = gr.Radio(
                    choices=["Pressure", "Pressure Deviation", "Vorticity"],
                    value="Pressure", label="Visualization Mode"
                )
                btn = gr.Button("RUN SIMULATION", variant="primary")
                btn.click(simulate, inputs=[frames, frequency, relative_density, k_attract, motor_stiffness, tau, vis_mode], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)