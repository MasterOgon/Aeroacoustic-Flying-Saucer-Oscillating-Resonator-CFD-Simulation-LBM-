import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gradio as gr
import tempfile
import copy
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# Лайт-версия: размер сетки уменьшен для скорости (было 500, стало 300)
N = 300 
X, Y = np.meshgrid(np.arange(N), np.arange(N))

c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
w = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=float)

# Оставлена только одна цветовая схема для давления
cfd_multicolor_cmap = LinearSegmentedColormap.from_list("cfd_multicolor",
    [(0.00,"#004d99"), (0.10, "#00b7ff"), (0.25, "#00ffd0"), (0.40, "#00ff66"),
     (0.60, "#dfff00"), (0.85, "#ff9a00"), (1.00, "#ff2200")], N=512)
cfd_multicolor_cmap.set_bad(color='lightgray')

def equilibrium(rho, u):
    feq = np.zeros((9, N, N), dtype=float)
    usqr = u[0]**2 + u[1]**2
    for i in range(9):
        cu = 3 * (c[i,0] * u[0] + c[i,1] * u[1])
        feq[i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*usqr)
    return feq

# Габариты аппарата уменьшены (было 34 и 8, стало 16 и 4.5)
def get_rotated_ellipse_mask(cx, cy, angle_deg, semi_major=16.0, semi_minor=4.5):
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    Xc = X - cx
    Yc = Y - cy

    Xr = Xc * cos_a + Yc * sin_a
    Yr = -Xc * sin_a + Yc * cos_a

    body = (Xr**2)/semi_major**2 + (Yr**2)/semi_minor**2 < 1.0
    dome_radius = semi_minor * 1.3
    dome_center_x = 0.0
    dome_center_y = -semi_minor * 0.5  

    dome = (
        ((Xr - dome_center_x)**2 + (Yr - dome_center_y)**2) < dome_radius**2
    ) & (Yr < dome_center_y)

    return body | dome

# ============================================================
def simulate(
    frames=400,
    attack_angle=0.0,
    slide_angle=0.0,
    buoyancy=0.0,
    frequency=3.0,
    amplitude=1.5,
    asymmetry=0.0,
    gravity=0.06,
    tau=0.58,
    mass=0.65,
    damping=0.5,
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
    
    peak_drag = 0.0
    release_force = 0.0
    last_direction = 0.0
    current_phase = 0.0
    # Жесткий лимит амплитуды: половина от semi_minor (4.5 / 2 = 2.25)
    amplitude = min(amplitude, 2.25)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#121212')
    ax.set_facecolor('#121212')
    ax.axis('off')
    fig.tight_layout()

    imgs = []
    alpha = np.deg2rad(attack_angle)

    for t in range(frames):
        # ----------------------------------------------------
        # ВЛИЯНИЕ СРЕДЫ НА РАБОТУ ДВИГАТЕЛЯ
        # ----------------------------------------------------
        feedback_strength = peak_drag * damping * 0.15 
        dyn_freq = max(0.5, frequency - feedback_strength * frequency)
        dyn_asym = min(1.8, asymmetry + feedback_strength * 0.6)
        
        current_phase += 2 * np.pi * dyn_freq / 60.0
        
        v_base = -amplitude * np.sin(current_phase) * (dyn_freq / 30.0)
        v_target = v_base * (1.0 + dyn_asym * np.cos(current_phase)) if dyn_asym > 0 else v_base

        solid_mask = get_rotated_ellipse_mask(cx, cy, attack_angle)

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

        u[0][solid_mask] = drive_x
        u[1][solid_mask] = drive_y
        rho_sum[solid_mask] = 1.0

        feq = equilibrium(rho_sum, u)
        for i in range(9):
            f[i, solid_mask] = feq[i, solid_mask]

        for i in range(9):
            f[i] = np.roll(f[i], c[i].astype(int), axis=(1,0))

        # ----------------------------------------------------
        # РАСЧЕТ ТЯГИ ПО ФАКТИЧЕСКОМУ ДАВЛЕНИЮ СЕТКИ LBM
        # ----------------------------------------------------
        effective_area = float(np.sum(solid_mask))
        eta_collapse = 0.15  # КПД гидродинамического возврата (15%)

        # 1. Извлекаем фактическую плотность среды под маской/у поверхности
        rho_near_body = rho_sum[solid_mask] 
        
        # 2. Фактический перепад давления (где 1.0 - базовая плотность покоя)
        # max(0.05, ...) защищает от отрицательного или нулевого значения при разрывах сетки
        actual_delta_P = max(0.05, 1.0 - float(np.mean(rho_near_body)))

        # 3. Энергия каверны на основе РЕАЛЬНОГО давления сетки
        energy_stored = actual_delta_P * effective_area * abs(v_target)
        
        current_direction = np.sign(v_target) if abs(v_target) > 1e-5 else 0.0

        # 4. В момент смены направления движения происходит коллапс каверны:
        if current_direction != last_direction and last_direction != 0:
            # Тяга пропорциональна фактическому перепаду давления actual_delta_P
            release_force = eta_collapse * actual_delta_P * effective_area * last_direction * 1.5
            peak_drag = 0.0

        if energy_stored > peak_drag:
            peak_drag = energy_stored
        else:
            # Динамическое затухание импульса с учетом вязкости
            decay_rate = 0.78 + (0.20 * damping * (1.0 - dyn_freq / max(frequency, 0.1)))
            decay_rate = min(0.98, decay_rate)
            release_force *= decay_rate

        last_direction = current_direction
        thrust = release_force
        
        thrust_x = thrust * thrust_dir_x
        thrust_y = thrust * thrust_dir_y

        total_fx = thrust_x
        total_fy = thrust_y + gravity * mass - buoyancy * mass

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
        # ВЫБОР РЕЖИМА ВИЗУАЛИЗАЦИИ
        # ----------------------------------------------------
        pressure = rho_sum - 1.0

        if vis_mode == "Pressure":
            data = np.nan_to_num(pressure, nan=0.0)
            vabs = float(np.percentile(np.abs(data), 99.5)) or 1e-5
            vmin, vmax = -vabs, vabs
        elif vis_mode == "Pressure Deviation":
            data = np.nan_to_num(np.abs(pressure), nan=0.0)
            vmax = float(np.percentile(data, 99.7)) or 1e-5
            vmin = 0.0
        elif vis_mode == "Vorticity":
            vort = np.gradient(u[1], axis=0) - np.gradient(u[0], axis=1)
            data = np.nan_to_num(np.abs(vort), nan=0.0)
            vmax = float(np.percentile(data, 99.7)) or 1e-5
            vmin = 0.0
        else:
            data = np.nan_to_num(pressure, nan=0.0)
            vabs = float(np.percentile(np.abs(data), 99.5)) or 1e-5
            vmin, vmax = -vabs, vabs

        data[solid_mask] = np.nan
        im = ax.imshow(data, cmap=cfd_multicolor_cmap, animated=True, vmin=vmin, vmax=vmax, interpolation='bilinear')
        
        current_speed = np.sqrt(v_x**2 + v_y**2)

        telemetry_box = (
            f"TELEMETRY\n"
            f"---------------------\n"
            f"DYN FREQ : {dyn_freq:.2f} Hz\n"
            f"DYN ASYM : {dyn_asym:.2f}\n"
            f"RELEASE  : {release_force:.3f}\n"
            f"VELOCITY : {current_speed:.4f}\n" 
            f"COORD    : ({cx:.1f}, {cy:.1f})\n"
        )

        tx = ax.text(10, 15, telemetry_box, color='#ffffff', fontsize=8, family='monospace',
                     va='top', ha='left', bbox=dict(facecolor='#111111', alpha=0.65,
                                                    edgecolor='#333333', boxstyle='round,pad=0.5'))
        
        imgs.append([im, tx])

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
                frames = gr.Slider(minimum=50, maximum=1000, value=100, step=50, label="Frames")
            
            with gr.Column():
                gr.Markdown("#### Flight Control")
                attack_angle = gr.Slider(minimum=-45, maximum=45, value=0.0, step=22.5, label="Attack Angle (°)")
                slide_angle = gr.Slider(minimum=-180, maximum=180, value=0.0, step=1.0, label="Slide Angle (°)")
                buoyancy = gr.Slider(minimum=-0.25, maximum=0.25, value=0.0, step=0.005, label="Buoyancy")

            with gr.Column():
                gr.Markdown("#### Engine")
                frequency = gr.Slider(minimum=0.5, maximum=8.0, value=3.0, step=0.5, label="Base Frequency (Hz)")
                amplitude = gr.Slider(minimum=0.1, maximum=2.25, value=1.5, step=0.05, label="Amplitude")
                asymmetry = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Base Asymmetry")
                
            with gr.Column():
                gr.Markdown("#### Physics & Vis")
                gravity = gr.Slider(minimum=0.0, maximum=0.25, value=0.06, step=0.002, label="Gravity")
                tau = gr.Slider(minimum=0.51, maximum=0.95, value=0.58, step=0.01, label="Viscosity (Tau)")
                mass = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="Mass")
                damping = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Feedback")
                vis_mode = gr.Radio(
                    choices=["Pressure", "Pressure Deviation", "Vorticity"],
                    value="Pressure",
                    label="Visualization Mode"
                )

        btn = gr.Button("RUN SIMULATION", variant="primary")

    btn.click(
        simulate, 
        inputs=[
            frames, attack_angle, slide_angle, buoyancy, 
            frequency, amplitude, asymmetry, gravity, tau, mass, damping, vis_mode
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
            footer {display: none !important;}
        """
    )