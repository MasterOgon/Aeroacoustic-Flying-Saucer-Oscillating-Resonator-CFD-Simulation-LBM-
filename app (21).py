import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gradio as gr
import tempfile
import copy
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
N = 500
X, Y = np.meshgrid(np.arange(N), np.arange(N))

c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
w = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=float)

cfd_multicolor_cmap = LinearSegmentedColormap.from_list("cfd_multicolor",
    [(0.00,"#004d99"), (0.10, "#00b7ff"), (0.25, "#00ffd0"), (0.40, "#00ff66"),
     (0.60, "#dfff00"), (0.85, "#ff9a00"), (1.00, "#ff2200")], N=512)

def equilibrium(rho, u):
    feq = np.zeros((9, N, N), dtype=float)
    usqr = u[0]**2 + u[1]**2
    for i in range(9):
        cu = 3 * (c[i,0] * u[0] + c[i,1] * u[1])
        feq[i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*usqr)
    return feq

def get_rotated_ellipse_mask(cx, cy, angle_deg, semi_major=34.0, semi_minor=8.0):
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
    frames=500,
    view_mode="Vorticity",
    pro_gradient=True,
    attack_angle=0.0,
    slide_angle=0.0,
    buoyancy=0.0,
    frequency=3.0,
    amplitude=4.2,
    asymmetry=0.0,
    gravity=0.06,
    tau=0.58,
    mass=0.65,
    damping=0.5  # <--- НОВЫЙ ПАРАМЕТР ФИЗИКИ
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
    
    # Переменная для непрерывного накопления фазы (из-за динамической частоты)
    current_phase = 0.0
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#121212')
    ax.set_facecolor('#121212')
    ax.axis('off')
    fig.tight_layout()

    imgs = []

    if pro_gradient:
        cmap_choice = copy.copy(cfd_multicolor_cmap)
    else:
        cmap_choice = copy.copy(plt.get_cmap('RdBu_r'))
        
    cmap_choice.set_bad(color='lightgray') 
    
    alpha = np.deg2rad(attack_angle)

    for t in range(frames):
        # ----------------------------------------------------
        # ВЛИЯНИЕ СРЕДЫ НА РАБОТУ ДВИГАТЕЛЯ (КИНЕМАТИКА МАЯТНИКА)
        # ----------------------------------------------------
        # Расчет модификатора сопротивления. Чем больше пиковое сопротивление и демпинг,
        # тем сильнее влияние среды на внутреннюю механику аппарата.
        feedback_strength = peak_drag * damping * 0.15 
        
        # 1. Частота падает при увеличении сопротивления (эффект залипания маятников)
        dyn_freq = max(0.5, frequency - feedback_strength * frequency)
        
        # 2. Асимметрия увеличивается (более резкий срыв после прохождения точки сопротивления)
        dyn_asym = min(1.8, asymmetry + feedback_strength * 0.6)
        
        # Накопление фазы с учетом ТЕКУЩЕЙ частоты (чтобы график не рвался)
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

        rho = np.sum(f, axis=0)
        rho = np.clip(rho, 0.1, 8.0)
        ux = np.sum(f * c[:,0].reshape(9,1,1), axis=0) / (rho + 1e-5)
        uy = np.sum(f * c[:,1].reshape(9,1,1), axis=0) / (rho + 1e-5)

        u[0] = np.clip(ux, -0.32, 0.32)
        u[1] = np.clip(uy, -0.32, 0.32)

        f = f * (1 - omega) + omega * equilibrium(rho, u)

        u[0][solid_mask] = drive_x
        u[1][solid_mask] = drive_y
        rho[solid_mask] = 1.0

        feq = equilibrium(rho, u)
        for i in range(9):
            f[i, solid_mask] = feq[i, solid_mask]

        for i in range(9):
            f[i] = np.roll(f[i], c[i].astype(int), axis=(1,0))

        effective_area = float(np.sum(solid_mask))
        
        drag_energy = abs(0.0068 * v_target * effective_area * abs(np.cos(alpha)))
        current_direction = np.sign(v_target) if abs(v_target) > 1e-5 else 0.0

        if current_direction != last_direction and last_direction != 0:
            release_force = peak_drag * last_direction * 1.4
            peak_drag = 0.0

        if drag_energy > peak_drag:
            peak_drag = drag_energy
        else:
            # 3. Время разрядки. Если частота упала из-за сопротивления,
            # разрядка происходит медленнее (имитация натяжения среды).
            decay_rate = 0.78 + (0.20 * damping * (1.0 - dyn_freq/max(frequency, 0.1)))
            decay_rate = min(0.98, decay_rate) # Ограничитель, чтобы энергия уходила
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
        
        if view_mode == "Vorticity":
            vort = ((np.roll(u[1], -1, axis=1) - np.roll(u[1], 1, axis=1)) * 0.5 -
                    (np.roll(u[0], -1, axis=0) - np.roll(u[0], 1, axis=0)) * 0.5)
            data = np.abs(np.nan_to_num(vort, nan=0.0))
            vmax = float(np.percentile(data, 99.7)) or 1e-5
            vmin = 0.0
        else:
            data = np.abs(np.nan_to_num(rho - 1.0, nan=0.0))
            vmax = float(np.percentile(data, 99.7)) or 1e-5
            vmin = 0.0
        
        data[solid_mask] = np.nan
        im = ax.imshow(data, cmap=cmap_choice, animated=True, vmin=vmin, vmax=vmax, interpolation='bilinear')
        
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

        tx = ax.text(15, 20, telemetry_box, color='#ffffff', fontsize=9, family='monospace',
                     va='top', ha='left', bbox=dict(facecolor='#111111', alpha=0.65,
                                                    edgecolor='#333333', boxstyle='round,pad=0.5'))
        title_text = ax.text(
            N // 2,
            N - 18,
            "OSCILLATORY VORTEX PROPULSION FLYING SAUCER\nCFD SIMULATION",
            color="#d8d8d8",
            fontsize=11,
            family="monospace",
            ha="center",
            va="bottom",
            alpha=0.92
        )

        imgs.append([im, tx])

    ani = animation.ArtistAnimation(fig, imgs, interval=28, blit=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ani.save(tmp.name, fps=25, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)

    return tmp.name

# ============================================================
# ============================================================
with gr.Blocks() as demo:

    
    with gr.Column(elem_classes="border-class"):
        output = gr.Video(label="Simulation Video (1:1)", elem_classes="square-video-box")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Global")
                frames = gr.Slider(minimum=50, maximum=700, value=50, step=50, label="Frames")
                view_mode = gr.Radio(["Vorticity", "Pressure"], value="Vorticity", label="Mode")
                pro_gradient = gr.Checkbox(value=True, label="Multicolor")
                
            with gr.Column():
                gr.Markdown("#### Flight Control")
                attack_angle = gr.Slider(minimum=-45, maximum=45, value=0.0, step=22.5, label="Attack Angle (°)")
                slide_angle = gr.Slider(minimum=-180, maximum=180, value=0.0, step=1.0, label="Thrust Vector (°)")
                buoyancy = gr.Slider(minimum=-0.25, maximum=0.25, value=0.0, step=0.005, label="Buoyancy")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Engine")
                frequency = gr.Slider(minimum=0.5, maximum=8.0, value=3.0, step=0.5, label="Frequency (Hz)")
                amplitude = gr.Slider(minimum=0.5, maximum=6.0, value=4.2, step=0.1, label="Amplitude")
                asymmetry = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Asymmetry")
                
            with gr.Column():
                gr.Markdown("#### Physics")
                gravity = gr.Slider(minimum=0.0, maximum=0.25, value=0.06, step=0.002, label="Gravity")
                tau = gr.Slider(minimum=0.51, maximum=0.95, value=0.58, step=0.01, label="Viscosity (Tau)")
                mass = gr.Slider(minimum=0.1, maximum=2.0, value=0.65, step=0.05, label="Mass")
                # НОВЫЙ ПОЛЗУНОК ДЕМПИНГА
                damping = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Damping")

        btn = gr.Button("RUN SIMULATION", variant="primary")

    btn.click(
        simulate, 
        inputs=[
            frames, view_mode, pro_gradient, attack_angle, slide_angle, 
            buoyancy, frequency, amplitude, asymmetry, gravity, tau, mass, 
            damping # Прокидываем новую переменную
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
            .gradio-container {max-width: 920px !important; margin: 0 auto;}
            .square-video-box {width: 100% !important; max-width: 560px !important; margin: 0 auto !important; aspect-ratio: 1 / 1 !important;}
            .border-class {border: 1px solid #262626; background-color: #1c1c1c; padding: 15px; border-radius: 4px;}
            footer {display: none !important;}
        """
    )