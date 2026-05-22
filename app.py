import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gradio as gr
import tempfile
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# GRID AND LBM CONSTANTS
# ============================================================
N = 500
X, Y = np.meshgrid(np.arange(N), np.arange(N))

c = np.array([
    [0,0], [1,0], [0,1], [-1,0], [0,-1],
    [1,1], [-1,1], [-1,-1], [1,-1]
], dtype=float)

w = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=float)

# ============================================================
# PROFESSIONAL MULTICOLOR CFD PALETTE
# ============================================================

cfd_multicolor_cmap = LinearSegmentedColormap.from_list(
    "cfd_multicolor",
    [
        (0.00, "#a8ff60"),   # light lime background
        (0.10, "#00b7ff"),   # blue (ранний переход)
        (0.25, "#00ffd0"),   # cyan
        (0.40, "#00ff66"),   # green
        (0.60, "#dfff00"),   # yellow (расширяем зону)
        (0.85, "#ff9a00"),   # orange (расширяем зону)
        (1.00, "#ff2200")    # red (только для жестких пиков)
    ],
    N=512
)



def get_saucer_mask_and_velocity(cx, cy, v_body):
    X_c = X - cx
    Y_c = Y - cy

    a, b = 28.0, 6.0

    mask = (X_c**2)/(a*a) + (Y_c**2)/(b*b) < 1.0

    u_solid_x = np.zeros_like(X, dtype=float)
    u_solid_y = np.full_like(X, v_body, dtype=float)

    return mask, u_solid_x, u_solid_y

def equilibrium(rho, u):
    feq = np.zeros((9, N, N), dtype=float)

    usqr = u[0]**2 + u[1]**2

    for i in range(9):
        cu = 3 * (c[i,0] * u[0] + c[i,1] * u[1])

        feq[i] = rho * w[i] * (
            1 + cu + 0.5*cu**2 - 1.5*usqr
        )

    return feq

# ============================================================
# MAIN SIMULATION ENGINE
# ============================================================
def simulate(
    frames=650,
    view_mode="Vorticity",
    mass=1.0,
    frequency=3.0,
    amplitude=3.8,
    asymmetry=0.0,
    tau=0.58,
    pro_gradient=True
):

    omega = 1.0 / tau

    rho = np.ones((N, N), dtype=float)
    u = np.zeros((2, N, N), dtype=float)

    f = equilibrium(rho, u)

    cx = float(N // 2)
    cy = float(N * 0.5)

    v_body = 0.0

    peak_energy_up = 0.0
    peak_energy_down = 0.0

    stored_impulse_up = 0.0
    stored_impulse_down = 0.0

    release_force = 0.0

    last_v_target = 0.0

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#121212')

    ax.set_facecolor('#121212')
    ax.axis('off')

    fig.tight_layout()

    imgs = []

    # ============================================================
    # COLOR MODES
    # ============================================================
    if pro_gradient:
        # MULTICOLOR CFD MODE
        cmap_choice = cfd_multicolor_cmap
    else:
        # CLASSIC RED/BLUE SIGNED MODE
        cmap_choice = 'RdBu_r'

    for t in range(frames):

        phase = t * 2 * np.pi * frequency / 60.0

        v_base = -amplitude * np.sin(phase) * (frequency / 30.0)

        if asymmetry > 0.0:
            distortion = asymmetry * np.cos(phase)
            v_target = v_base * (1.0 + distortion)
        else:
            v_target = v_base

        mask, u_solid_x, u_solid_y = get_saucer_mask_and_velocity(
            cx,
            cy,
            v_body
        )

        u_solid_y[mask] += v_target

        rho = np.sum(f, axis=0)
        rho = np.clip(rho, 0.1, 8.0)

        ux = np.sum(
            f * c[:,0].reshape(9,1,1),
            axis=0
        ) / (rho + 1e-5)

        uy = np.sum(
            f * c[:,1].reshape(9,1,1),
            axis=0
        ) / (rho + 1e-5)

        u[0] = np.clip(ux, -0.25, 0.25)
        u[1] = np.clip(uy, -0.25, 0.25)

        f = f * (1 - omega) + omega * equilibrium(rho, u)

        u[0][mask] = u_solid_x[mask]
        u[1][mask] = u_solid_y[mask]

        rho[mask] = 1.0

        feq = equilibrium(rho, u)

        for i in range(9):
            f[i, mask] = feq[i, mask]

        for i in range(9):
            f[i] = np.roll(
                f[i],
                c[i].astype(int),
                axis=(1,0)
            )

                # ============================================================
        # INERTIOID DYNAMICS — ПРАВИЛЬНОЕ НАПРАВЛЕНИЕ
        # ============================================================
        
        drag_energy = abs(0.006 * v_target * np.sum(mask))

        current_direction = np.sign(v_target) if v_target != 0 else 0.0

        if 'peak_drag' not in locals():
            peak_drag = 0.0
            last_direction = 0.0
            release_force = 0.0

        # Смена направления — высвобождаем импульс в сторону ПРЕДЫДУЩЕГО рывка
        if current_direction != last_direction and last_direction != 0:
            release_force = peak_drag * last_direction * 1.25   # ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
            peak_drag = 0.0

        # Накопление в текущей фазе
        if drag_energy > peak_drag:
            peak_drag = drag_energy

        # Плавное затухание
        else:
            release_force *= 0.83

        last_direction = current_direction

        # Применяем силу
        v_body += release_force / mass * 0.15
        cy += v_body * 0.38

        cy = np.clip(cy, 60, N - 60)
                
        
                
        
        
        
       
             
         

        # ============================================================
        # FLOW VISUALIZATION
        # ============================================================
        # ============================================================
        # FLOW VISUALIZATION
        # ============================================================
        # ============================================================
        # FLOW VISUALIZATION (DYNAMIC PERCENTILE SCALING)
        # ============================================================
        if view_mode == "Vorticity":

            vort = (
                (np.roll(u[1], -1, axis=1) -
                 np.roll(u[1], 1, axis=1)) * 0.5
                -
                (np.roll(u[0], -1, axis=0) -
                 np.roll(u[0], 1, axis=0)) * 0.5
            )

            if pro_gradient:
                data = np.abs(np.nan_to_num(vort, nan=0.0))
                
                # Отсекаем 0.3% самых жестких граничных артефактов
                vmax = float(np.percentile(data, 99.7))
                if vmax <= 0: 
                    vmax = 1e-5
                vmin = 0.0

            else:
                data = np.nan_to_num(vort, nan=0.0)
                vmin, vmax = -0.028, 0.028

        else:

            pressure = np.nan_to_num(
                rho - 1.0,
                nan=0.0
            )

            if pro_gradient:
                data = np.abs(pressure)
                
                # То же самое для давления
                vmax = float(np.percentile(data, 99.7))
                if vmax <= 0: 
                    vmax = 1e-5
                vmin = 0.0

            else:
                data = pressure
                vmin, vmax = -0.012, 0.012
                
        
            
                

        data[mask] = 0

        im = ax.imshow(
            data,
            cmap=cmap_choice,
            animated=True,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )

        telemetry_box = (
            f"TELEMETRY\n"
            f"---------------------\n"
            f"FREQ     : {frequency:.1f} Hz\n"
            f"AMP      : {amplitude:.1f}\n"
            f"ASYM     : {asymmetry:.2f}\n"
            f"MASS     : {mass:.1f}\n"
            f"COORD Y  : {cy:.1f}\n"
            f"VELOCITY : {v_body:+.4f}\n"
            f"PULSE +  : {stored_impulse_up:.2f}\n"
            f"PULSE -  : {stored_impulse_down:.2f}"
        )

        tx = ax.text(
            15,
            20,
            telemetry_box,
            color='#ffffff',
            fontsize=9,
            family='monospace',
            va='top',
            ha='left',
            bbox=dict(
                facecolor='#111111',
                alpha=0.65,
                edgecolor='#333333',
                boxstyle='round,pad=0.5'
            )
        )

        imgs.append([im, tx])

    ani = animation.ArtistAnimation(
        fig,
        imgs,
        interval=28,
        blit=False
    )

    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )

    ani.save(
        tmp.name,
        fps=25,
        extra_args=['-vcodec', 'libx264']
    )

    plt.close(fig)

    return tmp.name

# ============================================================
# GRADIO INTERFACE
# ============================================================
with gr.Blocks(
    theme=gr.themes.Base(),
    css="""
    body {
        background-color: #121212;
        color: #d0d0d0;
        font-family: monospace;
    }

    .gradio-container {
        max-width: 900px !important;
        margin: 0 auto;
    }

    h1, gr.Markdown {
        color: #ffffff;
        font-weight: 400;
    }

    :root {
        --primary-50: #262626 !important;
        --primary-100: #333333 !important;
        --primary-200: #404040 !important;
        --primary-300: #525252 !important;
        --primary-400: #737373 !important;
        --primary-500: #8e8e8e !important;
        --primary-600: #a3a3a3 !important;
        --primary-700: #d4d4d4 !important;
        --primary-800: #e5e5e5 !important;
        --primary-900: #ffffff !important;
    }

    input[type="range"] {
        accent-color: #737373 !important;
    }

    .square-video-box {
        width: 100% !important;
        max-width: 550px !important;
        margin: 0 auto !important;
        aspect-ratio: 1 / 1 !important;
    }

    .border-class {
        border: 1px solid #262626;
        background-color: #1c1c1c;
        padding: 15px;
        border-radius: 4px;
    }

    footer {
        display: none !important;
    }
"""
) as demo:

    #gr.Markdown("# FLYING SAUCER SIMULATOR")
    #gr.Markdown("### Experimental aeroacoustic/vibration-based airship")

    with gr.Column(elem_classes="border-class"):

        output = gr.Video(
            label="Simulation Video (1:1)",
            elem_classes="square-video-box"
        )

        with gr.Row():

            with gr.Column(scale=1, min_width=280):

                gr.Markdown("#### Global Parameters")

                frames = gr.Slider(
                    50,
                    1000,
                    50,
                    step=50,
                    label="Simulation Frames"
                )

                view_mode = gr.Radio(
                    ["Vorticity", "Pressure"],
                    value="Vorticity",
                    label="Analysis Mode"
                )

                pro_gradient = gr.Checkbox(
                    value=True,
                    label="Multicolor CFD Mode"
                )

                mass = gr.Slider(
                    0.1,
                    1.0,
                    1.0,
                    step=0.1,
                    label="Object Mass"
                )

            with gr.Column(scale=1, min_width=280):

                gr.Markdown("#### Dynamic Waveform")

                frequency = gr.Slider(
                    0.5,
                    8.0,
                    3.0,
                    step=0.5,
                    label="Frequency (Hz)"
                )

                amplitude = gr.Slider(
                    0.5,
                    6.0,
                    6.0,
                    step=0.1,
                    label="Oscillation Amplitude"
                )

                asymmetry = gr.Slider(
                    0.0,
                    1.0,
                    1.0,
                    step=0.05,
                    label="Velocity Profile Asymmetry"
                )

                tau = gr.Slider(
                    0.51,
                    0.95,
                    0.58,
                    step=0.01,
                    label="Lattice Relaxation (Tau)"
                )

        btn = gr.Button(
            "RUN SIMULATION",
            variant="primary"
        )

    btn.click(
        simulate,
        inputs=[
            frames,
            view_mode,
            mass,
            frequency,
            amplitude,
            asymmetry,
            tau,
            pro_gradient
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )