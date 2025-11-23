# tcl_manim_scene.py

from manim import *
from manim import config
import numpy as np
from tcl_config import TAM_VENTANA, ERRORS_FILE, N_STEPS, N_BINS_MEDIAS

# Fondo blanco para toda la escena
config.background_color = WHITE

# ==========================
#   FUNCIONES AUXILIARES
# ==========================

def compute_tcl_data(errores_decision, tam_ventana):
    """Acomoda los errores en ventanas y calcula las medias."""
    n_total = len(errores_decision)
    n_ventanas = n_total // tam_ventana
    n_usadas = n_ventanas * tam_ventana

    errores_recortados = errores_decision[:n_usadas]
    errores_por_ventana = errores_recortados.reshape(n_ventanas, tam_ventana)
    medias_ventana = errores_por_ventana.mean(axis=1)

    return errores_recortados, medias_ventana, n_ventanas, n_usadas


def compute_micro_error_steps(errores_recortados, tam_ventana, n_ventanas, n_steps):
    """
    Para cada paso i genera:
      - p0_steps[i], p1_steps[i] = proporciones de aciertos y errores
      - decisions_steps[i]       = decisiones usadas
      - windows_steps[i]         = ventanas usadas
    """
    n_decisiones = len(errores_recortados)
    p0_steps = []
    p1_steps = []
    decisions_steps = []
    windows_steps = []

    for i in range(1, n_steps + 1):
        ventanas_usadas = max(1, int(n_ventanas * i / n_steps))
        decisiones_usadas = min(ventanas_usadas * tam_ventana, n_decisiones)
        ventanas_usadas = decisiones_usadas // tam_ventana

        datos = errores_recortados[:decisiones_usadas]
        p1 = datos.mean()
        p0 = 1.0 - p1

        p0_steps.append(p0)
        p1_steps.append(p1)
        decisions_steps.append(decisiones_usadas)
        windows_steps.append(ventanas_usadas)

    return p0_steps, p1_steps, decisions_steps, windows_steps


def compute_media_hist_steps(medias_ventana, n_bins, n_steps):
    """
    Devuelve:
      - bins: bordes de bins
      - densities_steps: lista de arrays de densidad por paso
      - max_density: máximo para escalar ejes
    """
    n = len(medias_ventana)
    med_min = float(medias_ventana.min())
    med_max = float(medias_ventana.max())
    if med_min == med_max:
        med_min -= 0.1
        med_max += 0.1

    bins = np.linspace(med_min, med_max, n_bins + 1)
    densities_steps = []
    max_density = 0.0

    for i in range(1, n_steps + 1):
        n_usadas = max(1, int(n * i / n_steps))
        data = medias_ventana[:n_usadas]
        counts, _ = np.histogram(data, bins=bins, density=True)
        densities_steps.append(counts)
        if counts.max() > max_density:
            max_density = counts.max()

    return bins, densities_steps, max_density


def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def create_micro_error_bars(axes, p0, p1, bar_width=0.6, color=BLUE):
    """Crea 2 barras (x=0, x=1) para los micro-errores."""
    bars = VGroup()
    xs = [0, 1]
    hs = [p0, p1]

    for x, h in zip(xs, hs):
        h = max(h, 0)
        rect = Rectangle(
            width=axes.x_axis.unit_size * bar_width,
            height=axes.y_axis.unit_size * h,
            fill_opacity=0.7,
            stroke_width=0,
            color=color,
        )
        rect.move_to(axes.c2p(x, h / 2))
        bars.add(rect)

    return bars


def create_media_hist_bars(axes, bins, densities, color=GREEN):
    """Crea barras de histograma para las medias."""
    bars = VGroup()
    for left, right, h in zip(bins[:-1], bins[1:], densities):
        h = max(h, 0)
        width = right - left
        rect = Rectangle(
            width=axes.x_axis.unit_size * width,
            height=axes.y_axis.unit_size * h,
            fill_opacity=0.7,
            stroke_width=0,
            color=color,
        )
        x_center = (left + right) / 2
        rect.move_to(axes.c2p(x_center, h / 2))
        bars.add(rect)
    return bars


# ==========================
#   ESCENA PRINCIPAL MANIM
# ==========================

class TCLIARealScene(Scene):
    def construct(self):
        # ----------------------------
        # 1. Cargar datos precomputados
        # ----------------------------
        errores_decision = np.load(ERRORS_FILE)
        errores_recortados, medias_ventana, n_ventanas, n_decisiones_usadas = compute_tcl_data(
            errores_decision, TAM_VENTANA
        )

        print(f"Ventanas totales: {n_ventanas}, decisiones usadas: {n_decisiones_usadas}")

        # Parámetros normales teóricos para las medias (TCL)
        mu = medias_ventana.mean()
        sigma = medias_ventana.std()
        if sigma == 0:
            sigma = 1e-6

        # Datos por pasos para animación
        p0_steps, p1_steps, decisions_steps, windows_steps = compute_micro_error_steps(
            errores_recortados, TAM_VENTANA, n_ventanas, N_STEPS
        )

        bins_med, densities_med_steps, max_density = compute_media_hist_steps(
            medias_ventana, N_BINS_MEDIAS, N_STEPS
        )

        # ----------------------------
        # 2. Crear ejes (tipo plano cartesiano)
        # ----------------------------

        # Ejes de micro-errores (0/1)
        axes_err = Axes(
            x_range=[-0.5, 1.5, 1],
            y_range=[0, 1, 0.2],
            x_length=4,
            y_length=3,
            axis_config={"color": BLACK},
            tips=False,
        ).to_edge(LEFT).shift(DOWN * 0.3)

        labels_err = axes_err.get_axis_labels(
            Text("Error", font_size=24, color=BLACK),
            Text("Probabilidad", font_size=24, color=BLACK),
        )

        # Ejes de medias de error
        y_max_med = max_density * 1.2 if max_density > 0 else 1
        med_min = bins_med[0]
        med_max = bins_med[-1]
        step_x_med = (med_max - med_min) / 4 if med_max > med_min else 0.1

        axes_med = Axes(
            x_range=[med_min, med_max, step_x_med],
            y_range=[0, y_max_med, y_max_med / 5],
            x_length=4,
            y_length=3,
            axis_config={"color": BLACK},
            tips=False,
        ).to_edge(RIGHT).shift(DOWN * 0.3)

        labels_med = axes_med.get_axis_labels(
            Text("Error promedio", font_size=24, color=BLACK),
            Text("Densidad", font_size=24, color=BLACK),
        )

        # Título general
        title = Text(
            "Micro-errores de una IA real\n"
            "y el Teorema Central del Límite",
            font_size=30,
            color=BLACK
        ).to_edge(UP)

        # Títulos sobre cada gráfico
        title_err = Text(
            "Micro-errores por decisión\n(0 = acierto, 1 = error)",
            font_size=22,
            color=BLACK
        ).next_to(axes_err, UP, buff=0.3)

        title_med = Text(
            "Distribución de medias de error\n(TCL sobre la IA)",
            font_size=22,
            color=BLACK
        ).next_to(axes_med, UP, buff=0.3)

        # Dibujar todo eso
        self.play(Write(title))
        self.play(
            Create(axes_err), Write(labels_err),
            Create(axes_med), Write(labels_med),
            Write(title_err), Write(title_med),
        )

        # ----------------------------
        # 3. Normal teórica en el gráfico de medias
        # ----------------------------
        normal_curve = axes_med.plot(
            lambda x: normal_pdf(x, mu, sigma),
            x_range=[med_min, med_max],
            color=RED
        )
        normal_label = Text("Normal teórica (TCL)", font_size=18, color=RED)
        normal_label.next_to(axes_med, UR, buff=0.2)

        self.play(Create(normal_curve), FadeIn(normal_label))

        # ----------------------------
        # 4. Barras iniciales (paso 0)
        # ----------------------------
        bars_err = create_micro_error_bars(
            axes_err,
            p0_steps[0],
            p1_steps[0],
            color=BLUE
        )
        bars_med = create_media_hist_bars(
            axes_med,
            bins_med,
            densities_med_steps[0],
            color=GREEN
        )

        info_err = Text(
            f"Decisiones usadas: {decisions_steps[0]}",
            font_size=20,
            color=BLACK
        ).next_to(axes_err, DOWN, buff=0.4)

        info_med = Text(
            f"Ventanas usadas: {windows_steps[0]} / {n_ventanas}",
            font_size=20,
            color=BLACK
        ).next_to(axes_med, DOWN, buff=0.4)

        # Guardamos posiciones para que luego NO se muevan
        info_err_pos = info_err.get_center()
        info_med_pos = info_med.get_center()

        self.play(
            FadeIn(bars_err),
            FadeIn(bars_med),
            Write(info_err),
            Write(info_med),
        )

        # ----------------------------
        # 5. Animar la evolución por pasos
        # ----------------------------
        for i in range(1, N_STEPS):
            new_bars_err = create_micro_error_bars(
                axes_err,
                p0_steps[i],
                p1_steps[i],
                color=BLUE
            )
            new_bars_med = create_media_hist_bars(
                axes_med,
                bins_med,
                densities_med_steps[i],
                color=GREEN
            )

            new_info_err = Text(
                f"Decisiones usadas: {decisions_steps[i]}",
                font_size=20,
                color=BLACK
            ).move_to(info_err_pos)

            new_info_med = Text(
                f"Ventanas usadas: {windows_steps[i]} / {n_ventanas}",
                font_size=20,
                color=BLACK
            ).move_to(info_med_pos)

            self.play(
                Transform(bars_err, new_bars_err),
                Transform(bars_med, new_bars_med),
                Transform(info_err, new_info_err),
                Transform(info_med, new_info_med),
                run_time=0.7
            )

        self.wait(2)
