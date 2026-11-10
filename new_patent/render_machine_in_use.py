from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


OUTPUT_DIR = FilePath(__file__).resolve().parent


def add_capsule(ax, center, width, height, angle_deg=20):
    capsule = FancyBboxPatch(
        (center[0] - width / 2, center[1] - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={width / 2}",
        linewidth=1.5,
        edgecolor="#17415b",
        facecolor="#f3fbff",
        mutation_aspect=1,
    )
    transform = plt.matplotlib.transforms.Affine2D().rotate_deg_around(center[0], center[1], angle_deg) + ax.transData
    capsule.set_transform(transform)
    ax.add_patch(capsule)

    lens = Circle((center[0] - width * 0.18, center[1]), width * 0.12, facecolor="#16324a", edgecolor="white", linewidth=0.8)
    lens.set_transform(transform)
    ax.add_patch(lens)

    for offset, color in [(-0.18, "#ef8c34"), (0.0, "#46a4c7"), (0.18, "#ef8c34")]:
        coil = Ellipse((center[0] + offset * width, center[1]), width * 0.18, height * 0.72, facecolor="none", edgecolor=color, linewidth=1.2)
        coil.set_transform(transform)
        ax.add_patch(coil)


def add_stomach(ax, offset=(0, 0), scale=1.0):
    ox, oy = offset
    verts = np.array([
        [0.00, 0.00],
        [0.10, 0.32],
        [0.32, 0.45],
        [0.48, 0.30],
        [0.56, 0.05],
        [0.48, -0.20],
        [0.28, -0.34],
        [0.10, -0.36],
        [-0.02, -0.16],
        [-0.10, 0.04],
        [0.00, 0.00],
    ])
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    verts = verts * scale + np.array([ox, oy])
    patch = PathPatch(MplPath(verts, codes), facecolor="#f9b7a4", edgecolor="#b55c52", linewidth=2.0, alpha=0.92)
    ax.add_patch(patch)
    pylorus = Ellipse((ox + 0.47 * scale, oy - 0.19 * scale), 0.11 * scale, 0.07 * scale, angle=-25, facecolor="#ef9a8f", edgecolor="#b55c52", linewidth=1.2)
    ax.add_patch(pylorus)


def add_field_arcs(ax, start, end, color, n=5, alpha=0.22):
    x0, y0 = start
    x1, y1 = end
    mid_x = (x0 + x1) / 2
    span = abs(x1 - x0)
    for idx in range(n):
        h = 0.10 + idx * 0.07
        arc = Arc((mid_x, (y0 + y1) / 2), span, h, theta1=8, theta2=172, linewidth=1.6, color=color, alpha=alpha)
        ax.add_patch(arc)


def add_callout(ax, xy, text, color="#20465f", width=0.18):
    box = FancyBboxPatch((xy[0], xy[1]), width, 0.065, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor="white", edgecolor="#c7d7e2", linewidth=1.2)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + 0.0325, text, ha="center", va="center", fontsize=10, color=color, fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x = np.linspace(0, 1, 800)
    y = np.linspace(0, 1, 450)
    xv, yv = np.meshgrid(x, y)
    bg = 0.78 + 0.18 * (1 - yv) + 0.04 * np.sin(6 * xv)
    img = np.dstack((0.92 * bg, 0.96 * bg, 1.00 * bg))
    ax.imshow(np.clip(img, 0, 1), extent=[0, 1, 0, 1], origin="lower")

    floor = Rectangle((0, 0), 1, 0.17, facecolor="#dfe8ee", edgecolor="none", alpha=0.9)
    ax.add_patch(floor)

    ax.text(0.05, 0.92, "Wireless Magnetic Capsule Endoscopy System", fontsize=24, color="#12344a", fontweight="bold")
    ax.text(0.05, 0.875, "Concept render of the full machine in use for stomach examination", fontsize=13, color="#426579")

    bed_shadow = FancyBboxPatch((0.14, 0.22), 0.56, 0.10, boxstyle="round,pad=0.0,rounding_size=0.04", facecolor="#95aab8", edgecolor="none", alpha=0.35)
    ax.add_patch(bed_shadow)
    bed = FancyBboxPatch((0.12, 0.24), 0.56, 0.10, boxstyle="round,pad=0.0,rounding_size=0.04", facecolor="#f6f7f8", edgecolor="#9aaab4", linewidth=1.5)
    ax.add_patch(bed)
    pillow = FancyBboxPatch((0.145, 0.47), 0.12, 0.08, boxstyle="round,pad=0.0,rounding_size=0.03", facecolor="#fbfcfd", edgecolor="#c4d0d8", linewidth=1.2)
    ax.add_patch(pillow)

    head = Circle((0.26, 0.53), 0.055, facecolor="#e8b79f", edgecolor="#c58a72", linewidth=1.2)
    ax.add_patch(head)
    hair = Ellipse((0.248, 0.555), 0.09, 0.06, angle=10, facecolor="#55423a", edgecolor="none", alpha=0.95)
    ax.add_patch(hair)
    neck = Rectangle((0.292, 0.46), 0.03, 0.045, facecolor="#e3b098", edgecolor="none")
    ax.add_patch(neck)

    torso = FancyBboxPatch((0.30, 0.38), 0.26, 0.17, boxstyle="round,pad=0.01,rounding_size=0.05", facecolor="#74a9c8", edgecolor="#4a7f9b", linewidth=1.8)
    ax.add_patch(torso)
    pelvis = FancyBboxPatch((0.54, 0.36), 0.11, 0.13, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor="#3d5268", edgecolor="#2e4358", linewidth=1.4)
    ax.add_patch(pelvis)
    leg1 = Rectangle((0.635, 0.355), 0.18, 0.065, angle=-6, facecolor="#495d73", edgecolor="#344859", linewidth=1.2)
    leg2 = Rectangle((0.63, 0.295), 0.18, 0.058, angle=-2, facecolor="#54677a", edgecolor="#344859", linewidth=1.2)
    ax.add_patch(leg1)
    ax.add_patch(leg2)
    arm = Rectangle((0.39, 0.30), 0.20, 0.04, angle=-12, facecolor="#e7b59d", edgecolor="#c58a72", linewidth=1.0)
    ax.add_patch(arm)

    torso_window = FancyBboxPatch((0.345, 0.405), 0.165, 0.125, boxstyle="round,pad=0.0,rounding_size=0.04", facecolor="#eef6fb", edgecolor="#8cb1c7", linewidth=1.2, alpha=0.88)
    ax.add_patch(torso_window)
    add_stomach(ax, offset=(0.405, 0.465), scale=0.12)
    add_capsule(ax, center=(0.445, 0.47), width=0.048, height=0.018, angle_deg=28)

    coil_left = FancyBboxPatch((0.05, 0.28), 0.08, 0.30, boxstyle="round,pad=0.01,rounding_size=0.035", facecolor="#263f52", edgecolor="#173041", linewidth=1.8)
    coil_right = FancyBboxPatch((0.70, 0.28), 0.08, 0.30, boxstyle="round,pad=0.01,rounding_size=0.035", facecolor="#263f52", edgecolor="#173041", linewidth=1.8)
    ax.add_patch(coil_left)
    ax.add_patch(coil_right)
    for x0 in [0.09, 0.74]:
        for r in [0.018, 0.028, 0.038]:
            ax.add_patch(Circle((x0, 0.43), r, facecolor="none", edgecolor="#9ed2e6", linewidth=1.4, alpha=0.9))

    mount_left = Rectangle((0.082, 0.19), 0.015, 0.09, facecolor="#a6b6c1", edgecolor="none")
    mount_right = Rectangle((0.732, 0.19), 0.015, 0.09, facecolor="#a6b6c1", edgecolor="none")
    ax.add_patch(mount_left)
    ax.add_patch(mount_right)

    add_field_arcs(ax, (0.13, 0.43), (0.70, 0.43), "#4aa0c8", n=5, alpha=0.25)
    add_field_arcs(ax, (0.13, 0.40), (0.70, 0.40), "#ef8c34", n=4, alpha=0.15)

    ctrl_body = FancyBboxPatch((0.80, 0.19), 0.13, 0.30, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#f7fafc", edgecolor="#b6c8d3", linewidth=1.6)
    ax.add_patch(ctrl_body)
    screen = FancyBboxPatch((0.82, 0.37), 0.09, 0.085, boxstyle="round,pad=0.008,rounding_size=0.012", facecolor="#173041", edgecolor="#20465f", linewidth=1.2)
    ax.add_patch(screen)
    ax.plot([0.828, 0.846, 0.868, 0.888], [0.40, 0.42, 0.395, 0.435], color="#8de0ff", linewidth=2.0)
    ax.text(0.865, 0.332, "Power +\nfield controller", ha="center", va="center", fontsize=12, color="#26485d", fontweight="bold")
    ax.add_patch(Circle((0.835, 0.25), 0.012, facecolor="#ef8c34", edgecolor="none"))
    ax.add_patch(Circle((0.875, 0.25), 0.012, facecolor="#4aa0c8", edgecolor="none"))
    ax.add_patch(Circle((0.915, 0.25), 0.012, facecolor="#6e8ea2", edgecolor="none"))

    cable_color = "#728999"
    ax.plot([0.80, 0.77, 0.77, 0.745], [0.33, 0.33, 0.43, 0.43], color=cable_color, linewidth=2.4)
    ax.plot([0.80, 0.76, 0.76, 0.095], [0.29, 0.29, 0.15, 0.15], color=cable_color, linewidth=2.1)
    ax.plot([0.095, 0.095, 0.09], [0.15, 0.28, 0.28], color=cable_color, linewidth=2.1)

    arrow = FancyArrowPatch((0.41, 0.73), (0.46, 0.55), arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#20465f")
    ax.add_patch(arrow)
    add_callout(ax, (0.34, 0.73), "Wireless capsule", width=0.16)

    arrow2 = FancyArrowPatch((0.015, 0.68), (0.09, 0.54), arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#20465f")
    ax.add_patch(arrow2)
    add_callout(ax, (0.02, 0.69), "Left field coil", width=0.15)

    arrow3 = FancyArrowPatch((0.76, 0.66), (0.74, 0.54), arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#20465f")
    ax.add_patch(arrow3)
    add_callout(ax, (0.75, 0.67), "Right field coil", width=0.17)

    arrow4 = FancyArrowPatch((0.77, 0.56), (0.84, 0.46), arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#20465f")
    ax.add_patch(arrow4)
    add_callout(ax, (0.77, 0.57), "Drive electronics", width=0.18)

    ax.text(0.05, 0.08, "Shared external modules provide coarse magnetic bias and high-frequency wireless power. Internal 3-axis coils steer, roll, and orient the capsule along the stomach wall.", fontsize=12.5, color="#365769")

    png_path = OUTPUT_DIR / "wireless_capsule_system_in_use.png"
    svg_path = OUTPUT_DIR / "wireless_capsule_system_in_use.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"CREATED: {png_path}")
    print(f"CREATED: {svg_path}")


if __name__ == "__main__":
    main()