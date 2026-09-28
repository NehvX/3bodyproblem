"""
3-BODY VISUALISER
=================
Replays any case from the search results as an animation. It re-runs the
exact same physics engine from the saved starting conditions, so what you
see is what the search saw.

    python visualize_case.py 42                     animate case 42
    python visualize_case.py custom_1               animate a custom-mode case
    python visualize_case.py --preset figure8       animate a famous example
    python visualize_case.py --custom               animate the custom case in settings.json
    python visualize_case.py --list                 list the high-interest cases
    python visualize_case.py 42 --save case42.gif   save instead of showing (.gif, .mp4 or .png)

While the window is open: SPACE = pause / play.
"""

import argparse
import csv
import os
import sys

import numpy as np

import three_body_search as tb

DEFAULT_WINDOW = 100_000      # steps shown by default for long cases
BG = "#0b0f1a"
FG = "#d8dee9"
GRID = "#2a3142"
COLORS = ["#4fc3f7", "#ffb74d", "#ef5350"]


# ---------------------------------------------------------------------------
# finding and preparing a case
# ---------------------------------------------------------------------------

def find_case(case_id, s):
    """Look for the case in the high-interest, all-cases and custom CSV files."""
    for name in (s["high_priority_file"], s["all_cases_file"], s["custom_cases_file"]):
        path = tb.in_base(name)
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("case_id") == case_id:
                    return row, path
    return None, None


def setup_from_row(row, s):
    """Settings + starting positions/velocities for a CSV row."""
    s = dict(s)
    s["masses"] = [float(row[k]) for k in ("m1", "m2", "m3")]
    for key in ("G", "dt", "softening"):
        if row.get(key):
            s[key] = float(row[key])
    if row.get("integrator"):
        s["integrator"] = row["integrator"]
    if row.get("timestep"):
        parts = row["timestep"].split()
        s["timestep"] = parts[0]
        if len(parts) > 1:
            s["adaptive_accuracy"] = float(parts[1])
    pos = [[float(row[f"p{b}_{c}"]) for c in "xyz"] for b in (1, 2, 3)]
    vel = [[float(row[f"v{b}_{c}"]) for c in "xyz"] for b in (1, 2, 3)]

    if "dt" not in row:
        # A file from the ORIGINAL script: it used fixed steps and rounded the
        # starting numbers to 8 decimals, which is not exact enough for a chaotic
        # system - so rebuild the exact start from the case number instead.
        s["timestep"] = "fixed"
        if row["case_id"].isdigit():
            pos, vel = tb.random_initial_conditions(int(row["case_id"]), tb.DEFAULTS)
        print("Note: this row comes from the original script's CSV format.")
    return s, np.array(pos), np.array(vel)


def list_cases(s, limit=40):
    path = tb.in_base(s["high_priority_file"])
    if not os.path.exists(path):
        print(f"No {os.path.basename(path)} yet - run three_body_search.py first.")
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No high-interest cases found yet.")
        return
    rows.sort(key=lambda r: int(r["survival_steps"]), reverse=True)
    print(f"\n{'case':>10}  {'interest':<10} {'result':<16} {'steps':>10}  {'energy err':>10}")
    print("-" * 64)
    for r in rows[:limit]:
        err = r.get("energy_error", "")
        err = f"{float(err):.1e}" if err else "-"
        print(f"{r['case_id']:>10}  {r['interest_level']:<10} {r['end_reason']:<16} "
              f"{int(r['survival_steps']):>10,}  {err:>10}")
    if len(rows) > limit:
        print(f"... and {len(rows) - limit} more (showing the longest-lived {limit})")
    print("\nWatch one with:  python visualize_case.py <case>")


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def style_axes(ax, three_d):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    if three_d:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0, 0, 0, 0))
            axis._axinfo["grid"]["color"] = GRID
            axis.label.set_color(FG)
    else:
        ax.grid(color=GRID, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)


def build_figure(plt, P, times, steps_axis, dist, s, info, end_reason, args):
    """Draw the figure. Returns it plus update(k), which moves everything to frame k."""
    three_d = not args.two_d
    n = len(P)
    masses = s["masses"]

    fig = plt.figure(figsize=(13, 7.5), facecolor=BG)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], height_ratios=[1, 1.1],
                            left=0.03, right=0.97, top=0.92, bottom=0.07, wspace=0.12, hspace=0.3)
    ax = fig.add_subplot(grid[:, 0], projection="3d" if three_d else None)
    axd = fig.add_subplot(grid[0, 1])
    axi = fig.add_subplot(grid[1, 1])
    style_axes(ax, three_d)
    style_axes(axd, False)
    axi.axis("off")

    # how much space to show: everything, but not further than a bit past the ejection distance
    lim = args.zoom or min(np.abs(P).max() * 1.1, s["ejection_dist"] * 1.2)
    lim = max(lim, 1e-3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if three_d:
        ax.set_zlim(-lim, lim)
        ax.set_zlabel("z")
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect("equal")

    biggest = max(masses)
    trails, dots = [], []
    for b in range(3):
        size = 5 + 9 * (masses[b] / biggest) ** (1 / 3)
        label = f"Body {b + 1}  (m = {masses[b]:.4g})"
        if three_d:
            trail, = ax.plot([], [], [], "-", color=COLORS[b], lw=1.1, alpha=0.75)
            dot, = ax.plot([], [], [], "o", color=COLORS[b], ms=size, label=label,
                           markeredgecolor="white", markeredgewidth=0.6)
        else:
            trail, = ax.plot([], [], "-", color=COLORS[b], lw=1.1, alpha=0.75)
            dot, = ax.plot([], [], "o", color=COLORS[b], ms=size, label=label,
                           markeredgecolor="white", markeredgewidth=0.6)
        trails.append(trail)
        dots.append(dot)
    legend = ax.legend(loc="upper left", fontsize=9, facecolor=BG, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(FG)

    # distance between each pair over time
    pair_names = ["1-2", "1-3", "2-3"]
    pair_colors = ["#b39ddb", "#80cbc4", "#f48fb1"]
    for k in range(3):
        axd.plot(times, np.maximum(dist[:, k], 1e-6), color=pair_colors[k], lw=1, label=f"bodies {pair_names[k]}")
    if s["collision_dist"] > 0:
        axd.axhline(s["collision_dist"], color="#ef5350", ls="--", lw=1, label="collision distance")
    axd.set_yscale("log")
    axd.set_xlim(times[0], times[-1] if times[-1] > times[0] else times[0] + 1)
    axd.set_title("Distance between each pair", color=FG, fontsize=10)
    axd.set_xlabel("time")
    dl = axd.legend(fontsize=7, facecolor=BG, edgecolor=GRID, loc="upper right")
    for text in dl.get_texts():
        text.set_color(FG)
    cursor = axd.axvline(times[0], color="white", lw=1, alpha=0.8)

    axi.text(0, 1, "\n".join(info), va="top", ha="left", color=FG, fontsize=9.5,
             family="monospace", transform=axi.transAxes)
    title = fig.suptitle("", color=FG, fontsize=13)
    event_text = fig.text(0.36, 0.035, "", color="#ff5252", fontsize=20, ha="center",
                          weight="bold")
    header = info[0]

    def update(k):
        lo = 0 if args.trail <= 0 else max(0, k - args.trail)
        for b in range(3):
            trails[b].set_data(P[lo:k + 1, b, 0], P[lo:k + 1, b, 1])
            dots[b].set_data([P[k, b, 0]], [P[k, b, 1]])
            if three_d:
                trails[b].set_3d_properties(P[lo:k + 1, b, 2])
                dots[b].set_3d_properties([P[k, b, 2]])
        cursor.set_xdata([times[k], times[k]])
        title.set_text(f"{header}      t = {times[k]:.2f}   (step {steps_axis[k]:,})")
        event_text.set_text(end_reason.upper() + "!"
                            if k == n - 1 and end_reason in ("Collision", "Ejection") else "")
        if three_d and not args.no_rotate:
            ax.view_init(elev=22, azim=35 + 0.25 * k)
        return trails + dots + [cursor, title, event_text]

    return fig, update


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate a 3-body case from the search results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python visualize_case.py --list\n"
               "  python visualize_case.py 42\n"
               "  python visualize_case.py 42 --2d --trail 0\n"
               "  python visualize_case.py 42 --full --frames 3000\n"
               "  python visualize_case.py --preset figure8 --save figure8.gif")
    parser.add_argument("case", nargs="?", help="case id, e.g. 42 or custom_1")
    parser.add_argument("--list", action="store_true", help="list the high-interest cases")
    parser.add_argument("--preset", choices=sorted(tb.PRESETS), help="show a famous example")
    parser.add_argument("--custom", action="store_true", help="show the custom case from settings.json")
    parser.add_argument("--settings", default="settings.json", help="settings file (default settings.json)")
    parser.add_argument("--start", type=int, help="first step to show")
    parser.add_argument("--end", type=int, help="last step to show")
    parser.add_argument("--full", action="store_true", help="show the whole run from step 0")
    parser.add_argument("--frames", type=int, help="number of animation frames (default 1500, or 300 when saving)")
    parser.add_argument("--trail", type=int, default=150, help="trail length in frames (0 = whole path)")
    parser.add_argument("--interval", type=int, default=25, help="milliseconds per frame (smaller = faster)")
    parser.add_argument("--2d", dest="two_d", action="store_true", help="flat top-down (x-y) view")
    parser.add_argument("--no-rotate", action="store_true", help="don't slowly rotate the 3D camera")
    parser.add_argument("--no-com", action="store_true", help="don't keep the centre of mass fixed at the middle")
    parser.add_argument("--zoom", type=float, help="half-width of the view (e.g. 3)")
    parser.add_argument("--save", metavar="FILE", help="save to .gif / .mp4 (animation) or .png (picture)")
    parser.add_argument("--fps", type=int, default=30, help="frames per second when saving (default 30)")
    args = parser.parse_args()

    try:
        s = tb.load_settings(tb.in_base(args.settings))
    except tb.SettingsError as e:
        sys.exit(f"SETTINGS PROBLEM: {e}")

    if args.list:
        list_cases(s)
        return

    row = None
    if args.preset:
        s.update(tb.PRESETS[args.preset])
        label = f"Preset: {args.preset}"
        pos, vel = np.array(s["custom_positions"]), np.array(s["custom_velocities"])
    elif args.custom:
        label = "Custom case (settings.json)"
        pos, vel = np.array(s["custom_positions"]), np.array(s["custom_velocities"])
    elif args.case:
        row, path = find_case(args.case, s)
        if row is None:
            sys.exit(f"Case '{args.case}' was not found in the CSV files. Try --list.")
        label = f"Case {args.case}"
        s, pos, vel = setup_from_row(row, s)
    else:
        parser.print_help()
        print()
        list_cases(s, limit=10)
        return

    # 1) run once without recording to find out how (and when) it ends
    tb.warm_up_engine()
    result = tb.run_simulation(pos, vel, s)[0]
    end_step = result["steps"]
    if end_step == 0:           # e.g. an unbound case that the search skipped
        end_step = min(20_000, s["long_test_steps"])
        s = dict(s, collision_dist=0.0)

    # 2) choose which part to show
    end = min(args.end or end_step, end_step)
    if args.full:
        start = 0
    elif args.start is not None:
        start = max(0, min(args.start, end - 1))
    elif end > DEFAULT_WINDOW and result["end_reason"] != "Survived":
        start = end - DEFAULT_WINDOW          # show the ending (the collision / ejection)
    else:
        start = 0
        if end > DEFAULT_WINDOW and args.end is None:
            end = DEFAULT_WINDOW
    frames_wanted = args.frames or (300 if args.save and not args.save.lower().endswith(".png") else 1500)
    record_every = max(1, (end - start) // frames_wanted)

    # 3) run again, this time recording positions for the animation
    replay, P, _, _ = tb.run_simulation(pos, vel, s, max_steps=end, record_every=record_every,
                                        record_start=start, max_frames=frames_wanted + 5)
    if len(P) < 2:
        sys.exit("Nothing to animate in that range - try --full.")
    steps_axis = np.minimum(start + np.arange(len(P)) * record_every, replay["steps"])
    steps_axis[-1] = replay["steps"]
    times = steps_axis * s["dt"]

    m = np.array(s["masses"])
    if not args.no_com:
        com = (P * m[None, :, None]).sum(axis=1) / m.sum()
        P = P - com[:, None, :]
    dist = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in ((0, 1), (0, 2), (1, 2))], axis=1)

    shown = f"steps {start:,} - {replay['steps']:,}"
    info = [
        label,
        f"Result:        {result['end_reason']} {result['detail']}".rstrip(),
        f"Survived:      {result['steps']:,} steps (t = {result['time']:.4g})",
        f"Interest:      {tb.interest_level(result['steps'], result['end_reason'], s)}",
        f"Closest pass:  {result['min_distance']:.3g}",
        f"Energy error:  {result['energy_error']:.1e}",
        "",
        f"Masses:        {', '.join(f'{x:.4g}' for x in s['masses'])}",
        f"Integrator:    {s['integrator']}, {s['timestep']} step",
        f"dt:            {s['dt']}   softening: {s['softening']}",
        f"Showing:       {shown}",
    ]
    if row is not None and row.get("end_reason") and row["end_reason"] != result["end_reason"]:
        info.append(f"(CSV said {row['end_reason']} - settings differ?)")
    print("\n".join(info))
    if start > 0 or replay["steps"] < result["steps"]:
        print("Tip: use --full, or --start / --end, to choose which part to watch.")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    # the event banner only appears if the shown window reaches the actual end
    end_reason = result["end_reason"] if replay["steps"] == result["steps"] else ""
    fig, update = build_figure(plt, P, times, steps_axis, dist, s, info, end_reason, args)

    ext = os.path.splitext(args.save)[1].lower() if args.save else ""
    if args.save and ext not in (".png", ".gif", ".mp4"):
        sys.exit("--save must end in .gif, .mp4 or .png")
    if ext == ".png":
        args.trail = 0                       # a still picture shows the whole path
        update(len(P) - 1)
        fig.savefig(args.save, dpi=120, facecolor=BG)
        print(f"Saved {os.path.abspath(args.save)}")
        return

    anim = FuncAnimation(fig, update, frames=len(P), interval=args.interval, blit=False,
                         repeat=True, repeat_delay=1500)
    if not args.save:
        paused = {"on": False}

        def on_key(event):
            if event.key == " ":
                (anim.resume if paused["on"] else anim.pause)()
                paused["on"] = not paused["on"]

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        return

    if ext == ".gif":
        writer, dpi = PillowWriter(fps=args.fps), 60
    elif FFMpegWriter.isAvailable():
        writer, dpi = FFMpegWriter(fps=args.fps, bitrate=2400), 100
    else:
        sys.exit("Saving .mp4 needs ffmpeg installed. Save as .gif instead.")

    def progress(i, total):
        if i % max(1, total // 10) == 0:
            print(f"  saving... {100 * i // total}%", flush=True)

    anim.save(args.save, writer=writer, dpi=dpi, progress_callback=progress,
              savefig_kwargs={"facecolor": BG})
    print(f"Saved {os.path.abspath(args.save)}")


if __name__ == "__main__":
    main()
