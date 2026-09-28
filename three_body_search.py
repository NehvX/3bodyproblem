"""
3-BODY ORBIT SEARCH  (optimised version)
========================================
Tries random starting conditions for three bodies that pull on each other with
gravity, and records how long each system lasts before two bodies collide or
one of them is thrown out. Long-lived systems are saved as "high interest".

    python three_body_search.py                  run with settings.json
    python three_body_search.py --setup          change settings in a menu first
    python three_body_search.py --help           all options

Settings:      PARAMETER_GUIDE.txt
How it works:  EXPLANATION.txt
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # the program still works without numba, just much slower
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def in_base(filename):
    """Relative file names are kept next to this script, wherever you run it from."""
    return filename if os.path.isabs(filename) else os.path.join(BASE_DIR, filename)


# ---------------------------------------------------------------------------
# PARAMETERS  (defaults = the values from the original script)
# ---------------------------------------------------------------------------

FIGURE8_POSITIONS = [[-0.97000436, 0.24308753, 0.0],
                     [0.97000436, -0.24308753, 0.0],
                     [0.0, 0.0, 0.0]]
FIGURE8_VELOCITIES = [[0.46620368, 0.43236573, 0.0],
                      [0.46620368, 0.43236573, 0.0],
                      [-0.93240737, -0.86473146, 0.0]]


def _param(name, default, kind, group, help_text):
    return {"name": name, "default": default, "kind": kind, "group": group, "help": help_text}


PARAMETERS = [
    _param("mode", "random", ("random", "custom"), "What to run",
           "random = search random starting conditions, case after case. "
           "custom = simulate ONE system using custom_positions / custom_velocities."),

    _param("G", 1.0, "pos_float", "Physics",
           "Gravitational constant. 1.0 = simple simulation units."),
    _param("masses", [1.007825, 4.002603, 15.994915], "masses", "Physics",
           "Masses of body 1, body 2 and body 3."),
    _param("dt", 0.001, "pos_float", "Physics",
           "Time step. Smaller = more accurate but slower."),
    _param("softening", 0.01, "nonneg_float", "Physics",
           "Stops gravity becoming infinite when two bodies almost touch."),
    _param("integrator", "rk4", ("rk4", "leapfrog"), "Physics",
           "rk4 = accurate 4th-order method (original). "
           "leapfrog = ~4x less CPU per step, keeps energy steady on long runs."),
    _param("timestep", "adaptive", ("adaptive", "fixed"), "Physics",
           "adaptive = automatically takes smaller steps during close encounters (far more "
           "accurate there, same overall CPU); dt becomes the largest step. "
           "fixed = every step is exactly dt (original behaviour)."),
    _param("adaptive_accuracy", 0.02, "pos_float", "Physics",
           "Only used when timestep = adaptive. Step = this x the time-scale of the closest pair. "
           "Smaller = more accurate but slower."),

    _param("min_steps", 50000, "pos_int", "Search rules",
           "Steps a case must survive to count as High interest."),
    _param("long_test_steps", 500000, "pos_int", "Search rules",
           "Maximum steps per case. Surviving all of them = Very High interest."),
    _param("collision_dist", 0.05, "nonneg_float", "Search rules",
           "Two bodies closer than this = Collision."),
    _param("ejection_dist", 10.0, "pos_float", "Search rules",
           "A body further than this from the centre of mass may be an Ejection."),
    _param("ejection_test", "energy", ("energy", "distance"), "Search rules",
           "energy = far away AND moving fast enough to escape (physically correct). "
           "distance = far away is enough (original behaviour)."),
    _param("check_every", 50, "pos_int", "Search rules",
           "Run the ejection + energy checks every this many steps. "
           "(Collisions are checked every step - it costs nothing.)"),
    _param("skip_unbound", True, "bool", "Search rules",
           "Skip systems with total energy >= 0. Physics guarantees they fly apart, "
           "so simulating them is wasted CPU."),
    _param("energy_error_limit", 0.0, "nonneg_float", "Search rules",
           "Stop a case early if its energy error grows above this (e.g. 0.001). "
           "0 = never stop, only record the error."),

    _param("position_range", [-2.0, 2.0], "range", "Random starts",
           "Random starting x, y, z are picked between these two numbers."),
    _param("velocity_range", [-0.5, 0.5], "range", "Random starts",
           "Random starting vx, vy, vz are picked between these two numbers."),
    _param("start_case", 0, "nonneg_int", "Random starts",
           "0 = carry on where the last run stopped. Any other number = start from that case."),
    _param("max_cases", 0, "nonneg_int", "Random starts",
           "Stop after this many cases. 0 = run forever (until Ctrl+C)."),

    _param("workers", 1, "workers", "Speed & CPU",
           "CPU cores to use. 1 = gentle background use. \"auto\" = all cores except one."),
    _param("throttle_delay", 0.1, "nonneg_float", "Speed & CPU",
           "Pause in seconds after every case."),
    _param("cpu_limit_percent", 100, "percent", "Speed & CPU",
           "Target CPU use per worker (1-100). 50 = work half the time, rest half. 100 = no limit."),
    _param("low_priority", True, "bool", "Speed & CPU",
           "Run at below-normal priority so other programs always get the CPU first."),
    _param("print_each_case", True, "bool", "Speed & CPU",
           "Print one line per case. false = only high-interest finds + a status line every 100 cases."),

    _param("all_cases_file", "3body_all_cases.csv", "filename", "Files",
           "CSV file that gets every random case."),
    _param("high_priority_file", "3body_high_interest.csv", "filename", "Files",
           "CSV file that gets only High / Very High cases."),
    _param("custom_cases_file", "3body_custom_cases.csv", "filename", "Files",
           "CSV file that gets custom-mode runs."),
    _param("progress_file", "3body_progress.json", "filename", "Files",
           "Small file that remembers the last finished case."),

    _param("custom_positions", FIGURE8_POSITIONS, "matrix", "Custom case",
           "Starting [x, y, z] of body 1, 2 and 3 (used when mode = custom)."),
    _param("custom_velocities", FIGURE8_VELOCITIES, "matrix", "Custom case",
           "Starting [vx, vy, vz] of body 1, 2 and 3 (used when mode = custom)."),

    _param("startup_prompt_seconds", 10, "nonneg_int", "Start-up",
           "Seconds to wait for you to press C (customise) before starting. 0 = start at once."),
]

SPEC = {p["name"]: p for p in PARAMETERS}
DEFAULTS = {p["name"]: p["default"] for p in PARAMETERS}

PRESETS = {
    # Famous stable periodic orbit (Chenciner & Montgomery, 2000): three equal
    # masses chasing each other around a figure "8".
    "figure8": {"mode": "custom", "masses": [1.0, 1.0, 1.0],
                "custom_positions": FIGURE8_POSITIONS,
                "custom_velocities": FIGURE8_VELOCITIES},
    # Burrau's "Pythagorean" problem (1913): masses 3, 4, 5 start at rest on the
    # corners of a 3-4-5 triangle. Chaotic dance with extremely close passes,
    # ending around t = 60 with the lightest body thrown out. The close passes
    # need a very fine adaptive step and no collision cut-off.
    "pythagorean": {"mode": "custom", "masses": [3.0, 4.0, 5.0],
                    "custom_positions": [[1.0, 3.0, 0.0], [-2.0, -1.0, 0.0], [1.0, -1.0, 0.0]],
                    "custom_velocities": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "timestep": "adaptive", "adaptive_accuracy": 0.005,
                    "softening": 0.0, "collision_dist": 0.0, "long_test_steps": 100000},
}

CSV_HEADERS = (
    ["case_id", "m1", "m2", "m3"]
    + [f"p{b}_{c}" for b in (1, 2, 3) for c in "xyz"]
    + [f"v{b}_{c}" for b in (1, 2, 3) for c in "xyz"]
    + ["survival_steps", "end_reason", "interest_level",
       # new columns (added after the original ones)
       "survival_time", "event_detail", "min_distance", "energy_error",
       "initial_energy", "G", "dt", "softening", "integrator", "timestep"]
)

HIGH_LEVELS = ("High", "Very High")


# ---------------------------------------------------------------------------
# PHYSICS ENGINE  (compiled to machine code by Numba)
# ---------------------------------------------------------------------------

SURVIVED, COLLISION, EJECTION, NUMERICAL = 0, 1, 2, 3
END_NAMES = {SURVIVED: "Survived", COLLISION: "Collision",
             EJECTION: "Ejection", NUMERICAL: "Numerical error"}


@njit(cache=True, error_model="numpy")
def _accelerations(pos, m, G, eps2, acc):
    """Fill acc with the gravitational acceleration of each body.
    Returns the smallest squared distance between any two bodies."""
    for i in range(3):
        for d in range(3):
            acc[i, d] = 0.0
    min_r2 = 1.0e300
    for i in range(2):
        for j in range(i + 1, 3):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < min_r2:
                min_r2 = r2
            s2 = r2 + eps2
            g_over_r3 = G / (s2 * math.sqrt(s2))
            # Newton's 3rd law: one calculation gives the pull on BOTH bodies
            ai = m[j] * g_over_r3
            aj = m[i] * g_over_r3
            acc[i, 0] += ai * dx
            acc[i, 1] += ai * dy
            acc[i, 2] += ai * dz
            acc[j, 0] -= aj * dx
            acc[j, 1] -= aj * dy
            acc[j, 2] -= aj * dz
    return min_r2


@njit(cache=True, error_model="numpy")
def _total_energy(pos, vel, m, G, eps2):
    """Kinetic energy + (softened) gravitational potential energy."""
    kinetic = 0.0
    for i in range(3):
        kinetic += 0.5 * m[i] * (vel[i, 0] ** 2 + vel[i, 1] ** 2 + vel[i, 2] ** 2)
    potential = 0.0
    for i in range(2):
        for j in range(i + 1, 3):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            potential -= G * m[i] * m[j] / math.sqrt(dx * dx + dy * dy + dz * dz + eps2)
    return kinetic + potential


@njit(cache=True, error_model="numpy")
def _escaping_body(pos, vel, m, G, ejection_dist, energy_test):
    """Index of a body that has been ejected, or -1 if none."""
    total = m[0] + m[1] + m[2]
    cx = (m[0] * pos[0, 0] + m[1] * pos[1, 0] + m[2] * pos[2, 0]) / total
    cy = (m[0] * pos[0, 1] + m[1] * pos[1, 1] + m[2] * pos[2, 1]) / total
    cz = (m[0] * pos[0, 2] + m[1] * pos[1, 2] + m[2] * pos[2, 2]) / total
    for k in range(3):
        dx = pos[k, 0] - cx
        dy = pos[k, 1] - cy
        dz = pos[k, 2] - cz
        if dx * dx + dy * dy + dz * dz <= ejection_dist * ejection_dist:
            continue
        if not energy_test:
            return k
        # Treat the other two bodies as one object at their centre of mass and
        # ask: does body k have enough energy to escape it, and is it moving away?
        i = (k + 1) % 3
        j = (k + 2) % 3
        mp = m[i] + m[j]
        rx = pos[k, 0] - (m[i] * pos[i, 0] + m[j] * pos[j, 0]) / mp
        ry = pos[k, 1] - (m[i] * pos[i, 1] + m[j] * pos[j, 1]) / mp
        rz = pos[k, 2] - (m[i] * pos[i, 2] + m[j] * pos[j, 2]) / mp
        vx = vel[k, 0] - (m[i] * vel[i, 0] + m[j] * vel[j, 0]) / mp
        vy = vel[k, 1] - (m[i] * vel[i, 1] + m[j] * vel[j, 1]) / mp
        vz = vel[k, 2] - (m[i] * vel[i, 2] + m[j] * vel[j, 2]) / mp
        r = math.sqrt(rx * rx + ry * ry + rz * rz)
        mu = m[k] * mp / (m[k] + mp)
        energy = 0.5 * mu * (vx * vx + vy * vy + vz * vz) - G * m[k] * mp / r
        if energy > 0.0 and rx * vx + ry * vy + rz * vz > 0.0:
            return k
    return -1


@njit(cache=True, error_model="numpy")
def _step_size_limit(pos, vel, m, G, eps2):
    """Shortest 'time-scale' of any pair: how long until the pair's
    distance could change a lot (free-fall time, or distance / speed)."""
    shortest = 1.0e300
    for i in range(2):
        for j in range(i + 1, 3):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dvx = vel[j, 0] - vel[i, 0]
            dvy = vel[j, 1] - vel[i, 1]
            dvz = vel[j, 2] - vel[i, 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            r = math.sqrt(r2)
            free_fall = math.sqrt(r2 * r / (G * (m[i] + m[j])))
            if free_fall < shortest:
                shortest = free_fall
            v = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
            if v > 0.0 and r / v < shortest:
                shortest = r / v
    return shortest


@njit(cache=True, error_model="numpy")
def _simulate(pos, vel, m, G, dt, eps, max_steps, check_every, collision_dist,
              ejection_dist, energy_test, use_leapfrog, adaptive, accuracy,
              energy_limit, record_every, record_start, frames):
    """Move the system forward step by step until something happens.
    pos and vel are updated in place. "Steps" always means steps of size dt;
    with adaptive time-steps one dt may be split into several smaller moves.
    Returns (steps, time, end_code, ejected_body, min_distance,
             max_energy_error, frames_recorded, moves_made)."""
    eps2 = eps * eps
    coll2 = collision_dist * collision_dist
    t_end = max_steps * dt

    # work arrays are made once per case, never inside the loop
    acc = np.empty((3, 3))
    k2 = np.empty((3, 3))
    k3 = np.empty((3, 3))
    k4 = np.empty((3, 3))
    v2 = np.empty((3, 3))
    v3 = np.empty((3, 3))
    p_tmp = np.empty((3, 3))

    e0 = _total_energy(pos, vel, m, G, eps2)
    e_scale = abs(e0) if abs(e0) > 1e-12 else 1.0
    max_err = 0.0

    r2 = _accelerations(pos, m, G, eps2, acc)
    min_r2 = r2
    n_frames = 0
    next_record = record_start
    if record_every > 0 and record_start == 0:
        frames[0, :, :] = pos
        n_frames = 1
        next_record = record_every
    next_check = check_every

    end_code = SURVIVED
    body = -1
    step = 0
    t = 0.0
    moves = 0
    if r2 < coll2:
        end_code = COLLISION

    while end_code == SURVIVED and step < max_steps:
        h = dt
        if adaptive:
            h = accuracy * _step_size_limit(pos, vel, m, G, eps2)
            if h > dt:
                h = dt
            if h < dt * 1e-7:   # safety floor so the program can never get stuck
                h = dt * 1e-7
            if t + h > t_end:
                h = t_end - t
        half = 0.5 * h
        sixth = h / 6.0

        if use_leapfrog:
            # kick - drift - kick
            for i in range(3):
                for d in range(3):
                    vel[i, d] += half * acc[i, d]
                    pos[i, d] += h * vel[i, d]
            r2 = _accelerations(pos, m, G, eps2, acc)
            for i in range(3):
                for d in range(3):
                    vel[i, d] += half * acc[i, d]
        else:
            # RK4. acc already holds k1 (it was worked out at the end of the last step)
            for i in range(3):
                for d in range(3):
                    p_tmp[i, d] = pos[i, d] + half * vel[i, d]
            _accelerations(p_tmp, m, G, eps2, k2)
            for i in range(3):
                for d in range(3):
                    v2[i, d] = vel[i, d] + half * acc[i, d]
                    p_tmp[i, d] = pos[i, d] + half * v2[i, d]
            _accelerations(p_tmp, m, G, eps2, k3)
            for i in range(3):
                for d in range(3):
                    v3[i, d] = vel[i, d] + half * k2[i, d]
                    p_tmp[i, d] = pos[i, d] + h * v3[i, d]
            _accelerations(p_tmp, m, G, eps2, k4)
            for i in range(3):
                for d in range(3):
                    v4 = vel[i, d] + h * k3[i, d]
                    pos[i, d] += sixth * (vel[i, d] + 2.0 * v2[i, d] + 2.0 * v3[i, d] + v4)
                    vel[i, d] += sixth * (acc[i, d] + 2.0 * k2[i, d] + 2.0 * k3[i, d] + k4[i, d])
            # acceleration at the new position = k1 of the next step, and it
            # also gives us the pair distances for the collision check for free
            r2 = _accelerations(pos, m, G, eps2, acc)

        moves += 1
        if adaptive:
            t += h
            step = max_steps if t >= t_end else int(t / dt + 1e-9)
        else:
            step += 1
            t = step * dt

        if r2 < min_r2:
            min_r2 = r2

        if record_every > 0 and step >= next_record and n_frames < frames.shape[0]:
            frames[n_frames, :, :] = pos
            n_frames += 1
            while next_record <= step:
                next_record += record_every

        if r2 < coll2:
            end_code = COLLISION
        elif step >= next_check:
            while next_check <= step:
                next_check += check_every
            err = abs(_total_energy(pos, vel, m, G, eps2) - e0) / e_scale
            if err > max_err:
                max_err = err
            if energy_limit > 0.0 and err > energy_limit:
                end_code = NUMERICAL
            else:
                body = _escaping_body(pos, vel, m, G, ejection_dist, energy_test)
                if body >= 0:
                    end_code = EJECTION

    err = abs(_total_energy(pos, vel, m, G, eps2) - e0) / e_scale
    if err > max_err:
        max_err = err
    # make sure the very last position (e.g. the collision) is in the recording
    if record_every > 0 and step >= record_start and n_frames < frames.shape[0]:
        last_recorded = next_record - record_every
        if step != last_recorded or n_frames == 0:
            frames[n_frames, :, :] = pos
            n_frames += 1

    return step, t, end_code, body, math.sqrt(min_r2), max_err, n_frames, moves


# ---------------------------------------------------------------------------
# RUNNING ONE CASE
# ---------------------------------------------------------------------------

def random_initial_conditions(case_id, s):
    """Same random numbers as the original script (np.random.seed(case_id)),
    so case N here starts exactly like case N in the original."""
    rng = np.random.RandomState(case_id)
    positions = rng.uniform(s["position_range"][0], s["position_range"][1], (3, 3))
    velocities = rng.uniform(s["velocity_range"][0], s["velocity_range"][1], (3, 3))
    return positions, velocities


def energies(positions, velocities, s):
    """(total energy, internal energy). Internal = total minus the energy of the
    whole system drifting through space, which can never be used to break it apart."""
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    m = np.asarray(s["masses"], dtype=np.float64)
    total = _total_energy(pos, vel, m, float(s["G"]), float(s["softening"]) ** 2)
    v_com = (m[:, None] * vel).sum(axis=0) / m.sum()
    return total, total - 0.5 * m.sum() * float(v_com @ v_com)


def run_simulation(positions, velocities, s, max_steps=None,
                   record_every=0, record_start=0, max_frames=0):
    """Run the physics engine. Returns (result dict, recorded frames, final pos, final vel)."""
    pos = np.array(positions, dtype=np.float64).reshape(3, 3)
    vel = np.array(velocities, dtype=np.float64).reshape(3, 3)
    m = np.array(s["masses"], dtype=np.float64)
    frames = np.zeros((max(max_frames, 1), 3, 3))
    if max_steps is None:
        max_steps = s["long_test_steps"]
    steps, t, code, body, min_d, err, n, moves = _simulate(
        pos, vel, m, float(s["G"]), float(s["dt"]), float(s["softening"]),
        int(max_steps), int(s["check_every"]), float(s["collision_dist"]),
        float(s["ejection_dist"]), s["ejection_test"] == "energy",
        s["integrator"] == "leapfrog", s["timestep"] == "adaptive",
        float(s["adaptive_accuracy"]), float(s["energy_error_limit"]),
        int(record_every), int(record_start), frames)
    result = {"steps": int(steps), "time": float(t), "end_reason": END_NAMES[int(code)],
              "min_distance": float(min_d), "energy_error": float(err), "moves": int(moves)}
    if code == COLLISION:
        pairs = [(0, 1), (0, 2), (1, 2)]
        i, j = min(pairs, key=lambda p: np.linalg.norm(pos[p[0]] - pos[p[1]]))
        result["detail"] = f"bodies {i + 1} & {j + 1}"
    elif code == EJECTION:
        result["detail"] = f"body {body + 1} escaped"
    elif code == NUMERICAL:
        result["detail"] = f"energy error {err:.2e} > limit"
    else:
        result["detail"] = ""
    return result, frames[:n], pos, vel


def interest_level(steps, end_reason, s):
    if steps < s["min_steps"]:
        return "Low"
    if end_reason == "Survived" and steps >= s["long_test_steps"]:
        return "Very High"
    return "High"


def evaluate_case(case_id, positions, velocities, s):
    """Simulate one system and return its CSV row."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    e_total, e_internal = energies(positions, velocities, s)

    if s["skip_unbound"] and e_internal >= 0.0:
        gaps = [np.linalg.norm(positions[a] - positions[b]) for a, b in ((0, 1), (0, 2), (1, 2))]
        result = {"steps": 0, "time": 0.0, "end_reason": "Unbound (skipped)",
                  "detail": f"internal energy {e_internal:+.4g} >= 0",
                  "min_distance": float(min(gaps)), "energy_error": 0.0}
    else:
        result = run_simulation(positions, velocities, s)[0]

    row = {"case_id": case_id,
           "m1": s["masses"][0], "m2": s["masses"][1], "m3": s["masses"][2]}
    for b in range(3):
        for c, axis in enumerate("xyz"):
            # full precision: a chaotic system needs the exact numbers to be replayed
            row[f"p{b + 1}_{axis}"] = float(positions[b, c])
            row[f"v{b + 1}_{axis}"] = float(velocities[b, c])
    row.update({
        "survival_steps": result["steps"],
        "end_reason": result["end_reason"],
        "interest_level": interest_level(result["steps"], result["end_reason"], s),
        "survival_time": round(result["time"], 10),
        "event_detail": result["detail"],
        "min_distance": float(f"{result['min_distance']:.6g}"),
        "energy_error": float(f"{result['energy_error']:.3e}"),
        "initial_energy": float(e_total),
        "G": s["G"], "dt": s["dt"], "softening": s["softening"],
        "integrator": s["integrator"],
        "timestep": "fixed" if s["timestep"] == "fixed" else f"adaptive {s['adaptive_accuracy']}",
    })
    return row


def run_random_case(case_id, s):
    positions, velocities = random_initial_conditions(case_id, s)
    return evaluate_case(case_id, positions, velocities, s)


def rest(compute_seconds, s):
    """Let the CPU rest: fixed pause + (optional) pause that keeps usage near cpu_limit_percent."""
    pause = s["throttle_delay"]
    if s["cpu_limit_percent"] < 100:
        pause += compute_seconds * (100.0 / s["cpu_limit_percent"] - 1.0)
    if pause > 0:
        time.sleep(pause)


def lower_priority():
    """Ask the operating system to give other programs the CPU first."""
    try:
        if os.name == "nt":
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(10)
    except Exception:
        pass


def warm_up_engine():
    """Compile the physics engine (first run only - after that it loads from cache)."""
    if not HAVE_NUMBA:
        print("NOTE: numba is not installed, so the physics runs in slow pure-Python mode.")
        print("      Install it for a big speed-up:  pip install numba\n")
        return
    print("Loading physics engine...", end=" ", flush=True)
    t0 = time.perf_counter()
    s = dict(DEFAULTS, long_test_steps=10)
    run_simulation(np.eye(3), np.zeros((3, 3)), s)
    run_simulation(np.eye(3), np.zeros((3, 3)), s, record_every=1, max_frames=2)
    print(f"ready ({time.perf_counter() - t0:.1f} s)")


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

class SettingsError(Exception):
    pass


def _number(value):
    if isinstance(value, bool):
        raise ValueError("must be a number")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("must be a number")
    if not math.isfinite(v):
        raise ValueError("must be a normal (finite) number")
    return v


def _integer(value):
    v = _number(value)
    if v != int(v):
        raise ValueError("must be a whole number")
    return int(v)


def _number_list(value, length, what):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"must be a list of {length} {what}, like {json.dumps(DEFAULTS_EXAMPLES[what])}")
    return [_number(v) for v in value]


DEFAULTS_EXAMPLES = {"numbers": [1.0, 2.0, 3.0], "numbers (low, high)": [-2.0, 2.0],
                     "rows": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}


def coerce(name, value):
    """Check one setting and convert it to the right type. Raises ValueError."""
    kind = SPEC[name]["kind"]
    if isinstance(kind, tuple):
        v = str(value).strip().lower()
        if v not in kind:
            raise ValueError("must be one of: " + ", ".join(kind))
        return v
    if kind == "bool":
        if isinstance(value, bool):
            return value
        v = str(value).strip().lower()
        if v in ("true", "yes", "y", "1", "on"):
            return True
        if v in ("false", "no", "n", "0", "off"):
            return False
        raise ValueError("must be true or false")
    if kind in ("pos_float", "nonneg_float"):
        v = _number(value)
        if kind == "pos_float" and v <= 0:
            raise ValueError("must be greater than 0")
        if v < 0:
            raise ValueError("must be 0 or more")
        return v
    if kind in ("pos_int", "nonneg_int", "percent"):
        v = _integer(value)
        if kind == "pos_int" and v < 1:
            raise ValueError("must be 1 or more")
        if kind == "nonneg_int" and v < 0:
            raise ValueError("must be 0 or more")
        if kind == "percent" and not 1 <= v <= 100:
            raise ValueError("must be between 1 and 100")
        return v
    if kind == "workers":
        if str(value).strip().lower() == "auto":
            return "auto"
        v = _integer(value)
        if v < 1:
            raise ValueError('must be 1 or more, or "auto"')
        return v
    if kind == "masses":
        v = _number_list(value, 3, "numbers")
        if min(v) <= 0:
            raise ValueError("every mass must be greater than 0")
        return v
    if kind == "range":
        v = _number_list(value, 2, "numbers (low, high)")
        if v[0] >= v[1]:
            raise ValueError("the first number (low) must be smaller than the second (high)")
        return v
    if kind == "matrix":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("must be 3 rows (one per body) of 3 numbers, like "
                             + json.dumps(DEFAULTS_EXAMPLES["rows"]))
        return [_number_list(row, 3, "numbers") for row in value]
    if kind == "filename":
        v = str(value).strip()
        if not v:
            raise ValueError("must not be empty")
        return v
    raise ValueError(f"unknown setting type {kind}")


def check_together(s):
    """Checks that involve more than one setting."""
    if s["min_steps"] > s["long_test_steps"]:
        raise SettingsError("min_steps must not be bigger than long_test_steps")
    files = [s["all_cases_file"], s["high_priority_file"], s["custom_cases_file"], s["progress_file"]]
    if len({in_base(f).lower() for f in files}) != len(files):
        raise SettingsError("the four file names must all be different")


def parse_text(text):
    """Turn typed text into a value: JSON first (numbers, lists, true/false),
    then '1 2 3' / '1, 2, 3' as a list of numbers, otherwise plain text."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    parts = text.replace(",", " ").split()
    if len(parts) > 1:
        try:
            return [float(p) for p in parts]
        except ValueError:
            pass
    return text


def load_settings(path):
    s = dict(DEFAULTS)
    if not os.path.exists(path):
        save_settings(path, s)
        print(f"Created {os.path.basename(path)} with the default settings.")
        return s
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise SettingsError(f"{os.path.basename(path)} has a typing mistake on line {e.lineno}: {e.msg}.\n"
                            f"Fix it, or run with --reset-settings to start again from the defaults.")
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if key not in SPEC:
            print(f"WARNING: unknown setting '{key}' in {os.path.basename(path)} was ignored.")
            continue
        try:
            s[key] = coerce(key, value)
        except ValueError as e:
            raise SettingsError(f"Setting '{key}' in {os.path.basename(path)} {e}.")
    return s


def save_settings(path, s):
    lines = ['  "_help": "Edit the values below. See PARAMETER_GUIDE.txt for what each one does."']
    lines += [f'  "{p["name"]}": {json.dumps(s[p["name"]])}' for p in PARAMETERS]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")
    os.replace(tmp, path)


def show(value):
    return json.dumps(value) if isinstance(value, (list, bool)) else str(value)


def print_settings(s):
    group = None
    for i, p in enumerate(PARAMETERS, 1):
        if p["group"] != group:
            group = p["group"]
            print(f"\n  -- {group} --")
        mark = "" if s[p["name"]] == p["default"] else "   (changed)"
        print(f"  {i:2d}. {p['name']:<24} {show(s[p['name']])}{mark}")


def settings_menu(s, path):
    """Simple text menu to change settings before the run starts."""
    s = dict(s)
    while True:
        print("\n" + "=" * 60 + "\nSETTINGS")
        print_settings(s)
        print("\n  Type a number to change that setting")
        print("  P = load a preset (" + ", ".join(PRESETS) + ")   D = reset all to defaults")
        print("  S = save to settings.json and start   ENTER = start without saving")
        choice = input("> ").strip().lower()
        if choice == "":
            break
        if choice == "s":
            save_settings(path, s)
            print(f"Saved to {os.path.basename(path)}.")
            break
        if choice == "d":
            s = dict(DEFAULTS)
            continue
        if choice == "p":
            name = input("Preset name: ").strip().lower()
            if name in PRESETS:
                s.update(PRESETS[name])
            else:
                print("No preset with that name.")
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(PARAMETERS):
            print("Please type one of the numbers in the list.")
            continue
        p = PARAMETERS[int(choice) - 1]
        print(f"\n{p['name']}: {p['help']}\nCurrent value: {show(s[p['name']])}")
        if p["kind"] == "matrix":
            rows = []
            for b in range(3):
                while True:
                    text = input(f"  body {b + 1} - three numbers x y z (ENTER = keep {s[p['name']][b]}): ")
                    if not text.strip():
                        rows.append(s[p["name"]][b])
                        break
                    try:
                        rows.append(_number_list(parse_text(text), 3, "numbers"))
                        break
                    except ValueError as e:
                        print(f"  That value {e}.")
            s[p["name"]] = rows
            continue
        while True:
            text = input("New value (ENTER = keep): ")
            if not text.strip():
                break
            try:
                s[p["name"]] = coerce(p["name"], parse_text(text))
                break
            except ValueError as e:
                print(f"  That value {e}.")
        try:
            check_together(s)
        except SettingsError as e:
            print(f"  WARNING: {e}")
    check_together(s)
    return s


def wait_for_key(seconds):
    """Count down; return the key pressed (or None if time ran out)."""
    end = time.time() + seconds
    if os.name == "nt":
        import msvcrt
        while time.time() < end:
            print(f"\r  Starting in {int(end - time.time()) + 1:2d} s ", end="", flush=True)
            if msvcrt.kbhit():
                print()
                return msvcrt.getwch()
            time.sleep(0.05)
    else:
        import select
        while time.time() < end:
            print(f"\r  Starting in {int(end - time.time()) + 1:2d} s ", end="", flush=True)
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                return sys.stdin.readline().strip()[:1] or "\n"
    print()
    return None


def startup_prompt(s, path):
    seconds = s["startup_prompt_seconds"]
    if seconds <= 0 or not sys.stdin.isatty():
        return s
    print("\nCurrent settings (from settings.json):")
    keys = ["mode", "masses", "dt", "integrator", "long_test_steps", "workers",
            "throttle_delay", "cpu_limit_percent"]
    for k in keys:
        print(f"  {k:<18} {show(s[k])}")
    print("\nPress C to customise, or ENTER to start now.")
    key = wait_for_key(seconds)
    if key and key.lower() == "c":
        return settings_menu(s, path)
    return s


# ---------------------------------------------------------------------------
# FILES
# ---------------------------------------------------------------------------

def prepare_csv(path):
    """Create the CSV with its header, or check an existing file is compatible."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)
        return
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header != CSV_HEADERS:
        raise SettingsError(
            f"{os.path.basename(path)} has different columns (probably made by the old script).\n"
            f"Move or rename it, or pick another file name in settings.json, then run again.")
    # If the program was killed while writing, the last line may be half-written: remove it.
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 65536))
        tail = f.read()
        if not tail.endswith(b"\n"):
            cut = tail.rfind(b"\n")
            f.truncate(size - len(tail) + cut + 1)
            print(f"Removed a half-written last line from {os.path.basename(path)}.")


def read_last_row(path):
    """Last data row of a CSV as a dict (reads only the end of the file, so it's fast)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 65536))
        tail = f.read().decode("utf-8", errors="replace")
    lines = [line for line in tail.splitlines() if line.strip()]
    if not lines:
        return None
    row = next(csv.reader([lines[-1]]))
    if row == CSV_HEADERS or len(row) != len(CSV_HEADERS):
        return None
    return dict(zip(CSV_HEADERS, row))


def save_progress(path, case_id):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"last_id": case_id}, f)
    os.replace(tmp, path)  # atomic: the file is never left half-written


def find_resume_point(s):
    """The CSV is the source of truth for what has been done."""
    all_path = in_base(s["all_cases_file"])
    high_path = in_base(s["high_priority_file"])
    last = read_last_row(all_path)
    last_id = int(last["case_id"]) if last else 0

    progress_path = in_base(s["progress_file"])
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                saved = int(json.load(f).get("last_id", 0))
            if saved != last_id:
                print(f"Note: progress file said case {saved}, but the CSV ends at case {last_id}. "
                      f"Using the CSV.")
        except (ValueError, OSError):
            print("Note: progress file was unreadable - using the CSV instead.")

    # if the program stopped between the two CSV writes, add the missing high-interest row
    if last and last["interest_level"] in HIGH_LEVELS:
        high_last = read_last_row(high_path)
        if high_last is None or int(high_last["case_id"]) < last_id:
            with open(high_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(last)
    return last_id


# ---------------------------------------------------------------------------
# THE SEARCH
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self.start = time.time()
        self.count = 0
        self.reasons = {}
        self.levels = {}

    def add(self, row):
        self.count += 1
        self.reasons[row["end_reason"]] = self.reasons.get(row["end_reason"], 0) + 1
        self.levels[row["interest_level"]] = self.levels.get(row["interest_level"], 0) + 1

    def line(self):
        elapsed = max(time.time() - self.start, 1e-9)
        parts = [f"{self.count:,} cases in {elapsed / 60:.1f} min ({self.count / elapsed:.2f}/s)"]
        parts += [f"{k}: {v:,}" for k, v in sorted(self.reasons.items())]
        parts += [f"{k}: {self.levels.get(k, 0):,}" for k in HIGH_LEVELS]
        return "[Status] " + " | ".join(parts)


def announce(row, s):
    if row["interest_level"] in HIGH_LEVELS:
        print()
        print("*" * 40)
        print("*** HIGH PRIORITY CASE FOUND! ***")
        print(f"Case: {row['case_id']}")
        print(f"Survived: {row['survival_steps']:,} steps (t = {row['survival_time']:g})")
        print(f"End reason: {row['end_reason']} {row['event_detail']}".rstrip())
        print(f"Classification: {row['interest_level']}")
        print(f"Energy error: {row['energy_error']:.1e}"
              + ("   (!) high - may be a numerical artefact" if row["energy_error"] > 1e-4 else ""))
        print("*" * 40)
    elif s["print_each_case"]:
        print(f"Case {row['case_id']} finished | Result: {row['end_reason']} | "
              f"Steps: {row['survival_steps']:,} | Rank: {row['interest_level']}")


_WORKER_SETTINGS = None


def _worker_init(s):
    global _WORKER_SETTINGS
    _WORKER_SETTINGS = s
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # only the main process handles Ctrl+C
    if s["low_priority"]:
        lower_priority()


def _worker_batch(first_id, count):
    """Run a small batch of cases in a worker process. Sending cases in batches
    instead of one by one cuts the cost of talking between processes."""
    rows = []
    for case_id in range(first_id, first_id + count):
        t0 = time.perf_counter()
        rows.append(run_random_case(case_id, _WORKER_SETTINGS))
        rest(time.perf_counter() - t0, _WORKER_SETTINGS)
    return rows


def _run_serial(first, s, record):
    case_id = first
    while s["max_cases"] == 0 or case_id < first + s["max_cases"]:
        t0 = time.perf_counter()
        row = run_random_case(case_id, s)
        record(row)
        rest(time.perf_counter() - t0, s)
        case_id += 1


def _run_parallel(first, s, record, workers):
    """Several processes work on different cases at once. Results are written
    to the CSV strictly in case order, so resuming always works."""
    results = queue.Queue()
    limit = first + s["max_cases"] if s["max_cases"] else None
    batch = 20
    window = workers * 8           # batches allowed to run ahead of the oldest unfinished one
    next_submit = next_write = first
    in_flight = 0
    finished = {}
    pool = mp.Pool(workers, initializer=_worker_init, initargs=(s,))
    try:
        while True:
            while in_flight + len(finished) < window and (limit is None or next_submit < limit):
                count = batch if limit is None else min(batch, limit - next_submit)
                pool.apply_async(_worker_batch, (next_submit, count),
                                 callback=results.put, error_callback=results.put)
                next_submit += count
                in_flight += 1
            if in_flight == 0:
                break
            try:
                item = results.get(timeout=0.5)  # timeout keeps Ctrl+C working on Windows
            except queue.Empty:
                continue
            in_flight -= 1
            if isinstance(item, BaseException):
                raise item
            finished[item[0]["case_id"]] = item
            while next_write in finished:
                rows = finished.pop(next_write)
                for row in rows:
                    record(row)
                next_write += len(rows)
    finally:
        pool.terminate()
        pool.join()


def run_search(s):
    all_path = in_base(s["all_cases_file"])
    high_path = in_base(s["high_priority_file"])
    progress_path = in_base(s["progress_file"])
    prepare_csv(all_path)
    prepare_csv(high_path)
    last_done = find_resume_point(s)
    first = s["start_case"] if s["start_case"] > 0 else last_done + 1
    workers = max(1, (os.cpu_count() or 2) - 1) if s["workers"] == "auto" else s["workers"]

    print()
    print("=" * 60)
    print("3-BODY ORBIT SEARCH")
    print("=" * 60)
    print(f"Starting from Case {first}")
    print(f"Minimum qualification: {s['min_steps']:,} steps")
    print(f"Long test: {s['long_test_steps']:,} steps")
    print(f"Collision distance: {s['collision_dist']}")
    print(f"Ejection distance: {s['ejection_dist']} (test: {s['ejection_test']})")
    print(f"Integrator: {s['integrator']}   dt: {s['dt']}   Workers: {workers}")
    print(f"Stops after: {s['max_cases']:,} cases" if s["max_cases"] else "Runs until you press Ctrl+C")
    print("=" * 60)
    print()

    stats = Stats()
    state = {"last": last_done, "saved_at": time.time()}

    with open(all_path, "a", newline="", encoding="utf-8") as all_file, \
            open(high_path, "a", newline="", encoding="utf-8") as high_file:
        all_writer = csv.DictWriter(all_file, fieldnames=CSV_HEADERS)
        high_writer = csv.DictWriter(high_file, fieldnames=CSV_HEADERS)

        def record(row):
            all_writer.writerow(row)
            all_file.flush()
            if row["interest_level"] in HIGH_LEVELS:
                high_writer.writerow(row)
                high_file.flush()
            announce(row, s)
            stats.add(row)
            state["last"] = row["case_id"]
            if stats.count % 100 == 0:
                print(stats.line())
            # the CSV already records progress, so this small file only needs
            # refreshing every few seconds instead of after every case
            if time.time() - state["saved_at"] > 5:
                save_progress(progress_path, state["last"])
                state["saved_at"] = time.time()

        try:
            if workers == 1:
                _run_serial(first, s, record)
            else:
                _run_parallel(first, s, record, workers)
            print("\nFinished the requested number of cases.")
        except KeyboardInterrupt:
            print("\nSearch safely paused.")
        finally:
            save_progress(progress_path, state["last"])
            print("Progress has been saved.")
            print(stats.line())


def run_custom(s):
    path = in_base(s["custom_cases_file"])
    prepare_csv(path)
    last = read_last_row(path)
    number = int(str(last["case_id"]).split("_")[-1]) + 1 if last else 1
    case_id = f"custom_{number}"

    print(f"\nRunning custom case {case_id} for up to {s['long_test_steps']:,} steps...")
    t0 = time.perf_counter()
    row = evaluate_case(case_id, s["custom_positions"], s["custom_velocities"], s)
    seconds = time.perf_counter() - t0
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)

    print(f"\nResult:          {row['end_reason']} {row['event_detail']}".rstrip())
    print(f"Survived:        {row['survival_steps']:,} steps (t = {row['survival_time']:g})")
    print(f"Interest level:  {row['interest_level']}")
    print(f"Closest approach:{row['min_distance']:>10.4g}")
    print(f"Energy error:    {row['energy_error']:.2e}")
    print(f"Took {seconds:.2f} s. Saved to {os.path.basename(path)}.")
    print(f"\nWatch it:  python visualize_case.py {case_id}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search for long-lived 3-body systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python three_body_search.py\n"
               "  python three_body_search.py --setup\n"
               "  python three_body_search.py --set workers=4 --set throttle_delay=0\n"
               "  python three_body_search.py --preset figure8\n"
               "  python three_body_search.py --set max_cases=100 --no-prompt\n"
               "See PARAMETER_GUIDE.txt for every setting.")
    parser.add_argument("--settings", default="settings.json",
                        help="settings file to use (default: settings.json)")
    parser.add_argument("--setup", action="store_true",
                        help="open the settings menu before starting")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="change one setting for this run only (can be repeated)")
    parser.add_argument("--preset", choices=sorted(PRESETS),
                        help="run a famous example system (custom mode)")
    parser.add_argument("--defaults", action="store_true",
                        help="ignore settings.json and use the built-in defaults")
    parser.add_argument("--reset-settings", action="store_true",
                        help="rewrite settings.json with the defaults and exit")
    parser.add_argument("--show-settings", action="store_true",
                        help="print the settings that would be used and exit")
    parser.add_argument("--no-prompt", action="store_true",
                        help="start immediately (no 'press C to customise' countdown)")
    args = parser.parse_args()

    path = in_base(args.settings)
    try:
        if args.reset_settings:
            save_settings(path, DEFAULTS)
            print(f"{os.path.basename(path)} reset to the default settings.")
            return
        s = dict(DEFAULTS) if args.defaults else load_settings(path)
        if args.preset:
            s.update(PRESETS[args.preset])
        for item in args.set:
            if "=" not in item:
                raise SettingsError(f"--set needs NAME=VALUE, got '{item}'")
            key, text = item.split("=", 1)
            key = key.strip()
            if key not in SPEC:
                raise SettingsError(f"unknown setting '{key}'. See PARAMETER_GUIDE.txt for the list.")
            try:
                s[key] = coerce(key, parse_text(text))
            except ValueError as e:
                raise SettingsError(f"--set {key}: value {e}.")
        check_together(s)

        if args.show_settings:
            print_settings(s)
            return
        if args.setup:
            s = settings_menu(s, path)
        elif not args.no_prompt:
            s = startup_prompt(s, path)

        if s["low_priority"]:
            lower_priority()
        warm_up_engine()
        if s["mode"] == "custom":
            run_custom(s)
        else:
            run_search(s)
    except SettingsError as e:
        print(f"\nSETTINGS PROBLEM: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
