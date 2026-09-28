"""
SPEED + CPU BENCHMARK
=====================
Measures the original code against the optimised engine on this computer.

    python benchmark.py

Takes about a minute. Nothing is written to your result files.
"""

import time

import numpy as np

import three_body_search as tb

# ---------------------------------------------------------------------------
# The ORIGINAL physics code, copied unchanged (only shortened) for comparison
# ---------------------------------------------------------------------------
G = 1.0
MASSES = np.array([1.007825, 4.002603, 15.994915])
DT = 0.001
SOFTENING = 0.01


def original_accelerations(positions):
    accel = np.zeros_like(positions)
    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = positions[j] - positions[i]
                dist = np.sqrt(np.sum(r_vec ** 2) + SOFTENING ** 2)
                accel[i] += G * MASSES[j] * r_vec / (dist ** 3)
    return accel


def original_rk4_step(positions, velocities):
    k1_v = original_accelerations(positions)
    k1_p = velocities
    k2_v = original_accelerations(positions + 0.5 * DT * k1_p)
    k2_p = velocities + 0.5 * DT * k1_v
    k3_v = original_accelerations(positions + 0.5 * DT * k2_p)
    k3_p = velocities + 0.5 * DT * k2_v
    k4_v = original_accelerations(positions + DT * k3_p)
    k4_p = velocities + DT * k3_v
    new_positions = positions + (DT / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
    new_velocities = velocities + (DT / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return new_positions, new_velocities


def time_per_step(function, steps):
    t0 = time.perf_counter()
    function(steps)
    return (time.perf_counter() - t0) / steps


def main():
    s = dict(tb.DEFAULTS)
    no_events = dict(s, collision_dist=0.0, ejection_dist=1e9, timestep="fixed")
    pos, vel = tb.random_initial_conditions(54, s)   # a case that survives a long time

    print("\n1) Same answer?  (original vs new engine, 2,000 fixed RK4 steps)")
    p, v = pos.copy(), vel.copy()
    for _ in range(2000):
        p, v = original_rk4_step(p, v)
    _, _, p2, _ = tb.run_simulation(pos, vel, no_events, max_steps=2000)
    print(f"   largest difference in any position: {np.abs(p - p2).max():.1e}  (0 = identical)")

    print("\n2) Time per step")
    tb.run_simulation(pos, vel, no_events, max_steps=10)          # compile / load first

    def run_original(n):
        p, v = pos.copy(), vel.copy()
        for _ in range(n):
            p, v = original_rk4_step(p, v)

    def run_pure_python(n):
        frames = np.zeros((1, 3, 3))
        tb._simulate.py_func(pos.copy(), vel.copy(), np.array(s["masses"]), 1.0, DT, SOFTENING,
                             n, 50, 0.0, 1e9, False, False, False, 0.02, 0.0, 0, 0, frames)

    def engine(settings):
        return lambda n: tb.run_simulation(pos, vel, settings, max_steps=n)

    rows = [("original code", time_per_step(run_original, 5000))]
    if tb.HAVE_NUMBA:
        rows.append(("new engine without numba", time_per_step(run_pure_python, 5000)))
    rows.append(("new engine, RK4", time_per_step(engine(no_events), 500_000)))
    rows.append(("new engine, leapfrog", time_per_step(engine(dict(no_events, integrator="leapfrog")), 500_000)))
    base = rows[0][1]
    for name, sec in rows:
        print(f"   {name:<26} {sec * 1e6:10.3f} microseconds/step   "
              f"{base / sec:8.0f}x   (500,000 steps = {sec * 500_000:8.2f} s)")

    print("\n3) Whole search, first 300 cases (default settings, no pause)")
    fixed_steps = 0
    for label, settings in (("fixed dt (original style)", dict(s, timestep="fixed")),
                            ("adaptive dt (new default)", s)):
        t0 = time.perf_counter()
        errors, unreliable, high = [], 0, 0
        for case_id in range(1, 301):
            p0, v0 = tb.random_initial_conditions(case_id, settings)
            result = tb.run_simulation(p0, v0, settings)[0]
            if settings["timestep"] == "fixed":
                fixed_steps += result["moves"]
            errors.append(result["energy_error"])
            if tb.interest_level(result["steps"], result["end_reason"], settings) != "Low":
                high += 1
                unreliable += result["energy_error"] > 1e-4
        took = time.perf_counter() - t0
        print(f"   {label:<27} {took:6.2f} s   median energy error {np.median(errors):.1e}   "
              f"high-interest: {high} ({unreliable} unreliable)")
    print(f"   original code: at least {base * fixed_steps / 60:.0f} minutes for the same 300 cases")
    print(f"   ({fixed_steps:,} RK4 steps x its time per step - really longer, because it misses")
    print("    most collisions and keeps going)")

    print("\n4) CPU use with the default 0.1 s pause, 100 cases")
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for case_id in range(1, 101):
        t0 = time.perf_counter()
        tb.run_random_case(case_id, s)
        tb.rest(time.perf_counter() - t0, s)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    print(f"   busy {cpu:.2f} s out of {wall:.1f} s  ->  about {100 * cpu / wall:.1f}% of one CPU core")
    print("   (the original kept one core at ~100% the whole time)\n")


if __name__ == "__main__":
    main()
