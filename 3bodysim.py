import csv
import os
import json
import time
import numpy as np


ALL_CASES_FILE = '3body_all_cases.csv'
HIGH_PRIORITY_FILE = '3body_high_interest.csv'
PROGRESS_FILE = '3body_progress.json'


G = 1.0

MASSES = np.array([1.007825, 4.002603, 15.994915])

MIN_STEPS = 50000

LONG_TEST_STEPS = 500000

DT = 0.001

SOFTENING = 0.01

COLLISION_DIST = 0.05

EJECTION_DIST = 10.0

CHECK_EVERY = 50

THROTTLE_DELAY = 0.1

CSV_HEADERS = [
    'case_id',
    'm1',
    'm2',
    'm3',
    'p1_x',
    'p1_y',
    'p1_z',
    'p2_x',
    'p2_y',
    'p2_z',
    'p3_x',
    'p3_y',
    'p3_z',
    'v1_x',
    'v1_y',
    'v1_z',
    'v2_x',
    'v2_y',
    'v2_z',
    'v3_x',
    'v3_y',
    'v3_z',
    'survival_steps',
    'end_reason',
    'interest_level'
]


def get_accelerations(positions):
    accel = np.zeros_like(positions)

    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = positions[j] - positions[i]

                dist = np.sqrt(
                    np.sum(r_vec ** 2) +
                    SOFTENING ** 2
                )

                accel[i] += (
                    G *
                    MASSES[j] *
                    r_vec /
                    (dist ** 3)
                )

    return accel


def rk4_step(positions, velocities):
    k1_v = get_accelerations(positions)
    k1_p = velocities

    k2_v = get_accelerations(
        positions + 0.5 * DT * k1_p
    )

    k2_p = (
        velocities +
        0.5 * DT * k1_v
    )

    k3_v = get_accelerations(
        positions + 0.5 * DT * k2_p
    )

    k3_p = (
        velocities +
        0.5 * DT * k2_v
    )

    k4_v = get_accelerations(
        positions + DT * k3_p
    )

    k4_p = (
        velocities +
        DT * k3_v
    )

    new_positions = (
        positions +
        (DT / 6.0) *
        (
            k1_p +
            2 * k2_p +
            2 * k3_p +
            k4_p
        )
    )

    new_velocities = (
        velocities +
        (DT / 6.0) *
        (
            k1_v +
            2 * k2_v +
            2 * k3_v +
            k4_v
        )
    )

    return new_positions, new_velocities


def generate_initial_conditions(case_id):
    np.random.seed(case_id)

    positions = np.random.uniform(
        -2.0,
        2.0,
        (3, 3)
    )

    velocities = np.random.uniform(
        -0.5,
        0.5,
        (3, 3)
    )

    return positions, velocities


def check_system(positions):
    d12 = np.linalg.norm(
        positions[0] - positions[1]
    )

    d13 = np.linalg.norm(
        positions[0] - positions[2]
    )

    d23 = np.linalg.norm(
        positions[1] - positions[2]
    )

    minimum_distance = min(
        d12,
        d13,
        d23
    )

    if minimum_distance < COLLISION_DIST:
        return "Collision"

    center_of_mass = np.average(
        positions,
        axis=0,
        weights=MASSES
    )

    distances_from_center = [
        np.linalg.norm(
            position - center_of_mass
        )
        for position in positions
    ]

    if max(distances_from_center) > EJECTION_DIST:
        return "Ejection"

    return None


def evaluate_orbit(case_id):
    positions, velocities = (
        generate_initial_conditions(case_id)
    )

    start_positions = np.copy(positions)
    start_velocities = np.copy(velocities)

    end_reason = "Survived"
    survival_steps = 0

    qualified = False

    for step in range(1, LONG_TEST_STEPS + 1):
        positions, velocities = rk4_step(
            positions,
            velocities
        )

        survival_steps = step

        if step % CHECK_EVERY == 0:
            result = check_system(
                positions
            )

            if result is not None:
                end_reason = result

                break

        if (
            step == MIN_STEPS
            and end_reason == "Survived"
        ):
            qualified = True

            print(
                f"    >>> CASE {case_id} "
                f"QUALIFIED at {MIN_STEPS:,} steps"
            )

    if not qualified:
        interest_level = "Low"

    elif survival_steps >= LONG_TEST_STEPS:
        interest_level = "Very High"

    else:
        interest_level = "High"

    return {
        'case_id': case_id,

        'm1': MASSES[0],
        'm2': MASSES[1],
        'm3': MASSES[2],

        'p1_x': round(start_positions[0][0], 8),
        'p1_y': round(start_positions[0][1], 8),
        'p1_z': round(start_positions[0][2], 8),

        'p2_x': round(start_positions[1][0], 8),
        'p2_y': round(start_positions[1][1], 8),
        'p2_z': round(start_positions[1][2], 8),

        'p3_x': round(start_positions[2][0], 8),
        'p3_y': round(start_positions[2][1], 8),
        'p3_z': round(start_positions[2][2], 8),

        'v1_x': round(start_velocities[0][0], 8),
        'v1_y': round(start_velocities[0][1], 8),
        'v1_z': round(start_velocities[0][2], 8),

        'v2_x': round(start_velocities[1][0], 8),
        'v2_y': round(start_velocities[1][1], 8),
        'v2_z': round(start_velocities[1][2], 8),

        'v3_x': round(start_velocities[2][0], 8),
        'v3_y': round(start_velocities[2][1], 8),
        'v3_z': round(start_velocities[2][2], 8),

        'survival_steps': survival_steps,

        'end_reason': end_reason,

        'interest_level': interest_level
    }


def get_last_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(
                PROGRESS_FILE,
                'r'
            ) as f:
                data = json.load(f)

                return data.get(
                    'last_id',
                    0
                )

        except Exception:
            return 0

    return 0


def save_progress(case_id):
    with open(
        PROGRESS_FILE,
        'w'
    ) as f:
        json.dump(
            {
                'last_id': case_id
            },
            f
        )


def initialize_csvs():
    for filename in [
        ALL_CASES_FILE,
        HIGH_PRIORITY_FILE
    ]:
        if not os.path.exists(filename):
            with open(
                filename,
                'w',
                newline='',
                encoding='utf-8'
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=CSV_HEADERS
                )

                writer.writeheader()


def run_background_search():
    initialize_csvs()

    current_case_id = (
        get_last_progress()
    )

    print()
    print("=" * 60)
    print("3-BODY ORBIT SEARCH")
    print("=" * 60)

    print(
        f"Starting from Case {current_case_id + 1}"
    )

    print(
        f"Minimum qualification: "
        f"{MIN_STEPS:,} steps"
    )

    print(
        f"Long test: "
        f"{LONG_TEST_STEPS:,} steps"
    )

    print(
        f"Collision distance: "
        f"{COLLISION_DIST}"
    )

    print(
        f"Ejection distance: "
        f"{EJECTION_DIST}"
    )

    print("=" * 60)
    print()

    with open(
        ALL_CASES_FILE,
        'a',
        newline='',
        encoding='utf-8'
    ) as all_file, \
    open(
        HIGH_PRIORITY_FILE,
        'a',
        newline='',
        encoding='utf-8'
    ) as high_file:

        all_writer = csv.DictWriter(
            all_file,
            fieldnames=CSV_HEADERS
        )

        high_writer = csv.DictWriter(
            high_file,
            fieldnames=CSV_HEADERS
        )

        while True:
            current_case_id += 1

            print(
                f"\nCase {current_case_id} starting..."
            )

            result = evaluate_orbit(
                current_case_id
            )

            all_writer.writerow(result)

            all_file.flush()

            if result['interest_level'] in [
                "High",
                "Very High"
            ]:
                high_writer.writerow(result)

                high_file.flush()

                print()
                print(
                    "****************************************"
                )

                print(
                    f"*** HIGH PRIORITY CASE FOUND! ***"
                )

                print(
                    f"Case: "
                    f"{current_case_id}"
                )

                print(
                    f"Survived: "
                    f"{result['survival_steps']:,} steps"
                )

                print(
                    f"End reason: "
                    f"{result['end_reason']}"
                )

                print(
                    f"Classification: "
                    f"{result['interest_level']}"
                )

                print(
                    "****************************************"
                )

            else:
                print(
                    f"Case {current_case_id} finished | "
                    f"Result: {result['end_reason']} | "
                    f"Steps: {result['survival_steps']:,} | "
                    f"Rank: {result['interest_level']}"
                )

            save_progress(
                current_case_id
            )

            time.sleep(
                THROTTLE_DELAY
            )


if __name__ == "__main__":
    try:
        run_background_search()

    except KeyboardInterrupt:
        print()
        print(
            "Search safely paused."
        )

        print(
            "Progress has been saved."
        )
