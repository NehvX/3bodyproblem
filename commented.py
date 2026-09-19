import csv
import os
import json
import time
import numpy as np


# ==========================================
#   1. SEARCH PARAMETERS
# ==========================================

# File Tracking
ALL_CASES_FILE = '3body_all_cases.csv'
HIGH_PRIORITY_FILE = '3body_high_interest.csv'
PROGRESS_FILE = '3body_progress.json'


# ==========================================
#   2. PHYSICS PARAMETERS
# ==========================================

G = 1.0

# Equal masses
#1 mass unit = mass of earth
MASSES = np.array([1.007825, 4.002603, 15.994915])
#mass of top 3 elements found in universe (currently)


# ------------------------------------------
# Simulation length
# ------------------------------------------

# A system MUST survive this many steps
# to become a high-priority candidate.
MIN_STEPS = 50000

# High-priority candidates continue running
# to this many steps.
LONG_TEST_STEPS = 500000


# ------------------------------------------
# Numerical settings
# ------------------------------------------

DT = 0.001

# Prevents numerical problems when bodies
# get extremely close.
SOFTENING = 0.01


# ==========================================
#   3. COLLISION / EJECTION RULES
# ==========================================

# If any two bodies get closer than this:
# -> Collision
COLLISION_DIST = 0.05


# If any body gets farther than this from
# the center of mass:
# -> Ejection
EJECTION_DIST = 10.0


# How often we check collision/ejection
CHECK_EVERY = 50


# Pause between cases to reduce CPU usage
THROTTLE_DELAY = 0.1


# ==========================================
#   4. CSV FORMAT
# ==========================================

CSV_HEADERS = [

    # Case information
    'case_id',

    # Masses
    'm1',
    'm2',
    'm3',

    # Body 1 initial position
    'p1_x',
    'p1_y',
    'p1_z',

    # Body 2 initial position
    'p2_x',
    'p2_y',
    'p2_z',

    # Body 3 initial position
    'p3_x',
    'p3_y',
    'p3_z',

    # Body 1 initial velocity
    'v1_x',
    'v1_y',
    'v1_z',

    # Body 2 initial velocity
    'v2_x',
    'v2_y',
    'v2_z',

    # Body 3 initial velocity
    'v3_x',
    'v3_y',
    'v3_z',

    # Results
    'survival_steps',
    'end_reason',
    'interest_level'
]


# ==========================================
#   5. PHYSICS ENGINE
# ==========================================

def get_accelerations(positions):
    """
    Calculate gravitational acceleration
    on each of the three bodies.
    """

    accel = np.zeros_like(positions)

    for i in range(3):

        for j in range(3):

            if i != j:

                # Vector from body i -> body j
                r_vec = positions[j] - positions[i]

                # Distance with softening
                dist = np.sqrt(
                    np.sum(r_vec ** 2) +
                    SOFTENING ** 2
                )

                # Newtonian gravitational acceleration
                accel[i] += (
                    G *
                    MASSES[j] *
                    r_vec /
                    (dist ** 3)
                )

    return accel


def rk4_step(positions, velocities):
    """
    Advance the system by one timestep
    using fourth-order Runge-Kutta.
    """

    # k1
    k1_v = get_accelerations(positions)
    k1_p = velocities

    # k2
    k2_v = get_accelerations(
        positions + 0.5 * DT * k1_p
    )

    k2_p = (
        velocities +
        0.5 * DT * k1_v
    )

    # k3
    k3_v = get_accelerations(
        positions + 0.5 * DT * k2_p
    )

    k3_p = (
        velocities +
        0.5 * DT * k2_v
    )

    # k4
    k4_v = get_accelerations(
        positions + DT * k3_p
    )

    k4_p = (
        velocities +
        DT * k3_v
    )

    # Combine
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


# ==========================================
#   6. INITIAL CONDITIONS
# ==========================================

def generate_initial_conditions(case_id):
    """
    Generate a unique reproducible starting
    configuration for each case.

    The case ID is used as the random seed.
    """

    np.random.seed(case_id)

    # Random positions
    positions = np.random.uniform(
        -2.0,
        2.0,
        (3, 3)
    )

    # Random velocities
    velocities = np.random.uniform(
        -0.5,
        0.5,
        (3, 3)
    )

    return positions, velocities


# ==========================================
#   7. CHECK COLLISION / EJECTION
# ==========================================

def check_system(positions):
    """
    Check whether the system has collided
    or whether a body has been ejected.

    Returns:
        None
        "Collision"
        "Ejection"
    """

    # --------------------------------------
    # Pairwise distances
    # --------------------------------------

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

    # Collision
    if minimum_distance < COLLISION_DIST:
        return "Collision"


    # --------------------------------------
    # Center of mass
    # --------------------------------------

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

    # Ejection
    if max(distances_from_center) > EJECTION_DIST:
        return "Ejection"


    # Everything okay
    return None


# ==========================================
#   8. RUN ONE ORBIT
# ==========================================

def evaluate_orbit(case_id):
    """
    Run one 3-body configuration.

    Important:
    - A case must reach 50,000 steps to qualify.
    - If it qualifies, it continues to 500,000.
    """

    # Generate starting configuration
    positions, velocities = (
        generate_initial_conditions(case_id)
    )

    # Save exact starting configuration
    start_positions = np.copy(positions)
    start_velocities = np.copy(velocities)

    end_reason = "Survived"
    survival_steps = 0

    qualified = False


    # ======================================
    # MAIN SIMULATION
    # ======================================

    for step in range(1, LONG_TEST_STEPS + 1):

        # Advance simulation
        positions, velocities = rk4_step(
            positions,
            velocities
        )

        survival_steps = step


        # ----------------------------------
        # Check system
        # ----------------------------------

        if step % CHECK_EVERY == 0:

            result = check_system(
                positions
            )

            if result is not None:

                end_reason = result

                break


        # ----------------------------------
        # Qualification checkpoint
        # ----------------------------------

        if (
            step == MIN_STEPS
            and end_reason == "Survived"
        ):

            qualified = True

            print(
                f"    >>> CASE {case_id} "
                f"QUALIFIED at {MIN_STEPS:,} steps"
            )


    # ======================================
    # CLASSIFICATION
    # ======================================

    if not qualified:

        # Didn't reach 50k
        interest_level = "Low"

    elif survival_steps >= LONG_TEST_STEPS:

        # Survived the entire long test
        interest_level = "Very High"

    else:

        # Survived 50k but eventually failed
        interest_level = "High"


    # ======================================
    # PACKAGE RESULT
    # ======================================

    return {

        'case_id': case_id,

        'm1': MASSES[0],
        'm2': MASSES[1],
        'm3': MASSES[2],


        # ------------------------------
        # Initial positions
        # ------------------------------

        'p1_x': round(start_positions[0][0], 8),
        'p1_y': round(start_positions[0][1], 8),
        'p1_z': round(start_positions[0][2], 8),

        'p2_x': round(start_positions[1][0], 8),
        'p2_y': round(start_positions[1][1], 8),
        'p2_z': round(start_positions[1][2], 8),

        'p3_x': round(start_positions[2][0], 8),
        'p3_y': round(start_positions[2][1], 8),
        'p3_z': round(start_positions[2][2], 8),


        # ------------------------------
        # Initial velocities
        # ------------------------------

        'v1_x': round(start_velocities[0][0], 8),
        'v1_y': round(start_velocities[0][1], 8),
        'v1_z': round(start_velocities[0][2], 8),

        'v2_x': round(start_velocities[1][0], 8),
        'v2_y': round(start_velocities[1][1], 8),
        'v2_z': round(start_velocities[1][2], 8),

        'v3_x': round(start_velocities[2][0], 8),
        'v3_y': round(start_velocities[2][1], 8),
        'v3_z': round(start_velocities[2][2], 8),


        # ------------------------------
        # Results
        # ------------------------------

        'survival_steps': survival_steps,

        'end_reason': end_reason,

        'interest_level': interest_level
    }


# ==========================================
#   9. PROGRESS TRACKING
# ==========================================

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


# ==========================================
#   10. INITIALIZE CSV FILES
# ==========================================

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


# ==========================================
#   11. MAIN SEARCH
# ==========================================

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


    # Open files
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


        # ==================================
        # Infinite search
        # ==================================

        while True:

            current_case_id += 1

            print(
                f"\nCase {current_case_id} starting..."
            )


            # ----------------------------------
            # Run physics
            # ----------------------------------

            result = evaluate_orbit(
                current_case_id
            )


            # ----------------------------------
            # Save to ALL cases
            # ----------------------------------

            all_writer.writerow(result)

            all_file.flush()


            # ----------------------------------
            # High priority
            # ----------------------------------

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


            # ----------------------------------
            # Save progress
            # ----------------------------------

            save_progress(
                current_case_id
            )


            # ----------------------------------
            # CPU throttle
            # ----------------------------------

            time.sleep(
                THROTTLE_DELAY
            )


# ==========================================
#   12. START PROGRAM
# ==========================================

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
