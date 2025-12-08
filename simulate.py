# fixed_simulation.py
import time
import json
import copy
import random
import math
import os
import csv
from collections import defaultdict

# ============================
#  Build spatial index
# ============================
def build_spatial_index(level_json):
    arrows = level_json["arrows"]
    coord_map = defaultdict(set)

    for i, arrow in enumerate(arrows):
        for n in arrow["nodes"]:
            coord_map[(n["x"], n["y"])].add(i)

    return coord_map


# ============================
#  NEW: Direction logic (head1 -> head0)
# ============================
def get_direction_vec(nodes):
    """
    nodes is tail → ... → head
    head = last element
    before_head = second last
    direction = head - before_head
    """
    if len(nodes) < 2:
        return (0, 0)

    head = nodes[-1]
    prev = nodes[-2]

    dx = head["x"] - prev["x"]
    dy = head["y"] - prev["y"]
    return dx, dy

def can_move_arrow(arrow_idx, state, coord_map):
    nodes = state["arrows"][arrow_idx]["nodes"]
    dx, dy = get_direction_vec(nodes)
    head = nodes[-1]

    nx = head["x"] + dx
    ny = head["y"] + dy

    W, H = state["size"]["x"], state["size"]["y"]
    if not (0 <= nx < W and 0 <= ny < H):
        return False

    occ = coord_map.get((nx, ny), set())

    # nếu ô trống → OK
    if len(occ) == 0:
        return True

    # Nếu chỉ có chính arrow đó trong ô → chỉ hợp lệ nếu đó là tail
    if occ == {arrow_idx}:
        tail = nodes[0]
        if tail["x"] == nx and tail["y"] == ny:
            return True

    # bị chặn
    return False

def move_arrow_one_step(arrow_idx, state, coord_map):
    nodes = state["arrows"][arrow_idx]["nodes"]
    dx, dy = get_direction_vec(nodes)

    # xoá toàn bộ vị trí cũ
    for n in nodes:
        coord_map[(n["x"], n["y"])].discard(arrow_idx)

    # di chuyển tất cả nodes theo (dx,dy)
    for n in nodes:
        n["x"] += dx
        n["y"] += dy

    # cập nhật vị trí mới
    for n in nodes:
        coord_map[(n["x"], n["y"])].add(arrow_idx)

# ============================
#  Single simulation
# ============================
def simulate_one_run(level_json, record_steps=False):
    state = copy.deepcopy(level_json)
    coord_map = build_spatial_index(state)

    moves = 0
    branching_factors = []
    deadlocks = 0
    step_log = []

    start_time = time.time()

    while True:
        movable = []

        for i in range(len(state["arrows"])):
            if can_move_arrow(i, state, coord_map):
                movable.append(i)

        if not movable:
            deadlocks += 1
            break

        branching_factors.append(len(movable))

        chosen = random.choice(movable)

        if record_steps:
            step_log.append({
                "move_index": moves,
                "arrow_moved": chosen,
                "before_nodes": copy.deepcopy(state["arrows"][chosen]["nodes"])
            })

        move_arrow_one_step(chosen, state, coord_map)

        if record_steps:
            step_log[-1]["after_nodes"] = copy.deepcopy(state["arrows"][chosen]["nodes"])

        moves += 1

        if moves > 5000:  # safety
            deadlocks += 1
            break

    alpha_time = time.time() - start_time
    avg_branching = sum(branching_factors) / max(1, len(branching_factors))

    difficulty_score = min(
        10,
        (
            0.5 * math.log(1 + moves)
            + 0.7 * avg_branching
            + 0.4 * deadlocks
        )
    )

    result = {
        "alpha_time": alpha_time,
        "moves": moves,
        "deadlocks": deadlocks,
        "branching": avg_branching,
        "difficulty": difficulty_score
    }

    if record_steps:
        result["steps"] = step_log

    return result


# ============================
#  Multi-run aggregator
# ============================
def simulate_level(level_json, level_name, runs=30, beta=10000, record_steps=False):

    results = [simulate_one_run(level_json, record_steps=False) for _ in range(runs)]

    one_run_log = simulate_one_run(level_json, record_steps=True) if record_steps else None

    avg_alpha = sum(r["alpha_time"] for r in results) / runs
    avg_moves = sum(r["moves"] for r in results) / runs

    summary = {
        "Level_Name": level_name,
        "Estimated_Player_Time": beta * avg_alpha,
        "Avg_Moves": avg_moves,
    }

    return summary, one_run_log


# ============================
#  Run single file
# ============================
def run_single_file(path, runs=50, record_steps=True):
    with open(path, "r") as f:
        level = json.load(f)

    summary, log = simulate_level(level, os.path.basename(path), runs=runs, record_steps=record_steps)

    print("\n=== SUMMARY FOR", path, "===")
    print(json.dumps(summary, indent=4))

    # Save steps
    if record_steps:
        out_path = path.replace(".json", "_steps.json")
        with open(out_path, "w") as of:
            json.dump(log, of, indent=2)
        print("Saved steps:", out_path)


# ============================
#  Run folder and save CSV
# ============================
def run_folder(folder, runs=30, output_csv="results.csv"):
    rows = []

    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            path = os.path.join(folder, fname)
            with open(path, "r") as f:
                level = json.load(f)

            summary, _ = simulate_level(level, fname, runs=runs)

            rows.append(summary)

    # write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Level_Name", "Estimated_Player_Time", "Avg_Moves"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("CSV saved:", output_csv)

# ---------------------------
# If executed as script - examples
# ---------------------------
if __name__ == "__main__":
    # example: single file with debug on
    run_single_file("/Users/hoangnguyen/Documents/py/ArrowPuzzle/100lv/lv8.json", runs=2000, record_steps=False)

    # example: run whole folder
    # run_folder("100lv", runs=500, output_csv="levels_result.csv")
