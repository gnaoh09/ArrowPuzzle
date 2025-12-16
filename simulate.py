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

def build_mechanics_index(state):
    """
    Build spatial index for wayBlockers and blackHoles.
    Returns: (wayBlockers_map, blackHoles_set)
    - wayBlockers_map: {(x,y): index in wayBlockers list}
    - blackHoles_set: set of (x,y) positions
    """
    wayBlockers_map = {}
    for i, wb in enumerate(state.get("wayBlockers", [])):
        pos = wb.get("position", {})
        x, y = pos.get("x"), pos.get("y")
        if x is not None and y is not None:
            wayBlockers_map[(x, y)] = i
    
    blackHoles_set = set()
    for bh in state.get("blackHoles", []):
        pos = bh.get("position", {})
        x, y = pos.get("x"), pos.get("y")
        if x is not None and y is not None:
            blackHoles_set.add((x, y))
    
    return wayBlockers_map, blackHoles_set


# ============================
#  Direction logic (movement direction)
# ============================
def get_direction_vec(nodes):
    """
    nodes is tail → ... → head
    head = last element (node[-1])
    before_head = second last (node[-2])
    direction = head - before_head
    
    This gives us the direction the arrow is moving.
    """
    if len(nodes) < 2:
        return (0, 0)

    head = nodes[-1]
    prev = nodes[-2]

    dx = head["x"] - prev["x"]
    dy = head["y"] - prev["y"]
    return dx, dy

def can_move_arrow(arrow_idx, state, coord_map, wayBlockers_map):
    """
    Check if arrow can move (escape).
    An arrow can move if the ENTIRE path from head to board edge is clear.
    No other arrow or wayBlocker can be blocking any cell along the path.
    """
    nodes = state["arrows"][arrow_idx]["nodes"]
    head = nodes[-1]
    dx, dy = get_direction_vec(nodes)
    
    W, H = state["size"]["x"], state["size"]["y"]
    
    # Start from head position and check all cells along the direction until out of bounds
    check_x = head["x"]
    check_y = head["y"]
    
    while True:
        # Move one step in the direction
        check_x += dx
        check_y += dy
        
        # If we've reached outside the board, path is clear!
        if not (0 <= check_x < W and 0 <= check_y < H):
            break
        
        # Check if wayBlocker is blocking this cell
        if (check_x, check_y) in wayBlockers_map:
            return False
        
        # Check if any OTHER arrow occupies this cell
        occ = coord_map.get((check_x, check_y), set())
        
        # If another arrow (not this one) is blocking, can't move
        if occ and occ != {arrow_idx}:
            return False
        
        # If only this arrow occupies it, check if it's part of our body
        if arrow_idx in occ:
            # This cell is occupied by our own body
            # This is OK only if it will be vacated when we move
            # But since we're removing the entire arrow, this shouldn't happen
            # unless the arrow loops back on itself
            pass
    
    return True

def check_path_through_blackhole(arrow_idx, state, coord_map, wayBlockers_map, blackHoles_set):
    """
    Check if arrow can validly move into a black hole.
    Returns True if path can reach a black hole before being blocked by other arrows or wayBlockers.
    Arrow can only fall into blackhole if:
    1. Blackhole is immediately in front of arrow head, OR
    2. Arrow can move to blackhole without being blocked by other arrows or wayBlockers
    """
    nodes = state["arrows"][arrow_idx]["nodes"]
    head = nodes[-1]
    dx, dy = get_direction_vec(nodes)
    
    W, H = state["size"]["x"], state["size"]["y"]
    
    check_x = head["x"]
    check_y = head["y"]
    
    while True:
        check_x += dx
        check_y += dy
        
        # Out of bounds - no blackhole reached
        if not (0 <= check_x < W and 0 <= check_y < H):
            break
        
        # Check if blackhole is at this position - if yes, can fall into it!
        if (check_x, check_y) in blackHoles_set:
            return True
        
        # Check if wayBlocker is blocking - if yes, can't reach blackhole beyond this
        if (check_x, check_y) in wayBlockers_map:
            return False
        
        # Check if another arrow is blocking - if yes, can't reach blackhole beyond this
        occ = coord_map.get((check_x, check_y), set())
        if occ and occ != {arrow_idx}:
            return False
    
    return False

def remove_arrow(arrow_idx, state, coord_map, is_successful_escape=True):
    """
    Remove the entire arrow from the board.
    If is_successful_escape=True, decrease lockTime for all wayBlockers.
    """
    nodes = state["arrows"][arrow_idx]["nodes"]
    
    # Remove all nodes from spatial index
    for n in nodes:
        coord_map[(n["x"], n["y"])].discard(arrow_idx)
    
    # Remove arrow from state
    state["arrows"].pop(arrow_idx)
    
    # If this is a successful escape, decrease lockTime for wayBlockers
    if is_successful_escape:
        wayBlockers = state.get("wayBlockers", [])
        for wb in wayBlockers:
            if "lockTime" in wb:
                wb["lockTime"] = max(0, wb["lockTime"] - 1)
    
    # Update spatial index - decrement all indices greater than arrow_idx
    # Because we removed an arrow, all subsequent indices shift down by 1
    new_coord_map = defaultdict(set)
    for pos, arrow_set in coord_map.items():
        new_set = set()
        for idx in arrow_set:
            if idx > arrow_idx:
                new_set.add(idx - 1)
            elif idx < arrow_idx:
                new_set.add(idx)
            # idx == arrow_idx is already removed above
        if new_set:
            new_coord_map[pos] = new_set
    
    return new_coord_map

def remove_unlocked_wayBlockers(state, wayBlockers_map):
    """
    Remove wayBlockers with lockTime <= 0.
    Returns updated wayBlockers_map.
    """
    wayBlockers = state.get("wayBlockers", [])
    
    # Remove wayBlockers with lockTime <= 0
    state["wayBlockers"] = [wb for wb in wayBlockers if wb.get("lockTime", 0) > 0]
    
    # Rebuild wayBlockers map
    new_wayBlockers_map = {}
    for i, wb in enumerate(state["wayBlockers"]):
        pos = wb.get("position", {})
        x, y = pos.get("x"), pos.get("y")
        if x is not None and y is not None:
            new_wayBlockers_map[(x, y)] = i
    
    return new_wayBlockers_map


# ============================
#  Single simulation
# ============================
def simulate_one_run(level_json, record_steps=False, debug=False):
    state = copy.deepcopy(level_json)
    coord_map = build_spatial_index(state)
    wayBlockers_map, blackHoles_set = build_mechanics_index(state)

    moves = 0
    branching_factors = []
    step_log = []
    blackhole_deaths = 0

    start_time = time.time()

    while True:
        # Check win condition
        if len(state["arrows"]) == 0:
            if debug:
                print(f"\n🎉 ALL ARROWS ESCAPED! Total moves: {moves}")
            break
            
        movable = []
        blackhole_victims = []

        for i in range(len(state["arrows"])):
            # First check if this arrow would fall into black hole
            if check_path_through_blackhole(i, state, coord_map, wayBlockers_map, blackHoles_set):
                blackhole_victims.append(i)
            elif can_move_arrow(i, state, coord_map, wayBlockers_map):
                movable.append(i)

        if debug:
            print(f"\n--- Move {moves} ---")
            print(f"Arrows remaining: {len(state['arrows'])}")
            print(f"WayBlockers: {[(wb.get('position'), wb.get('lockTime')) for wb in state.get('wayBlockers', [])]}")
            print(f"BlackHole victims: {blackhole_victims}")
            print(f"Movable arrows: {movable}")
            if blackhole_victims:
                for idx in blackhole_victims:
                    nodes = state["arrows"][idx]["nodes"]
                    head = nodes[-1]
                    dx, dy = get_direction_vec(nodes)
                    print(f"  Arrow {idx}: head at ({head['x']},{head['y']}), direction ({dx},{dy}) - WILL FALL INTO BLACKHOLE 💀")
            if movable:
                for idx in movable:
                    nodes = state["arrows"][idx]["nodes"]
                    dx, dy = get_direction_vec(nodes)
                    head = nodes[-1]
                    
                    # Show the entire escape path
                    path = []
                    check_x, check_y = head["x"], head["y"]
                    W, H = state["size"]["x"], state["size"]["y"]
                    
                    while True:
                        check_x += dx
                        check_y += dy
                        if not (0 <= check_x < W and 0 <= check_y < H):
                            break
                        path.append((check_x, check_y))
                    
                    print(f"  Arrow {idx}: head at ({head['x']},{head['y']}), direction ({dx},{dy})")
                    print(f"    Escape path: {path} - ALL CLEAR ✓")

        # Handle blackhole victims first (if any can move into blackhole)
        if blackhole_victims:
            # Prioritize blackhole deaths in random selection
            all_choices = blackhole_victims + movable
        else:
            all_choices = movable
        
        if not all_choices:
            if debug:
                print("\n❌ DEADLOCK: No movable arrows!")
                for i, arrow in enumerate(state["arrows"]):
                    nodes = arrow["nodes"]
                    head = nodes[-1]
                    dx, dy = get_direction_vec(nodes)
                    
                    W, H = state["size"]["x"], state["size"]["y"]
                    
                    # Check escape path to find where it's blocked
                    check_x, check_y = head["x"], head["y"]
                    blocked_at = None
                    blocked_by = None
                    blocked_by_wb = False
                    
                    while True:
                        check_x += dx
                        check_y += dy
                        if not (0 <= check_x < W and 0 <= check_y < H):
                            break
                        
                        if (check_x, check_y) in wayBlockers_map:
                            blocked_at = (check_x, check_y)
                            blocked_by_wb = True
                            wb_idx = wayBlockers_map[(check_x, check_y)]
                            blocked_by = f"wayBlocker (lockTime={state['wayBlockers'][wb_idx].get('lockTime')})"
                            break
                        
                        occ = coord_map.get((check_x, check_y), set())
                        if occ and occ != {i}:
                            blocked_at = (check_x, check_y)
                            blocked_by = f"arrows {occ}"
                            break
                    
                    if blocked_at:
                        print(f"  Arrow {i}: head at ({head['x']},{head['y']}), blocked at {blocked_at} by {blocked_by}")
                    else:
                        print(f"  Arrow {i}: head at ({head['x']},{head['y']}), path is clear but shouldn't be here!")
            break

        branching_factors.append(len(all_choices))

        chosen = random.choice(all_choices)
        is_blackhole_death = chosen in blackhole_victims
        
        if debug:
            if is_blackhole_death:
                print(f"💀 Chosen arrow: {chosen} - FELL INTO BLACKHOLE (doesn't count as escape)")
            else:
                print(f"✓ Chosen arrow: {chosen} 🚀 ESCAPED SUCCESSFULLY")

        if record_steps:
            step_log.append({
                "move_index": moves,
                "arrow_moved": chosen,
                "arrow_nodes": copy.deepcopy(state["arrows"][chosen]["nodes"]),
                "blackhole_death": is_blackhole_death
            })

        # Remove the entire arrow from board
        # Both successful escapes and blackhole deaths decrease wayBlocker lockTime
        coord_map = remove_arrow(chosen, state, coord_map, is_successful_escape=True)
        
        if is_blackhole_death:
            blackhole_deaths += 1
        
        # Remove unlocked wayBlockers
        wayBlockers_map = remove_unlocked_wayBlockers(state, wayBlockers_map)

        moves += 1

        if moves > 5000:  # safety
            if debug:
                print("\n⚠️ REACHED MOVE LIMIT (5000)")
            break
        
        if debug and moves >= 50:
            print("\n[Stopping debug after 50 moves]")
            break

    alpha_time = time.time() - start_time
    avg_branching = sum(branching_factors) / max(1, len(branching_factors))

    difficulty_score = min(
        10,
        (
            0.5 * math.log(1 + moves)
            + 0.7 * avg_branching
            + 0.4 * (1 if len(state["arrows"]) > 0 else 0)
        )
    )

    result = {
        "alpha_time": alpha_time,
        "moves": moves,
        "branching": avg_branching,
        "difficulty": difficulty_score,
        "win": len(state["arrows"]) == 0,
        "remaining_arrows": len(state["arrows"]),
        "blackhole_deaths": blackhole_deaths
    }

    if record_steps:
        result["steps"] = step_log

    return result


# ============================
#  Multi-run aggregator
# ============================
def simulate_level(level_json, level_name, runs=30, beta=10000, record_steps=False, debug=False):

    results = [simulate_one_run(level_json, record_steps=False, debug=False) for _ in range(runs)]

    one_run_log = simulate_one_run(level_json, record_steps=True, debug=debug) if record_steps else None

    avg_alpha = sum(r["alpha_time"] for r in results) / runs
    avg_moves = sum(r["moves"] for r in results) / runs
    win_rate = sum(1 for r in results if r["win"]) / runs

    summary = {
        "Level_Name": level_name,
        "Estimated_Player_Time": beta * avg_alpha,
        "Avg_Moves": avg_moves,
        "Win_Rate": win_rate
    }

    return summary, one_run_log


# ============================
#  Run single file
# ============================
def run_single_file(path, runs=50, record_steps=True, debug=False):
    with open(path, "r") as f:
        level = json.load(f)

    summary, log = simulate_level(level, os.path.basename(path), runs=runs, record_steps=record_steps, debug=debug)

    print("\n=== SUMMARY FOR", path, "===")
    print(json.dumps(summary, indent=4))

    # Save steps
    if record_steps and log:
        out_path = path.replace(".json", "_steps.json")
        with open(out_path, "w") as of:
            json.dump(log, of, indent=2)
        print("Saved steps:", out_path)


# ============================
#  Run folder and save CSV
# ============================
def run_folder(folder, runs=30, output_csv="results.csv"):
    rows = []

    # Get all json files and sort by numeric part
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    
    # Sort by extracting the number from filename (e.g., lv0.json -> 0)
    def extract_number(filename):
        import re
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_number)
    
    for fname in files:
        path = os.path.join(folder, fname)
        with open(path, "r") as f:
            level = json.load(f)

        summary, _ = simulate_level(level, fname, runs=runs)
        print(f"{fname} done.")
        rows.append(summary)

    # write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Level_Name", "Estimated_Player_Time", "Avg_Moves", "Win_Rate"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("CSV saved:", output_csv)

# ---------------------------
# If executed as script - examples
# ---------------------------
if __name__ == "__main__":
    # example: single file with debug on
    # run_single_file("/Users/hoangnguyen/Documents/py/ArrowPuzzle/level_2/lv73.json", runs=1, record_steps=True, debug=True)

    # example: run whole folder
    run_folder("/Users/hoangnguyen/Documents/py/ArrowPuzzle/level_2 (1)", runs=1, output_csv="full.csv")