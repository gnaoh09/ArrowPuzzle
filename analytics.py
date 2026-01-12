import os
import re
import json
import csv
import networkx as nx
import pandas as pd
from collections import defaultdict, deque

# ============================================================
# 1. PARSER: LOAD LEVEL JSON
# ============================================================
def load_level(path):
    # Only process files explicitly marked as original; otherwise, skip.
    if not path.endswith(".json"):
        print(f"Skip non-original file: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# 2. SPATIAL INDEX + BUILD DEPENDENCY GRAPH (OPTIMIZED)
# ============================================================
def interpolate_line(x1, y1, x2, y2):
    """Return all grid cells on the line from (x1,y1) to (x2,y2)."""
    points = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1

    if dx >= dy:
        err = dx / 2
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x2, y2))
    return points

def build_spatial_index(level_json):
    arrows = level_json["arrows"]
    coord_map = defaultdict(set)
    rows = defaultdict(set)
    cols = defaultdict(set)

    for idx, a in enumerate(arrows):
        nodes = a["nodes"]

        # Interpolate every segment
        full_cells = []
        for i in range(len(nodes) - 1):
            x1, y1 = nodes[i]["x"], nodes[i]["y"]
            x2, y2 = nodes[i+1]["x"], nodes[i+1]["y"]
            segment = interpolate_line(x1, y1, x2, y2)
            full_cells.extend(segment)

        # Add to spatial maps
        for (x, y) in full_cells:
            coord_map[(x, y)].add(idx)
            rows[y].add(idx)
            cols[x].add(idx)

    size = level_json.get("size", {})
    max_x = size.get("x", None)
    max_y = size.get("y", None)

    return coord_map, rows, cols, max_x, max_y

def normalize_direction(nodes):
    """Return normalized direction vector (dx, dy) of arrow head."""
    head = nodes[-1]
    prev = nodes[-2]

    dx = head["x"] - prev["x"]
    dy = head["y"] - prev["y"]

    # Normalize direction to steps
    if dx != 0: dx = 1 if dx > 0 else -1
    if dy != 0: dy = 1 if dy > 0 else -1

    return dx, dy

def build_dependency_graph(level_json):
    arrows = level_json["arrows"]
    coord_map, rows, cols, max_x, max_y = build_spatial_index(level_json)
    G = nx.DiGraph()
    n = len(arrows)

    for i in range(n):
        G.add_node(i)

    # Infer size if missing
    if max_x is None or max_y is None:
        xs, ys = [], []
        for a in arrows:
            for nd in a["nodes"]:
                xs.append(nd["x"])
                ys.append(nd["y"])
        max_x = max(xs) + 1
        max_y = max(ys) + 1

    # Build dependency
    for i, a in enumerate(arrows):
        nodes = a["nodes"]
        if len(nodes) < 2:
            continue

        # Correct: tip is the LAST node
        tip = nodes[-1]

        dx, dy = normalize_direction(nodes)
        x, y = tip["x"] + dx, tip["y"] + dy

        # Ray cast until out of bounds
        while 0 <= x < max_x and 0 <= y < max_y:
            if (x, y) in coord_map:
                for j in coord_map[(x, y)]:
                    if j != i:
                        G.add_edge(i, j)
            x += dx
            y += dy

    return G

# ============================================================
# 3. GRAPH METRICS (existing + optimized)
# ============================================================
def metric_total_arrows(data):
    return len(data["arrows"])

def metric_total_dependencies(G):
    return G.number_of_edges()

def metric_max_in_degree(G):
    return max((deg for _, deg in G.in_degree()), default=0)

def metric_edges_per_arrow(G):
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return G.number_of_edges() / n

def metric_has_cycle_fast(G):
    """
    Fast check for the presence of cycles (boolean). Uses networkx built-in O(V+E).
    Also return count of SCCs with size>1 (we use for cycle severity too).
    """
    is_dag = nx.is_directed_acyclic_graph(G)
    return (not is_dag)

def metric_dependency_density(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n <= 1:
        return 0.0
    return e / (n * (n - 1))

def metric_blocking_density_per_axis(level_json, G):
    """
    Blocking Density per Axis:
      - For each row: count how many unique arrows occupy that row
      - For each column: count how many unique arrows occupy that column
    Returns:
      - row_count_map, col_count_map, max_row_blockers, max_col_blockers, avg_row_blockers, avg_col_blockers
    """
    rows = defaultdict(set)
    cols = defaultdict(set)
    for idx, a in enumerate(level_json["arrows"]):
        for n in a["nodes"]:
            rows[n["y"]].add(idx)
            cols[n["x"]].add(idx)

    row_counts = [len(s) for s in rows.values()] if rows else [0]
    col_counts = [len(s) for s in cols.values()] if cols else [0]

    max_row = max(row_counts) if row_counts else 0
    max_col = max(col_counts) if col_counts else 0
    avg_row = sum(row_counts) / len(row_counts) if row_counts else 0.0
    avg_col = sum(col_counts) / len(col_counts) if col_counts else 0.0

    # Also compute fraction of rows/cols that are "crowded" (>= threshold)
    threshold = max(2, int(0.1 * len(level_json["arrows"])))  # heuristic
    crowded_rows = sum(1 for c in row_counts if c >= threshold)
    crowded_cols = sum(1 for c in col_counts if c >= threshold)

    return {
        "avg_row_blockers": avg_row,
        "avg_col_blockers": avg_col,
        "crowded_rows": crowded_rows,
        "crowded_cols": crowded_cols,
    }

def metric_critical_path_load(G):
    """
    Critical Path Load:
      - Condense SCCs into DAG (cG). Each node in cG has weight = size of SCC.
      - Compute longest path (in terms of original nodes) using DP on DAG.
      - Also compute number of edges that lie on any longest path (approx).
    Return:
      - longest_path_nodes_count
      - longest_path_scc_chain (list of condensed nodes forming one longest path)
      - critical_edges_count (approx count of original edges that are on longest path chain)
    """
    if G.number_of_nodes() == 0:
        return {
            "critical_longest_nodes": 0,
            "critical_edges_on_chain": 0
        }

    cG = nx.condensation(G)
    weights = {n: len(cG.nodes[n]["members"]) for n in cG.nodes()}

    topo = list(nx.topological_sort(cG))
    # dp_forward[n] = max nodes count path ending at n
    dp_forward = {n: weights[n] for n in cG.nodes()}
    parent = {n: None for n in cG.nodes()}  # store predecessor for one longest path
    for u in topo:
        for v in cG.successors(u):
            cand = dp_forward[u] + weights[v]
            if cand > dp_forward[v]:
                dp_forward[v] = cand
                parent[v] = u

    # find node with max dp_forward
    end_node = max(dp_forward, key=lambda k: dp_forward[k])
    longest_nodes = dp_forward[end_node]

    # reconstruct one longest chain of condensed nodes
    chain = []
    cur = end_node
    while cur is not None:
        chain.append(cur)
        cur = parent[cur]
    chain = list(reversed(chain))

    # count original edges that are internal to these SCCs chain or between consecutive SCCs
    # collect members:
    chain_members = set()
    for scc_idx in chain:
        chain_members.update(cG.nodes[scc_idx]["members"])
    # count edges among chain members in original G
    edges_on_chain = 0
    for u, v in G.edges():
        if u in chain_members and v in chain_members:
            edges_on_chain += 1

    return {
        "critical_longest_nodes": longest_nodes,
        "critical_edges_on_chain": edges_on_chain
    }

# ============================================================
# 5. EXPORT
# ============================================================
def export_metrics_to_csv(metrics_dict, out_path):
    rows = []
    for k, v in metrics_dict.items():
        # For complex objects (lists/dicts) convert to JSON string for CSV readability
        if isinstance(v, (list, dict)):
            rows.append([k, json.dumps(v, ensure_ascii=False)])
        else:
            rows.append([k, v])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerows(rows)

# ============================================================
# 6. MAIN ANALYSIS FUNCTION
# ============================================================
def analyze_level_graph(json_path, csv_path="level_metrics.csv"):
    data = load_level(json_path)
    if data is None:
        return None, None
    G = build_dependency_graph(data)

    # Basic metrics
    basic = {
        "total_arrows": metric_total_arrows(data),
        "total_edges": metric_total_dependencies(G),
        "max_in_degree": metric_max_in_degree(G),
        "edges_per_arrow": metric_edges_per_arrow(G),
        "dependency_density": metric_dependency_density(G),
        "has_cycle_bool": metric_has_cycle_fast(G),
    }
    mechanic = metric_mechanics_basic(data)
    # Blocking density per axis
    blocking_density = metric_blocking_density_per_axis(data, G)

    # Critical path load
    critical_path = metric_critical_path_load(G)

    metrics = {}
    metrics.update(basic)
    metrics.update(critical_path)
    metrics.update(mechanic)
    metrics.update(blocking_density)

    # Export CSV
    export_metrics_to_csv(metrics, csv_path)
    print("CSV exported:", csv_path)
    return metrics, G

# ============================================================
# 7. MECHANIC ANALYSIS FUNCTION
# ============================================================
def metric_mechanics_basic(level_json):
    way_blockers = level_json.get("wayBlockers", [])
    black_holes = level_json.get("blackHoles", [])

    # Deduplicate mechanics by coordinate so overlapping entries count once.
    def dedup_by_coord(items):
        seen = set()
        unique = []
        for item in items:
            pos = item.get("position") or {}
            # Support both nested position and flat x/y
            coord = (
                pos.get("x", item.get("x")),
                pos.get("y", item.get("y")),
            )
            if coord in seen:
                continue
            seen.add(coord)
            unique.append(item)
        return unique

    wb_unique = dedup_by_coord(way_blockers)
    bh_unique = dedup_by_coord(black_holes)

    num_wb = len(wb_unique)
    num_bh = len(bh_unique)

    def get_lock_time(wb):
        # lockTime can sit at top-level per provided example
        return wb.get("lockTime", 0)

    lock_times = [get_lock_time(wb) for wb in wb_unique]

    return {
        "num_wayBlockers": num_wb,
        "num_blackHoles": num_bh,
        "sum_wayBlocker_lockTime": sum(lock_times),
        "avg_wayBlocker_lockTime": (
            sum(lock_times) / num_wb if num_wb > 0 else 0.0
        ),
    }

# ============================================================
# 7. MECHANIC ANALYSIS FUNCTION
# ============================================================
def sort_level_files(files):
    """
    Sort files like lv0.json, lv1.json, lv10.json numerically by level index.
    """
    def extract_level_num(f):
        # Accept suffix variants: lv0_original.json, lv0_ox.json, lv0.json
        match = re.search(r"lv(\d+)(?:_.*)?\.json$", f)
        return int(match.group(1)) if match else float("inf")

    # Stable sort keeps relative order among equal keys; fallback key keeps non-matching at end
    return sorted(files, key=lambda f: (extract_level_num(f), f))

def analyze_levels_in_folder(folder_path, csv_path="all_levels_metrics.csv", n_level=None):
    """
    Analyze multiple level JSON files in a folder and save metrics to a CSV.
    
    Args:
        folder_path (str): Path to folder containing level JSON files.
        csv_path (str): Output CSV path.
        n_level (int, optional): Maximum number of levels to process. If None, process all.
    """
    all_metrics = []
    # Process only original JSONs and keep deterministic numeric order
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    files = sort_level_files(files)
    if n_level is not None:
        files = files[:n_level]


    for file_name in files:
        json_path = os.path.join(folder_path, file_name)
        metrics, _ = analyze_level_graph(json_path)
        if metrics is None:
            continue
        metrics_row = {"Level_Name": file_name}
        metrics_row.update(metrics)
        all_metrics.append(metrics_row)

    # Convert to DataFrame and export
    df = pd.DataFrame(all_metrics)
    df.to_csv(csv_path, index=False)
    print(f"CSV exported for {len(all_metrics)} levels to:", csv_path)
    return df



if __name__ == "__main__":
    # json_path = "/Users/hoangnguyen/Documents/py/ArrowPuzzle/asset-game-level/lv8.json"
    # csv_path = "/Users/hoangnguyen/Documents/py/ArrowPuzzle/lv8_metrics_new.csv"
    # metrics, G = analyze_level_graph(json_path, csv_path=csv_path)

    folder_path = "/Users/hoangnguyen/Documents/py/ArrowPuzzle/level_2"
    csv_path = "/Users/hoangnguyen/Documents/py/ArrowPuzzle/100lv_1612_2.csv"
    df = analyze_levels_in_folder(folder_path, csv_path=csv_path, n_level=None)
    