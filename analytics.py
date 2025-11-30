import json
import itertools
import csv
import networkx as nx
from collections import deque, defaultdict


# ============================================================
# 1. PARSER: LOAD LEVEL JSON
# ============================================================

def load_level(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. BUILD DEPENDENCY GRAPH
# ============================================================

def is_blocking(nodes_A, nodes_B):
    """
    Một mũi tên A bị chắn bởi B nếu:
    - Hai node đầu tiên của A tạo vector hướng, nhưng A bị chặn nếu B có bất kỳ node
      nằm trên hướng đó.

    Giả sử:
        A_tip = nodes_A[0]
        A_next = nodes_A[1]
        Hướng A = vector (dx, dy)
    Nếu B chứa điểm trên ray hướng của A → A depends on B.

    Lưu ý: Mô hình đơn giản theo mô tả của bạn.
    """
    A_tip = nodes_A[0]
    A_next = nodes_A[1]

    dx = A_next["x"] - A_tip["x"]
    dy = A_next["y"] - A_tip["y"]

    # Hướng chuẩn (horizontal/vertical)
    if dx != 0:
        direction = ("x", dx)
        axis = "x"
        orth = "y"
    else:
        direction = ("y", dy)
        axis = "y"
        orth = "x"

    # B chặn A nếu trong nodes_B có điểm phía trước A_tip theo hướng
    for p in nodes_B:
        if axis == "x":
            if p["y"] == A_tip["y"]:
                if dx > 0 and p["x"] > A_tip["x"]:
                    return True
                if dx < 0 and p["x"] < A_tip["x"]:
                    return True
        else:  # movement along y
            if p["x"] == A_tip["x"]:
                if dy > 0 and p["y"] > A_tip["y"]:
                    return True
                if dy < 0 and p["y"] < A_tip["y"]:
                    return True

    return False


def build_dependency_graph(level_json):
    G = nx.DiGraph()
    arrows = level_json["arrows"]

    for i in range(len(arrows)):
        G.add_node(i)

    # Check all pairs
    for i, j in itertools.permutations(range(len(arrows)), 2):
        A = arrows[i]["nodes"]
        B = arrows[j]["nodes"]
        if is_blocking(A, B):
            G.add_edge(i, j)  # j blocks i → i depends on j

    return G

# -------------------------------
# 4. GRAPH METRICS
# -------------------------------
def metric_total_arrows(data):
    return len(data["arrows"])


def metric_total_dependencies(G):
    return G.number_of_edges()


def metric_max_in_degree(G):
    return max([deg for _, deg in G.in_degree()], default=0)


def metric_max_out_degree(G):
    return max([deg for _, deg in G.out_degree()], default=0)


def metric_degree_variance(G):
    degrees = [deg for _, deg in G.degree()]
    if not degrees:
        return 0
    mean = sum(degrees) / len(degrees)
    var = sum((d - mean) ** 2 for d in degrees) / len(degrees)
    return var


def metric_edges_per_arrow(G):
    n = G.number_of_nodes()
    if n == 0:
        return 0
    return G.number_of_edges() / n


def metric_has_cycle(G):
    cycles = list(nx.simple_cycles(G))
    return len(cycles)

def metric_approx_max_dependency_depth(G, depth_limit=100):
    """
    Approximate longest path depth even if cycles exist
    - depth_limit: maximum depth to explore to avoid infinite loops
    """
    def dfs(node, visited):
        if node in visited:
            return 0  # stop at cycle
        if len(visited) >= depth_limit:
            return len(visited)
        visited.add(node)
        max_depth = 0
        for succ in G.successors(node):
            max_depth = max(max_depth, dfs(succ, visited.copy()))
        return 1 + max_depth

    longest = 0
    for node in G.nodes():
        longest = max(longest, dfs(node, set()))
    return longest


def metric_connected_components(G):
    UG = G.to_undirected()
    return nx.number_connected_components(UG)


def metric_dependency_density(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n <= 1:
        return 0
    return e / (n * (n - 1))



def export_metrics_to_csv(metrics_dict, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics_dict.items():
            writer.writerow([k, v])


# ============================================================
# ====== 6. MAIN FUNCTION ====================================
# ============================================================

def analyze_level_graph(json_path, csv_path="level_metrics.csv"):
    data = load_level(json_path)
    G = build_dependency_graph(data)
    metrics = {
        "total_arrows": metric_total_arrows(data),
        "total_dependencies": metric_total_dependencies(G),
        "max_in_degree": metric_max_in_degree(G),
        "max_out_degree": metric_max_out_degree(G),
        "degree_variance": metric_degree_variance(G),
        "edges_per_arrow": metric_edges_per_arrow(G),
        "approx_max_dependency_depth": metric_approx_max_dependency_depth(G),
        "has_cycle": metric_has_cycle(G),
        "connected_components": metric_connected_components(G),
        "dependency_density": metric_dependency_density(G),
    }

    export_metrics_to_csv(metrics, csv_path)

    print("CSV exported:", csv_path)
    return metrics, G

# ============================================================
# RUN AS SCRIPT
# ============================================================

if __name__ == "__main__":
    analyze_level_graph("/Users/hoangnguyen/Documents/py/Arrow/asset-game-level/lv50.json",
                        csv_path="/Users/hoangnguyen/Documents/py/Arrow/lv50_metrics.csv")
