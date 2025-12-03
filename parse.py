import json
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
# from pyvis.network import Network

def load_level(path):
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------------------
# Determine arrow direction from first 2 nodes
# -----------------------------------------

def get_direction(nodes):
    x1, y1 = nodes[-2]["x"], nodes[-2]["y"]
    x2, y2 = nodes[-1]["x"], nodes[-1]["y"]

    dx = x2 - x1
    dy = y2 - y1

    # Cardinal directions
    if dx == 0 and dy > 0:  return "UP"
    if dx == 0 and dy < 0:  return "DOWN"
    if dy == 0 and dx > 0:  return "RIGHT"
    if dy == 0 and dx < 0:  return "LEFT"

    # Diagonal (45 degrees)
    if abs(dx) == abs(dy):
        if dx > 0 and dy > 0: return "DIAG_UP_RIGHT"
        if dx > 0 and dy < 0: return "DIAG_DOWN_RIGHT"
        if dx < 0 and dy > 0: return "DIAG_UP_LEFT"
        if dx < 0 and dy < 0: return "DIAG_DOWN_LEFT"

    return None

def ray_blocks_arrow(A_head_x, A_head_y, direction, B_nodes):
    B_set = {(n["x"], n["y"]) for n in B_nodes}

    x, y = A_head_x, A_head_y

    for _ in range(100):  # maximum grid size
        if direction == "UP":           y += 1
        elif direction == "DOWN":       y -= 1
        elif direction == "RIGHT":      x += 1
        elif direction == "LEFT":       x -= 1
        elif direction == "DIAG_UP_RIGHT":   x += 1; y += 1
        elif direction == "DIAG_UP_LEFT":    x -= 1; y += 1
        elif direction == "DIAG_DOWN_RIGHT": x += 1; y -= 1
        elif direction == "DIAG_DOWN_LEFT":  x -= 1; y -= 1

        # collision check
        if (x, y) in B_set:
            return True

    return False

def is_blocked(A, B):
    nodesA = A["nodes"]
    nodesB = B["nodes"]

    head = nodesA[-1]  # last node is head in gameplay
    hx, hy = head["x"], head["y"]

    direction = get_direction(nodesA)
    if direction is None:
        return False

    return ray_blocks_arrow(hx, hy, direction, nodesB)


# -----------------------------------------
# Build dependency graph
# -----------------------------------------

def build_dependency_graph(level):
    arrows = level["arrows"]
    graph = defaultdict(list)

    for i, A in enumerate(arrows):
        for j, B in enumerate(arrows):
            if i == j:
                continue
            if is_blocked(A, B):
                graph[i].append(j)   # arrow i depends on j

    return graph

# -----------------------------------------
# Print result
# -----------------------------------------

def print_graph(graph):
    if not graph:
        print("No dependencies found")
        return
    for arrow, deps in graph.items():
        print(f"Arrow {arrow} depends on {deps}")


def draw_dependency_graph(graph, arrows):
    G = nx.DiGraph()

    # Add nodes (arrow ids)
    for i in range(len(arrows)):
        color = arrows[i]["color"]
        G.add_node(i, label=f"{i}\ncolor={color}")

    # Add edges
    for a, deps in graph.items():
        for b in deps:
            G.add_edge(a, b)

    # Layout for nice drawing
    pos = nx.spring_layout(G, seed=42, k=0.7)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#87CEEB")

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20)

    # Labels
    labels = {i: f"A{i}\nC{arrows[i]['color']}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("ArrowPuzzle Dependency Graph")
    plt.axis("off")
    plt.show()


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    level = load_level("/Users/hoangnguyen/Documents/py/ArrowPuzzle/asset-game-level/lv8.json")
    graph = build_dependency_graph(level)
    # draw_dependency_graph(graph, level["arrows"])
    # draw_dependency_graph_interactive(graph, level["arrows"])
    print_graph(graph)
