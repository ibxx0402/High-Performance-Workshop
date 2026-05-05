import os
import psutil
import time
from collections import deque

# ── config ────────────────────────────────────────────────────────────────
TARGET_NODES = 350_000   # <-- change this to however many nodes you want

# ── graph generation ─────────────────────────────────────────────────────
print("=== Generating graph ===")
print(f"Target nodes: {TARGET_NODES:,}")

graph = {i: [] for i in range(TARGET_NODES)}

for n in range(TARGET_NODES):
    neighbor_ids = list(range(max(0, n - 1000), n))
    graph[n] = neighbor_ids
    for neighbor in neighbor_ids:
        graph[neighbor].append(n)

    if n % 10_000 == 0 and n > 0:
        print(f"  Progress: {n:,} / {TARGET_NODES:,} nodes")

print(f"\nGraph ready: {len(graph):,} nodes")
total_edges = sum(len(v) for v in graph.values()) // 2
print(f"Total edges: {total_edges:,}")

# ── BFS ───────────────────────────────────────────────────────────────────
print("\n=== Running BFS (10 iterations) ===")
start_node = max(graph.keys())
print(f"Starting from node {start_node:,}")

times = []

for run in range(1, 11):
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    nodes_visited = 0
    t_start = time.time()

    while queue:
        node = queue.popleft()
        nodes_visited += 1
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    elapsed = time.time() - t_start
    times.append(elapsed)
    print(f"  Run {run:>2}/10 | Nodes visited: {nodes_visited:,} | Time: {elapsed:.2f}s")

avg_time = sum(times) / len(times)
print(f"\nBFS complete!")
print(f"  Nodes visited : {nodes_visited:,}")
print(f"  Runs          : 10")
print(f"  Avg time      : {avg_time:.2f} seconds")
print(f"  Min time      : {min(times):.2f} seconds")
print(f"  Max time      : {max(times):.2f} seconds")