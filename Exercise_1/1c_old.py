import os
import psutil
from collections import deque

# ── helpers ──────────────────────────────────────────────────────────────
def get_used_pct():
    return psutil.virtual_memory().percent          # system-wide, matches Task Manager

def get_ram_info():
    vm = psutil.virtual_memory()
    used_gb  = (vm.total - vm.available) / (1024 ** 3)
    total_gb = vm.total / (1024 ** 3)
    return used_gb, total_gb, vm.percent

# ── graph generation ─────────────────────────────────────────────────────
print("=== Generating graph ===")
total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
print(f"Total RAM: {total_ram_gb:.1f} GB")

graph = {}
n = 0

while True:
    neighbor_ids = list(range(max(0, n - 1000), n))
    graph[n] = neighbor_ids

    for neighbor in neighbor_ids:
        graph[neighbor].append(n)

    n += 1

    if n % 10_000 == 0:
        used_gb, total_gb, pct = get_ram_info()
        print(f"Nodes: {n:,} | RAM: {used_gb:.2f} GB / {total_gb:.2f} GB ({pct:.1f}%)")

        if pct > 50:
            print(f"Reached {pct}% RAM — stopping generation at {n:,} nodes")
            break

print(f"\nGraph ready: {len(graph):,} nodes")
total_edges = sum(len(v) for v in graph.values()) // 2
print(f"Total edges: {total_edges:,}")

# ── BFS ───────────────────────────────────────────────────────────────────
import time

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