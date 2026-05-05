import os
import psutil
import time
from collections import deque
from mpi4py import MPI

# ── MPI setup ────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()    # which process am I?
size = comm.Get_size()    # how many processes total?

# ── config ────────────────────────────────────────────────────────────────
TARGET_NODES = 3_500_000

# ── graph generation (only rank 0 builds it, then broadcasts) ─────────────
if rank == 0:
    print("=== Generating graph ===")
    print(f"Target nodes: {TARGET_NODES:,}")
    print(f"MPI processes: {size}")

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
else:
    graph = None

# ── broadcast graph to all processes ─────────────────────────────────────
graph = comm.bcast(graph, root=0)

# ── split BFS runs across processes ──────────────────────────────────────
# e.g. with 4 processes and 10 runs: each process handles ~2-3 runs
all_runs = list(range(1, 11))
my_runs = all_runs[rank::size]   # e.g. rank 0 → [1,5,9], rank 1 → [2,6,10], etc.

if rank == 0:
    print(f"\n=== Running BFS (10 iterations across {size} MPI processes) ===")

start_node = max(graph.keys())
local_times = []

for run in my_runs:
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
    local_times.append(elapsed)
    print(f"  [Rank {rank}] Run {run:>2}/10 | Nodes visited: {nodes_visited:,} | Time: {elapsed:.2f}s")

# ── gather all times back to rank 0 ──────────────────────────────────────
all_times = comm.gather(local_times, root=0)

if rank == 0:
    # flatten list of lists
    times = [t for sublist in all_times for t in sublist]
    avg_time = sum(times) / len(times)
    print(f"\nBFS complete!")
    print(f"  Nodes visited : {nodes_visited:,}")
    print(f"  Runs          : 10")
    print(f"  Avg time      : {avg_time:.2f} seconds")
    print(f"  Min time      : {min(times):.2f} seconds")
    print(f"  Max time      : {max(times):.2f} seconds")