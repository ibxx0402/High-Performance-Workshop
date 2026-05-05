from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# ── Build the DAG from the image ──────────────────────────────────────────
graph = {
    0: [1, 2, 4],
    1: [3, 4],
    2: [4, 5],
    3: [6],
    4: [6, 7, 8],
    5: [7],
    6: [8],
    7: [8],
    8: []
}

# ── Parallel BFS by level ─────────────────────────────────────────────────
def parallel_bfs(graph, start, num_threads=5):
    visited = set()
    visited_lock = threading.Lock()

    current_level = [start]
    visited.add(start)
    level_num = 0

    print(f"=== Parallel BFS with {num_threads} threads ===\n")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        while current_level:
            print(f"Level {level_num}: processing nodes {current_level}")

            next_level_sets = [set() for _ in current_level]

            def process_node(args):
                idx, node = args
                local_neighbors = []
                for neighbor in graph[node]:
                    with visited_lock:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            local_neighbors.append(neighbor)
                next_level_sets[idx] = local_neighbors
                print(f"  Worker processing node {node} → neighbors: {local_neighbors}")

            # all nodes in this level are processed in parallel
            list(executor.map(process_node, enumerate(current_level)))

            # flatten results into next level
            current_level = [n for sublist in next_level_sets for n in sublist]
            level_num += 1

    print(f"\nBFS complete! Visited: {sorted(visited)}")

s = time.time()
parallel_bfs(graph, start=0, num_threads=5)
print(time.time()-s)

#0.0005078315734863281
#0.0004017353057861328