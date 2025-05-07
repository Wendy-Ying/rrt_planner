import numpy as np
from rrt import RRT

if __name__ == "__main__":
    start = np.array([0, 0, 0, 0, 0, 0])
    goal = np.array([1.0, -0.5, 0.3, -1.0, 0.2, 0.5])

    planner = RRT(start, goal)
    path = planner.plan()

    if path:
        print("Path found!")
        for p in path:
            print(p)
    else:
        print("No valid path found.")
