"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    stack = Stack()
    visited = list()
    parent = {}
    for travel in problem.successorStates(problem.startingState()):
        stack.push(travel)

    while not stack.isEmpty():
        vertex = stack.pop()
        if problem.isGoal(vertex[0]):
            path = list()
            while vertex is not None:
                path.append(vertex[1])
                vertex = parent.get(vertex)

            return path[::-1]
        if vertex not in visited:
            visited.append(vertex)
            for travel in problem.successorStates(vertex[0]):
                if travel not in visited:
                    stack.push(travel)
                    parent[travel] = vertex



    return None

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    queue = Queue()
    visited = list()
    parent = {}

    for travel in problem.successorStates(problem.startingState()):
        queue.push(travel)

    while not queue.isEmpty():
        node = queue.pop()
        if problem.isGoal(node[0]):
            path = list()
            while node is not None:
                path.append(node[1])
                node = parent.get(node)
            
            return path[::-1]

        if node not in visited:
            visited.append(node)
            for neighbor in problem.successorStates(node[0]):
                if neighbor not in visited:
                    queue.push(neighbor)
                    parent[neighbor] = node

    return None
def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***

    pq = PriorityQueue()
    visited = list()

    for travel in problem.successorStates(problem.startingState()):
        pq.push(travel, travel[2])

    while not pq.isEmpty():
        node = pq.pop()
        


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()
