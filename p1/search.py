"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueueWithFunction
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
    visited = set()
    parent = {} # Using parent dict as a way to traverse from start to finish
    for travel in problem.successorStates(problem.startingState()):
        stack.push(travel)

    while not stack.isEmpty():
        vertex = stack.pop()
        if problem.isGoal(vertex[0]):
            path = list() # Create path list from beginning to end
            while vertex is not None:
                path.append(vertex[1])
                vertex = parent.get(vertex)

            return path[::-1]
        if vertex not in visited:
            visited.add(vertex)
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
    visited.append(problem.startingState())
    queue.push(((problem.startingState(), [], 1)))

    while not queue.isEmpty():
        node, path, cost = queue.pop()
        print(node)
        if problem.isGoal(node):
            return path
        
        visited.append(node)
        for neighbor in problem.successorStates(node):
            if neighbor[0] not in visited:
                copy_list = path.copy()
                copy_list.append(neighbor[1])
                queue.push((neighbor[0], copy_list, cost)) #Grabbing current node cost and neighbor cost for traveling


    return None
def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***

    pq = PriorityQueueWithFunction(lambda item: item[2])
    visited = set()
    cost = 0
    pq.push((problem.startingState(), [], cost))
    visited.add(problem.startingState())

    while not pq.isEmpty():
        node, path, curr_cost = pq.pop()[1]
        if problem.isGoal(node):
            return path

        visited.add(node)
        for neighbor in problem.successorStates(node):
            print(neighbor)
            if neighbor[0] not in visited:
                copy_list = path.copy()
                copy_list.append(neighbor[1])
                next_cost = curr_cost + cost
                pq.push((neighbor[0], copy_list, next_cost)) #Grabbing current node cost and neighbor cost for traveling
    
    return None   
        
        


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***

    pq = PriorityQueue()
    visited = list()
    parent = {}
    for travel in problem.successorStates(problem.startingState()):
        pq.push(travel, travel[2])

    while not pq.isEmpty():
        node = pq.pop()
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
                     #Grabbing current node cost and neighbor cost for traveling
                    parent[neighbor] = node
                    pq.push(neighbor, neighbor[2] + node[2] + heuristic(neighbor[0], problem))
    
    return None
