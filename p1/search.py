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
        if problem.isGoal(node):
            return path
        
        for neighbor in problem.successorStates(node):
            if neighbor[0] not in visited:
                
                visited.append(neighbor[0])
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
    pq.push((problem.startingState(), [], 0))
    visited.add(problem.startingState())

    while not pq.isEmpty():
        node, path, curr_cost = pq.pop()
        if problem.isGoal(node):
            return path

        for neighbor in problem.successorStates(node):
            
            if neighbor[0] not in visited:
                copy_list = path.copy()
                copy_list.append(neighbor[1])
                pq.push((neighbor[0], copy_list, curr_cost + neighbor[2])) #Grabbing current node cost and neighbor cost for traveling
                visited.add(neighbor[0])

    return None   
        
        


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***

    pq = PriorityQueue()
    visited = list()
    start_heuristic = heuristic(problem.startingState(), problem)
    pq.push(((problem.startingState(), None, 0), [], 0, start_heuristic), start_heuristic)
    print(problem.startingState())
    visited.append(problem.startingState())

    while not pq.isEmpty():
        node, path, __, __ = pq.pop()
        if problem.isGoal(node[0]):

            return path
        
        for neighbor in problem.successorStates(node[0]):
            new_heuristic = heuristic(neighbor[0], problem)
            if neighbor[0] not in visited:
                #Grabbing current node cost and neighbor cost for traveling
                visited.append(neighbor[0])
                new_path = path.copy()
                new_path.append(neighbor[1])
                path_cost = problem.actionsCost(new_path)
                pq.push((neighbor, new_path, path_cost, new_heuristic), path_cost + new_heuristic)
    
    return None
