import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

<<<<<<< HEAD
        newPosition = successorGameState.getPacmanPosition()
        # oldPosition = currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        old_food_ct = oldFood.count()
        new_food_ct = newFood.count()
        # oldScore = currentGameState.getScore()
        newScore = successorGameState.getScore()

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        all_food = newFood.asList()
        curr_food = []

        # check if adversary is near
        for ghost in successorGameState.getGhostStates():
            isBrave = ghost.isBraveGhost()
            ghostPosition = ghost.getPosition()

            if not isBrave:
                continue
            else:
                if distance.manhattan(newPosition, ghostPosition) < 2:
                    # run away
                    return -999

        # get distance from pacman to all food
        for food in all_food:
            curr_food.append(1.0 / distance.euclidean(newPosition, food))
        curr_food.sort()

        # if pacman isnt moving or food is not being eaten --> penalize
        
        if oldFood.count() == newFood.count():
            return curr_food[-1] - abs(newScore)
        else:
            if new_food_ct == 0:
                return 0
            return newScore + curr_food[0]
        

=======
        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        # *** Your Code Here ***
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            
            if ghost.isBraveGhost():
                if distance.manhattan(newPosition, ghostPos) < 2:
                    return -999
            
        distance_food = []
        foodLocation = newFood.asList()
        for food in foodLocation:
            distance_food.append(1/ distance.manhattan(newPosition, food))
        
        
        distance_food.sort()   
        if oldFood.count() != newFood.count():
            if newFood.count() == 0:
                return 0
            return successorGameState.getScore() + distance_food[0]
        
        else:
            return distance_food[-1] - abs(successorGameState.getScore())
>>>>>>> f54075a5d05c49b15a927b34ab0a4367f741b282

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

<<<<<<< HEAD
    
    def getAction(self, gameState):

        def terminalState(self, gameState, depth):
            if ((gameState.isWin() or gameState.isLose()) or depth == self.getTreeDepth()):
                return self.getEvaluationFunction()(gameState)
        
        def value(self, gameState, depth, index):
            if (terminalState(self, gameState, depth)):
                return self.getevaluationFunction()(gameState)
            
            else:
                if index != 0:
                    return minValue(self, gameState, depth, index)
                else:
                    return maxValue(self, gameState, depth, index)
                

        def minValue(self, gameState, depth, index):
            numActions = gameState.getLegalActions(index)
            minValue = float('inf')
            numAgents = gameState.getNumAgents()
            if not numActions:
                return self.getEvaluationFunction()(gameState)
            else:
                for action in numActions:
                    newSuccessor = gameState.generateSuccessor(index, action)
                    if index + 1 == numAgents:
                        minValue = min(minValue, value(newSuccessor, depth + 1, 0))
                    else:
                        minValue = min(minValue, value(newSuccessor, depth, index + 1))
                
                return minValue 
            
            

    


    


=======
>>>>>>> f54075a5d05c49b15a927b34ab0a4367f741b282
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
