import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core.directions import Directions


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
        
    def getAction(self, gameState):
        value = float('-inf')
        newAction = None
        for action in gameState.getLegalActions(0):
            if (action == Directions.STOP):
                continue
            newSuccessor = gameState.generateSuccessor(0, action)
            new_value = self.value(newSuccessor, 0, 1)
            if new_value > value:
                value = new_value
                newAction = action
        return newAction
        
    def value(self, gameState, depth, index):
        if (self.terminalState(gameState, depth)):
            return self.getEvaluationFunction()(gameState)
        
        else:
            if index != 0:
                return self.minValue(gameState, depth, index)
            else:
                return self.maxValue(gameState, depth, index)
                
                
    def terminalState(self, gameState, depth):
        if ((gameState.isWin() or gameState.isLose()) or depth == self.getTreeDepth()):
                return self.getEvaluationFunction()(gameState)
        
        
                

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
                    minValue = min(minValue, self.value(newSuccessor, depth + 1, 0))
                else:
                    minValue = min(minValue, self.value(newSuccessor, depth, index + 1))
            
            return minValue 
        
    def maxValue(self, gameState, depth, index):
        numActions = gameState.getLegalActions(index)
        maxValue = float('-inf')
        if not numActions:
            return self.getEvaluationFunction()(gameState)
        else:
            for action in numActions:
                newSuccessor = gameState.generateSuccessor(index, action)
                maxValue = max(maxValue, self.value(newSuccessor, depth, index + 1))
                
            return maxValue

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
        # Create alpha and beta values to hold during execution
        self.alpha = float('-inf')
        self.beta = float('inf')
        
    def terminalState(self, gameState, depth):
        if ((gameState.isWin() or gameState.isLose()) or depth == self.getTreeDepth()):
                return self.getEvaluationFunction()(gameState)
            
    def value(self, gameState, alpha, beta, depth, index):
        if self.terminalState(gameState, depth):
            return self.getEvaluationFunction()(gameState)
        if index == 0:
            return self.maxValue(gameState, alpha, beta, depth, index)
        else:
            return self.minValue(gameState, alpha, beta, depth, index)
        
    def maxValue(self, gameState, alpha, beta, depth, index):
        numActions = gameState.getLegalActions(index)
        maxValue = float('-inf')
        
        if not numActions:
            return self.getEvaluationFunction()(gameState)
        else:
            for action in numActions:
                newSuccessor = gameState.generateSuccessor(index, action)
                maxValue = max(maxValue, self.value(newSuccessor, alpha, beta, depth, index + 1))
                if maxValue >= beta:
                    return maxValue
                alpha = max(alpha, maxValue)
            
            return maxValue
        
    def minValue(self, gameState, alpha, beta, depth, index):
        numActions = gameState.getLegalActions(index)
        minValue = float('inf')
        numAgents = gameState.getNumAgents()

        if not numActions:
            return self.getEvaluationFunction()(gameState)
        
        else:
            for action in numActions:
                newSuccessor = gameState.generateSuccessor(index, action)
                if index + 1 != numAgents:
                    minValue = min(minValue, self.value(newSuccessor, alpha, beta, depth, index + 1))
                else:
                    minValue = min(minValue, self.value(newSuccessor, alpha, beta, depth + 1, 0))
                    
                if minValue <= alpha:
                    return minValue
                else:
                    beta = min(beta, minValue)
            return minValue
        
        
    def getAction(self, gameState):
        returnedAction = None
        maxScore = float('-inf')
        for action in gameState.getLegalActions(0):
            if action == Directions.STOP:
                continue
            else:
                newSuccessor = gameState.generateSuccessor(0, action)
                newScore = self.value(newSuccessor, self.alpha, self.beta, 0, 1)

                if newScore > maxScore:
                    maxScore = newScore
                    returnedAction = action
                
        return returnedAction
    
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
