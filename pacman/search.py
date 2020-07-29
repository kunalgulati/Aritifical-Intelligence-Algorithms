# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

"""
<Your feedback goes here>

"""
#####################################################
#####################################################

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    #Returns all the CHILDREN of the Current Node 
    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# Reference : https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
def depthFirstSearch(problem):
    """
    Q1.1
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print ( problem.getStartState() )
    You will get (5,5)

    print (problem.isGoalState(problem.getStartState()) )
    You will get True

    print ( problem.getSuccessors(problem.getStartState()) )
    You will get [((x1,y1),'South',1),((x2,y2),'West',1)]
    """
    "*** YOUR CODE HERE ***"

    visitedNode = [] #Already Visited Node
    pathToGoal = [] #Corresponding path of the visited Nodes 
    stackContainer = util.Stack()

    #Check if the starting state is GOAL state, and
    # if TRUE, then return an empty array, showing that we are already at our goal state
    if problem.isGoalState(problem.getStartState()):
      return []
    
    stackContainer.push((problem.getStartState(), []))
    
    while(True):
      #Soltuion Doesn't exist 
      if not stackContainer:
        print("Empty Stack")
        return []
      
      #Current State of the PacMan
      candidate = stackContainer.pop()
      coordinate = candidate[0]
      pathToGoal = candidate[1]

      # Mark the current node as visited
      visitedNode.append(coordinate)

      # check if this is our Gaol, if TRUE, then return the path
      if problem.isGoalState(coordinate):
        return pathToGoal

      # If current, coordinates are not the goal state, then we need to keep checking 
      # ex. successor = ( (36,16) , "South" 1)
      successor = problem.getSuccessors(coordinate)

      # Check if the successor exist, and if yes, then go and visit successor and it's neighbors 
      if successor:
        # This loop Visits all the childer(succesor) of the current Node
        for eachSuccessor in successor:
          if eachSuccessor[0] not in visitedNode:
            # Path to Goal is the path we took to get to this node
            updatePath = pathToGoal + [eachSuccessor[1]]
            stackContainer.push((eachSuccessor[0], updatePath))

def breadthFirstSearch(problem):
    """
    Q1.2
    Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visitedNode = set() #Already Visited Node
    pathToGoal = [] #Corresponding path of the visited Nodes 
    queueContainer = util.Queue()

    #Check if the starting state is GOAL state, and
    # if TRUE, then return an empty array, showing that we are already at our goal state
    if problem.isGoalState(problem.getStartState()):
      return []
    # Initialize the Queue, as we initialize it with the Current node, so the path to the current node is []
    queueContainer.push((problem.getStartState(), []))
    
    while(True):
      #Soltuion Doesn't exist 
      if queueContainer.isEmpty():
        return []
      
      #Current State of the PacMan
      candidate = queueContainer.pop()
      coordinate = candidate[0]
      pathToGoal = candidate[1]
      # Skip the already visited Node
      if str(coordinate) in visitedNode:
        continue

      # Mark the current node as visited
      visitedNode.add(str(coordinate))

      # check if this is our Gaol, if TRUE, then return the path
      if problem.isGoalState(coordinate):
        return pathToGoal

      # If current, coordinates are not the goal state, then we need to keep checking 
      # ex. successor = ( (36,16) , "South" 1)
      successor = problem.getSuccessors(coordinate)

      # Check if the successor exist, and if yes, then go and visit successor and it's neighbors 
      if successor:
        # This loop Visits all the childer(succesor) of the current Node
        for eachSuccessor in successor:
          # print(eachSuccessor[0][0])
          #if eachSuccessor[0] not in visitedNode:
          if eachSuccessor[0][0] not in visitedNode:
            # Path to Goal is the path we took to get to this node
            updatePath = pathToGoal + [eachSuccessor[1]]
            queueContainer.push((eachSuccessor[0], updatePath))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# Reference : https://www.redblobgames.com/pathfinding/a-star/introduction.html
def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Q1.3
    Search the node that has the lowest combined cost and heuristic first."""
    """Call heuristic(s,problem) to get h(s) value."""
    "*** YOUR CODE HERE ***"

    visitedNode = set() #Already Visited Node
    pathToGoal = [] #Corresponding path of the visited Nodes 
    priorityQueueContainer = util.PriorityQueue()

    #Check if the starting state is GOAL state, and
    # if TRUE, then return an empty array, showing that we are already at our goal state
    if problem.isGoalState(problem.getStartState()):
      return []
    # Initialize the Queue, as we initialize it with the Current node, so the path to the current node is []
    priorityQueueContainer.push((problem.getStartState(), []), 0)
    
    while(True):
      #Soltuion Doesn't exist 
      if priorityQueueContainer.isEmpty():
        return []
      
      #Current State of the PacMan
      candidate = priorityQueueContainer.pop()
      coordinate = candidate[0]
      pathToGoal = candidate[1]

      # Skip the already visited Node
      if str(coordinate) in visitedNode:
        continue

      # Mark the current node as visited
      visitedNode.add(str(coordinate))

      # check if this is our Gaol, if TRUE, then return the path
      if problem.isGoalState(coordinate):
        return pathToGoal

      # If current, coordinates are not the goal state, then we need to keep checking 
      # ex. successor = ( (36,16) , "South" 1)
      successor = problem.getSuccessors(coordinate)

      # Check if the successor exist, and if yes, then go and visit successor and it's neighbors 
      if successor:
        # This loop Visits all the childer(succesor) of the current Node
        for eachSuccessor in successor:
          new_cost = problem.getCostOfActions(pathToGoal) + eachSuccessor[2]
          new_path_cost = problem.getCostOfActions(pathToGoal + [eachSuccessor[1]])
          # if eachSuccessor[0] not in visitedNode or new_cost < new_path_cost:
          if eachSuccessor[0][0] not in visitedNode or new_cost < new_path_cost:
            priority = new_cost + heuristic(eachSuccessor[0] , problem)
            # Path to Goal is the path we took to get to this node
            updatePath = pathToGoal + [eachSuccessor[1]]
            priorityQueueContainer.push((eachSuccessor[0], updatePath), priority)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch