import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time
import os
import heapq

class Node():
    def __init__(self,parent = None,position = None,g = 0,h = 0):
        self.parent = parent #type Node
        self.position = position #tuple (x,y)
        self.g = g
        self.h = h
    def __eq__(self, otherNode): # use == operator to call this for nodes to test if position is equal
        return otherNode.position == self.position
    def __lt__(self, otherNode): # use < operator to call this for nodes to test if position is less than
        if self.f == otherNode.f: #tie breaker for same f value
            return self.g>=otherNode.g #change this line to edit g-value tie breaking testing
        else:
            return self.f<otherNode.f
    @property
    def f(self):#automatically adds g + h when the value of f is needed, no need to manual set f=g+h
        return self.g + self.h

def createBtrackingMaze(width=101,height=101,save=False,imgfname=None,txtfname=None):
    shape=(width,height)
    print('createBtrackingMaze: maze size of ' + str(shape))
    maze = np.ones(shape, dtype=bool) #maze matrix
    visited = np.zeros(shape, dtype=bool) #visited matrix
    stack = [] #cells that should be visited
    current= rand.randrange(0,width-1,2),rand.randrange(0,height-1,2)
    stack.append(current)
    print('createBtrackingMaze: chose ' + str(current) + ' as starting node')
    start = time.time()
    while stack:#while stack is not empty
        nebs = [] #current unvisited neighbors for each iteration
        walls=[] #walls in between neighbors
        A,B=current
        ''' If unvisited neighbor exists, add to temporary list '''
        if A-2 in range(width) and visited[A-2,B]==0:
            nebs.append([A-2,B])
            walls.append([A-1,B])
        if A+2 in range(width) and visited[A+2,B]==0:
            nebs.append([A+2,B])
            walls.append([A+1,B])
        if B-2 in range(height) and visited[A,B-2]==0:
            nebs.append([A,B-2])
            walls.append([A,B-1])
        if B+2 in range(height) and visited[A,B+2]==0:
            nebs.append([A,B+2])
            walls.append([A,B+1])

        ''' If temporary list is not empty, mark current node as visited and choose a random neighbor to add to stack and explore next '''
        if nebs:
            maze[A,B]=0
            visited[A,B]=1
            index = rand.choice(range(len(nebs)))
            A,B=walls[index]
            maze[A,B]=0
            visited[A,B]=1
            current=nebs[index]
            stack.append(current)
        else: #if list temporary list is empty, mark current node as visited and unblock wall
            visited[A,B]=1
            maze[A,B]=0
            current=stack.pop()
    print('createBtrackingMaze: time to generate maze was %s seconds' %(time.time()-start))
    plt.figure()
    plt.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    if not save:    
        plt.show()
    else:
        try:
            plt.savefig(imgfname)
            np.savetxt(txtfname,maze,fmt='%d')
            print('sucessfully saved %s and %s' %(imgfname,txtfname))
        except:
            print('Error saving file with given filename')

def createRandMaze(width=101,height=101,prob=0.2,save=False,imgfname=None,txtfname=None): #prob is probability of a wall appearing
    shape=(width,height)
    print('createRandMaze: maze size of ' + str(shape))
    maze = np.random.choice([0,1],size=shape,p=[1-prob,prob])
    plt.figure()
    plt.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    if not save:    
        plt.show()
    else:
        try:
            plt.savefig(imgfname)
            np.savetxt(txtfname,maze,fmt='%d')
            print('sucessfully saved %s and %s' %(imgfname,txtfname))
        except:
            print('Error saving file with given filename')

def savemazes(num=1,fname_template='maze'):#num of mazes to save, name of files with 0-(num-1) as ending
    if not os.path.exists('generated_mazes/images'):
        os.makedirs('generated_mazes/images')
    if not os.path.exists('generated_mazes/maze_files'):
        os.makedirs('generated_mazes/maze_files')
    for i in range(int(num/2)):
        createBtrackingMaze(save=True,imgfname=('generated_mazes/images/'+str(fname_template)+str(i)+'.png'),txtfname=('generated_mazes/maze_files/'+str(fname_template)+str(i)+'.txt'))
    for i in range(int(num/2)):
        createRandMaze(save=True,imgfname=('generated_mazes/images/'+str(fname_template)+str(i+int(num/2))+'.png'),txtfname=('generated_mazes/maze_files/'+str(fname_template)+str(i+int(num/2))+'.txt'))

def loadmaze(num=0,fname_template='maze'):
    try:
        maze=np.loadtxt('generated_mazes/maze_files/'+fname_template+str(num)+'.txt')
        return maze
    except:
        print('Error loading maze from file')

def getHeuristics(maze,goal):#returns numpy matrix of manhatten distances of each node to goal node
    X,Y = maze.shape
    gX,gY=goal
    heaurstics = np.zeros((X,Y),dtype=int) #visited matrix
    for y in range(Y):
        for x in range(X):
            heaurstics[x,y] = (abs(gX-x)+abs(gY-y))
    return heaurstics  
def AstarCompute(maze,start,goal,heuristics, _debug=True):
    startTime = time.time()
    X,Y = maze.shape
    openList = []
    expanded = np.zeros(maze.shape, dtype=bool) #closed list
    startNode = Node(position=start,g = 0,h = heuristics[start])
    goalNode = Node(position=goal,g = 9999,h = 0)
    heapq.heappush(openList,startNode)
    visitedNodes = 0
    while(openList):
        '''get node in openlist with lowest f value'''
        current=heapq.heappop(openList)
        if expanded[current.position]==1:#if already explored, pop another node
            continue
        expanded[current.position]=1
        A,B = current.position
        '''right after appending to closed list, check if appended node is the goal and return if it is'''
        if current == goalNode:
            break
        for x in [(1,0),(0,1),(-1,0),(0,-1)]:
            if A+x[0] in range(X) and B+x[1] in range(Y) and maze[A+x[0],B+x[1]]==0:#if in range and not a wall, add to temporary list
                newNode = Node(parent=current, position=(A+x[0],B+x[1]), g=current.g+1, h=heuristics[(A+x[0],B+x[1])]) 
                visitedNodes = visitedNodes +1
                heapq.heappush(openList,newNode)            
    endTime = time.time()
    if not openList:
        print('AstarCompute: No solution')
        return None
    if _debug:
        print('AstarCompute: time to A* search maze was %s seconds' %(endTime-startTime))
        print('AstarCompute: %s nodes visited' %visitedNodes)
        print('AstarCompute: %s nodes expanded' %np.count_nonzero(expanded))
    return current #linked list end

def AdaptiveAstarCompute(maze,start,goal,heuristics, _debug=True):
    startTime = time.time()
    X,Y = maze.shape
    openList = []
    closedList = [] #returned for updating heuristics
    expanded = np.zeros(maze.shape, dtype=bool) #closed list
    startNode = Node(position=start,g = 0,h = heuristics[start])
    goalNode = Node(position=goal,g = 9999,h = 0)
    heapq.heappush(openList,startNode)
    visitedNodes = 0
    while(openList):
        '''get node in openlist with lowest f value'''
        current=heapq.heappop(openList)
        if expanded[current.position]==1:#if already explored, pop another node
            continue
        expanded[current.position]=1
        closedList.append(current)
        A,B = current.position
        '''right after appending to closed list, check if appended node is the goal and return if it is'''
        if current == goalNode:
            break
        for x in [(1,0),(0,1),(-1,0),(0,-1)]:
            if A+x[0] in range(X) and B+x[1] in range(Y) and maze[A+x[0],B+x[1]]==0:#if in range and not a wall, add to temporary list
                newNode = Node(parent=current, position=(A+x[0],B+x[1]), g=current.g+1, h=heuristics[(A+x[0],B+x[1])]) 
                visitedNodes = visitedNodes +1
                heapq.heappush(openList,newNode)            
    endTime = time.time()
    if not openList:
        print('AdaptiveAstarCompute: No solution')
        return None, None
    if _debug:
        print('AdaptiveAstarCompute: time to A* search maze was %s seconds' %(endTime-startTime))
        print('AdaaptiveAstarCompute: %s nodes visited' %visitedNodes)
        print('AdaptiveAstarCompute: %s nodes expanded' %np.count_nonzero(expanded))
    return current,closedList

def repeatedForwardAstar(maze, start, goal,useAdaptive = False):
    startTime = time.time()
    current = start
    X,Y = maze.shape
    followLength = 0 #total travelled path length
    g_n = 0 # last computed shortest path distance to goal
    closedList = [] #for adaptive A* only
    heur = getHeuristics(maze,goal)
    seenMap = np.zeros(maze.shape, dtype=bool) #map with only seen walls
    followedMap = [] #used to keep track of travelled path for visualization
    path = [] #current path being executed by latest A* call
    if not useAdaptive:
        nodes = AstarCompute(maze=seenMap,start=current,goal=goal,heuristics=heur,_debug=False) #initial path on empty (no wall) map
    else:
        nodes, closedList = AdaptiveAstarCompute(maze=seenMap,start=current,goal=goal,heuristics=heur,_debug=False)
    while nodes is not None: #get array of positions to travel from A* linked list
        path.append(nodes.position)
        nodes=nodes.parent

    while current != goal:
        
        A,B = current
        ''' update seen map with only adjacent walls that are visible '''
        for x in [(1,0),(-1,0),(0,1),(0,-1)]:
            if A+x[0] in range(X) and B+x[1] in range(Y) and maze[A+x[0],B+x[1]]==1:
                seenMap[A+x[0],B+x[1]] = 1
        
        ''' if new walls seen on current path, then compute A* again'''
        if seenMap[path[-1]] == 1:
            path.clear()
            if not useAdaptive:
                nodes = AstarCompute(maze=seenMap,start=current,goal=goal,heuristics=heur,_debug=False)
            else:
                for x in  closedList:
                    heur[x.position] = g_n - x.g #update heuristics for adaptive A*
                nodes, closedList = AdaptiveAstarCompute(maze=seenMap,start=current,goal=goal,heuristics=heur,_debug=False)
                g_n = 0
            if nodes is None:
                return None
            while nodes is not None:#get list of path to follow from linked list
                path.append(nodes.position)
                g_n = g_n + 1
                nodes=nodes.parent
            
        current = path.pop()
        followedMap.append(current)
        followLength = followLength + 1
    endTime = time.time()
    print('repeatedForwardAstar: time to A* search maze was %s seconds' %(endTime-startTime))
    print('repeatedForwardAstar: %s nodes travelled' %followLength)
    return followedMap

def repeatedBackwardAstar(maze, start, goal):
    startTime = time.time()
    current = start
    X,Y = maze.shape
    heur = getHeuristics(maze,current)
    followLength = 0
    seenMap = np.zeros(maze.shape, dtype=bool) #map with only seen walls
    followedMap = [] #used to keep track of travelled path
    path = [] #current path being executed by latest A* call
    nodes = AstarCompute(maze=seenMap,start=goal,goal=current,heuristics=heur,_debug=False) #initial path on empty (no wall) map
    while nodes is not None: #get array of positions to travel from A* linked list
        path.append(nodes.position)
        nodes=nodes.parent
    
    while current != goal:
        A,B = current
        ''' update seen map with only adjacent walls that are visible '''
        for x in [(1,0),(-1,0),(0,1),(0,-1)]:
            if A+x[0] in range(X) and B+x[1] in range(Y) and maze[A+x[0],B+x[1]]==1:
                seenMap[A+x[0],B+x[1]] = 1
        ''' if new walls seen on current path, then compute A* again'''
        if seenMap[path[0]] == 1:
            path.clear()
            heur = getHeuristics(maze,current)
            nodes = AstarCompute(maze=seenMap,start=goal,goal=current,heuristics=heur,_debug=False)
            if nodes is None:
                return None
            while nodes is not None:#get list of path to follow from linked list
                path.append(nodes.position)
                nodes=nodes.parent
        current = path.pop(0)
        followedMap.append(current)
        followLength = followLength + 1
    endTime = time.time()
    print('repeatedBackwardAstar: time to A* search maze was %s seconds' %(endTime-startTime))
    print('repeatedBackwardAstar: %s nodes travelled' %followLength)
    return followedMap
    

def performTests():
    runtimes = []

    for x in range(50):
        print('performTests: %s' %x)
        maze=loadmaze(num=x)
        A,B = maze.shape
        goal=(100,100)
        start=(0,0)
        heur = getHeuristics(maze,goal)
        seenmap = np.zeros(maze.shape, dtype=bool)
        startTime = time.time()
        seenmap = repeatedForwardAstar(maze,start,goal,useAdaptive=False)
        endTime = time.time()-startTime
        if seenmap is None:
            print('performTests: No solution')
            runtimes = np.append(arr = runtimes, values = -1)
        else:
            print('performTests: Took %s seconds' %(endTime))
            runtimes = np.append(arr = runtimes, values = endTime)
        np.savetxt(fname = 'generated_mazes/performTests.txt', X = runtimes)       

      

if __name__ == "__main__":
    #savemazes(num=50) #uncomment this and run file to generate and save 50 mazes


'''
maze=loadmaze(num=28)  
A,B = maze.shape
goal=(100,100)
start=(0,0)
heur = getHeuristics(maze,goal)
seenmap = np.zeros(maze.shape, dtype=bool)
total = 0
#startTime = time.time()
seenmap = repeatedForwardAstar(maze,start,goal,useAdaptive=False)
#seenmap = repeatedBackwardAstar(maze,start,goal)
plt.figure()
plt.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
plt.imshow(seenmap,alpha=0.5)
plt.xticks([]), plt.yticks([])
plt.show()

plt.ylim(0,60)
plt.xticks(np.arange(0, 50, step=1))
plt.yticks(np.arange(0, 60, step=1))
plt.title('Adaptive Forward A* vs Forward A*')
plt.xlabel('Maze')
plt.ylabel('Seconds')

maze=np.loadtxt('generated_mazes/ForwardAstar_greaterG.txt')
plt.plot(maze,"-b", label="Forward A*")
maze=np.loadtxt('generated_mazes/AdaptiveForwardAstart.txt')
plt.plot(maze,"-r", label="Adaptive A*")
plt.legend()
plt.grid()
plt.show()
'''
#performTests()

