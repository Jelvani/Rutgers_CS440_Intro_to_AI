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
    def __lt__(self, otherNode):
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

def createRandMaze(width=101,height=101,prob=0.1,save=False,imgfname=None,txtfname=None): #prob is probability of a wall appearing
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
        createBtrackingMaze(width=200,height=200,save=True,imgfname=('generated_mazes/images/'+str(fname_template)+str(i)+'.png'),txtfname=('generated_mazes/maze_files/'+str(fname_template)+str(i)+'.txt'))
    for i in range(int(num/2)):
        createRandMaze(width=200,height=200,save=True,imgfname=('generated_mazes/images/'+str(fname_template)+str(i+int(num/2))+'.png'),txtfname=('generated_mazes/maze_files/'+str(fname_template)+str(i+int(num/2))+'.txt'))

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

def AstarCompute(map,start,goal,heuristics):
    startTime = time.time()
    X,Y = map.shape
    closedList = []
    openList = []
    expanded = np.zeros(map.shape, dtype=bool) #expanded matrix
    startNode = Node(position=start,g = 0,h = heuristics[start])
    goalNode = Node(position=goal,g = 0,h = 0)
    heapq.heappush(openList,startNode)
    while(openList):

        '''get node in openlist with lowest f value'''
        index = 0
        current=heapq.heappop(openList)
        closedList.append(current)#add to closed once once chosen for expansion
        expanded[current.position]=True
        A,B = current.position
        '''right after appending to closed list, check if appended node is the goal and return if it is'''
        if current == goalNode:
            break

        for x in [(1,0),(-1,0),(0,1),(0,-1)]:
            if A+x[0] in range(X) and B+x[1] in range(Y) and map[A+x[0],B+x[1]]==0 and expanded[A+x[0],B+x[1]]==False:#if in range and not a wall, add to temporary list
                newNode = Node(parent=current, position=(A+x[0],B+x[1]), g=current.g+1, h=heuristics[(A+x[0],B+x[1])]) 
                openList.append(newNode)

    print('AstarCompute: time to A* search maze was %s seconds' %(time.time()-startTime))
    return current
    

#savemazes(10)

maze=loadmaze(num=8)
A,B = maze.shape
goal=(180,179)
start=(0,0)
heur = getHeuristics(maze,goal)
nodes = AstarCompute(map=maze,start=start,goal=goal,heuristics=heur)
path = np.zeros(maze.shape, dtype=float)
color = 200
while nodes is not None:
    color=color+1
    path[nodes.position]=color
    nodes=nodes.parent
path[start]=200
path[goal]=200
plt.imshow(maze,cmap=plt.cm.binary)
plt.imshow(path,alpha=0.5)
plt.xticks([]), plt.yticks([])
plt.show()
