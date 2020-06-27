import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time
import os

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

def createRandMaze(width=101,height=101,prob=0.3,save=False,imgfname=None,txtfname=None): #prob is probability of a wall appearing
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


savemazes(num=10)
'''
maze=loadmaze(num=1)
plt.figure()
plt.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
plt.xticks([]), plt.yticks([])
plt.show()
'''