import Astar
import pygame
import numpy as np

'''Choose maze number to load'''
_MAZE_NUMBER =42
_START = (0,0)
_GOAL = (100,100)
screen = (700,700) #screen size
pygame.init()
display = pygame.display.set_mode(screen)
running = True
maze=Astar.loadmaze(num=_MAZE_NUMBER)
followedMap = Astar.repeatedForwardAstar(maze,_START,_GOAL,useAdaptive=False)
heuristics = Astar.getHeuristics(maze,_GOAL)
optimalPath = Astar.AstarCompute(maze,_START,_GOAL,heuristics)
#followedMap = Astar.repeatedBackwardAstar(maze,(0,0),(100,100))
maze = (1 - maze)*255 #scale colors
while optimalPath:
    maze[optimalPath.position] = 20
    optimalPath = optimalPath.parent
colorIncrement = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if followedMap:
        colorIncrement = colorIncrement + 0.01
        x,y = followedMap.pop(0) # get next coordinate
        maze[x,y] = 120
    surf = pygame.surfarray.make_surface(maze) #convert array to surface
    maze[x,y] = 100
    surf = pygame.transform.scale(surf,screen) #scale for screen size
    display.blit(surf, (0, 0)) # draw maze
    pygame.display.update()
    pygame.time.delay(30)
pygame.quit()