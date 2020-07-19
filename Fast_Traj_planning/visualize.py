import Astar
import pygame
import numpy as np
import pygame_gui
import random as rand
###################################################
'''Choose maze number to load'''
_MAZE_NUMBER =27
_START = (0,0)
_GOAL = (100,100)
###################################################
screen = (900,700) #screen size
pygame.init()
display = pygame.display.set_mode(screen)
running = True
maze=Astar.loadmaze(num=_MAZE_NUMBER)
followedMap = []

heuristics = Astar.getHeuristics(maze,_GOAL)
optimalPath = Astar.AstarCompute(maze,_START,_GOAL,heuristics)

clock = pygame.time.Clock()
manager = pygame_gui.UIManager(screen)

maze = (1 - maze)*255 #scale colors
'''UI Buttons'''
runForwardAstar = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((700, 0), (200, 100)),text='Forward A*',manager=manager)
runBackwardAstar = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((700, 100), (200, 100)),text='Backward A*',manager=manager)
runAdaptiveAstar = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((700, 200), (200, 100)),text='Adaptive A*',manager=manager)
changeMaze = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((700, 300), (200, 100)),text='Change Maze',manager=manager)

optpathlength = 0
while optimalPath:
    maze[optimalPath.position] = 20
    optimalPath = optimalPath.parent
    optpathlength = optpathlength + 1

while running:
    optpathlength = 0
    time_delta = clock.tick(60)/1000.0
    for event in pygame.event.get():
        ''' for closing window '''
        if event.type == pygame.QUIT:
            running = False
    
        ''' for pygame gui elements '''
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == runForwardAstar:
                    maze=Astar.loadmaze(num=_MAZE_NUMBER)
                    heuristics = Astar.getHeuristics(maze,_GOAL)
                    optimalPath = Astar.AstarCompute(maze,_START,_GOAL,heuristics)
                    followedMap = Astar.repeatedForwardAstar(maze,_START,_GOAL,useAdaptive=False)
                    maze = (1 - maze)*255 #scale colors
                    while optimalPath:
                        maze[optimalPath.position] = 20
                        optimalPath = optimalPath.parent
                        optpathlength = optpathlength + 1
                    print("optimalPathLength: %s" %optpathlength)
                if event.ui_element == runBackwardAstar:
                    maze=Astar.loadmaze(num=_MAZE_NUMBER)
                    heuristics = Astar.getHeuristics(maze,_START)
                    optimalPath = Astar.AstarCompute(maze,_GOAL,_START,heuristics)
                    followedMap = Astar.repeatedBackwardAstar(maze,_START,_GOAL)
                    maze = (1 - maze)*255 #scale colors
                    while optimalPath:
                        maze[optimalPath.position] = 20
                        optimalPath = optimalPath.parent
                        optpathlength = optpathlength + 1
                    print("optimalPathLength: %s" %optpathlength)
                if event.ui_element == runAdaptiveAstar:
                    maze=Astar.loadmaze(num=_MAZE_NUMBER)
                    heuristics = Astar.getHeuristics(maze,_GOAL)
                    optimalPath = Astar.AstarCompute(maze,_START,_GOAL,heuristics)
                    followedMap = Astar.repeatedForwardAstar(maze,_START,_GOAL,useAdaptive=True)
                    maze = (1 - maze)*255 #scale colors
                    while optimalPath:
                        maze[optimalPath.position] = 20
                        optimalPath = optimalPath.parent
                        optpathlength = optpathlength + 1
                    print("optimalPathLength: %s" %optpathlength)
                if event.ui_element == changeMaze:
                    if _MAZE_NUMBER >24:
                        _MAZE_NUMBER = rand.randint(0,21)
                    else:
                        _MAZE_NUMBER = rand.randint(25,49)
        manager.process_events(event)
    manager.update(time_delta)
    
    if followedMap:
        x,y = followedMap.pop(0) # get next coordinate
        maze[x,y] = 120
    surf = pygame.surfarray.make_surface(maze) #convert array to surface
    if followedMap:
        maze[x,y] = 100
    surf = pygame.transform.scale(surf,(700,700)) #scale for screen size
    display.blit(surf, (0, 0)) # draw maze

    manager.draw_ui(display)
    pygame.display.update()
    #pygame.time.delay(3)
pygame.quit()