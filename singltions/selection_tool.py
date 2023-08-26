from functools import partial
import gui as wid
import pygame


class selectionTool():
    def __init__(self, trigger=lambda x: 0, trigger2=lambda x: 0) -> None:
        self.window = None
        self.startpos = None
        self.endpos = None
        self.selectionRect = None
        self.selectionColor = (200, 0, 0)
        self.trigger = trigger
        self.trigger2 = trigger2

        self.selectionSet = []

    def renderConsiderationSet(self, List, selectionAttributes=[], selection_map=[]):
        if self.endpos and self.startpos:
            self.selectionSet = []
            for index, item in enumerate(List):
                flag = 0
                for attr in selectionAttributes:
                    if flag: break
                    cord = getattr(item, attr)
                    if not self.selectionRect.collidepoint(cord):
                        flag = 1
                        continue
                if not flag:
                    if selection_map:
                        self.selectionSet.append(selection_map[index])
                    else:
                        self.selectionSet.append(item)

    def loadwindow(self, window):
        self.window = window

        pass

    def renderEvent(self, event):

        # if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
        #     self.trigger2(self.selectionSet)
        #     pass

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not pygame.key.get_pressed()[pygame.K_LSHIFT]:
            self.startpos = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.startpos = None
            self.endpos = None
            self.trigger(self.selectionSet)

        if pygame.mouse.get_pressed()[0]:
            self.endpos = pygame.mouse.get_pos()

    def renderWindow(self):
        if self.endpos and self.startpos:
            self.selectionRect = pygame.Rect(self.startpos[0], self.startpos[1], self.endpos[0] - self.startpos[0],
                                             self.endpos[1] - self.startpos[1])

            pygame.draw.rect(self.window, self.selectionColor, self.selectionRect, 1)

        pass
