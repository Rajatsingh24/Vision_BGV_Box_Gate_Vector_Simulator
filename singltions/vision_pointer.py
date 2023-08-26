from gui import TextMesh, interactiveShape
from util_functions import Timer
import pygame


class vision_pointer():
    def __init__(self, width=20, height=20) -> None:
        self.window = None
        self.x = 0
        self.y = 0
        self.width = width
        self.height = height
        self.pointerGui = None
        self.color = (200, 200, 0)
        self.borderW = 1
        self.timer = 0.5

        self.refresh()
        pass

    def follow(self, pos):
        self.x, self.y = pos
        self.refresh()

    def loadWindow(self, window):
        self.window = window

        pass

    def idleState(self):
        self.color = (200, 200, 0)
        self.borderW = 1
        pass

    def eventRenderer(self, event):

        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
            self.color = (200, 0, 0)
            self.borderW = 3
            Timer(self.timer, self.idleState).start()

            pass

        if (event.type == pygame.KEYDOWN and event.key == pygame.K_RSHIFT):
            self.color = (0, 200, 0)
            self.borderW = 3
            Timer(self.timer, self.idleState).start()

            pass

        if (event.type == pygame.KEYDOWN and event.key == pygame.K_RCTRL):
            self.color = (0, 0, 200)
            self.borderW = 3
            Timer(self.timer, self.idleState).start()
            pass

        if (event.type == pygame.KEYDOWN and event.key == pygame.K_RALT):
            self.color = (0, 200, 200)
            self.borderW = 3
            Timer(self.timer, self.idleState).start()
            pass

    def renderWidget(self):
        pygame.draw.rect(self.window, self.color, self.pointerGui, self.borderW)

        pass

    def refresh(self):
        self.pointerGui = pygame.Rect(self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)
