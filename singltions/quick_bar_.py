from functools import partial
import gui as wid
import pygame


class quickBar():
    def __init__(self, pos, size, buttons=5,
                 buttonLabels=["(+) ADD GATE", "(-) DELETE GATE", "(~) DISCONNECT", "(||) DUPLICATE", "CLEAR"],
                 trigger=lambda x: x):
        self.pos = pos
        self.size = size
        self.buttons = buttons
        self.window = None
        self.buttonLabels = buttonLabels
        self.trigger = trigger
        # setting
        self.modeBit = [1, 1]

        # internal state

        self.buttonArray = []
        self.backgroundRect = None
        self.gap = 30

        self.show = False
        self.keyBuffer = False

        self.editor = None

        # internal state

        self.prefabs = {}

        # command inputs
        self.inputcommand = None
        self.inputButton = None
        self.commandPromptVisibility = False

        pass

    def loadWinow(self, window):
        self.window = window
        pass

    def renderEvent(self, event):

        keypressed = pygame.key.get_pressed()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and keypressed[pygame.K_LSHIFT] and self.modeBit[
            0]:
            self.pos = pygame.mouse.get_pos()
            self.makeButtons()
            self.show = True

        # other way around
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RCTRL and self.modeBit[0]:
            self.pos = pygame.mouse.get_pos()
            self.makeButtons()
            self.show = True

        if self.show:
            for button in self.buttonArray:
                button.eventRender(event)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.modeBit[1]:
            self.show = False

        pass

        # if self.commandPromptVisibility:
        #     if self.inputcommand and self.inputButton:
        #         self.inputButton.eventRender(event)
        #         self.inputcommand.eventHandler(event)

    # util for mode bit 0 ////
    def showWindow(self):
        self.makeButtons()
        self.show = True
        pass

    def hideWindow(self):
        self.show = False
        pass

    def setPos(self, pos):
        self.pos = pos
        pass

    def renderWindow(self):
        if self.show:
            self.backgroundRect.renderWidget()

            for button in self.buttonArray:
                button.renderWidget()

        # if self.commandPromptVisibility:
        #     if self.inputcommand and self.inputButton:
        #         self.inputButton.renderWidget()
        #         self.inputcommand.renderWidget()

        pass

    def __trigger(self, index):
        # here ar ethe buttin events
        self.trigger(index)
        # button pressed common event
        self.show = False

    def makeButtons(self):
        # display

        # craeting all the buttons
        self.buttonArray = []
        rectwidth = 0
        rectheight = 0
        for i, buttonlabel in enumerate(self.buttonLabels):
            b = wid.Button(buttonlabel, self.pos[0], self.pos[1] + self.gap * i, (0, 0, 0), (200, 200, 200), self.size,
                           partial(self.__trigger, i))
            # some calucations
            rectwidth = b.TextRect.width if b.TextRect.width > rectwidth else rectwidth
            rectheight += self.gap

            b.loadWindow(self.window)
            self.buttonArray.append(b)
        # set the background rect
        self.backgroundRect = wid.textureRect(self.pos[0], self.pos[1], rectwidth, rectheight, (0, 0, 0))
        self.backgroundRect.loadWindow(self.window)
        pass
