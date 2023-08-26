from functools import partial
import gui as wid
import pygame


class propertyBar():
    def __init__(self, x, y, layout='b') -> None:

        self.window = None
        self.layout = layout

        self.x = x
        self.y = y
        self.gap = 30
        self.padding = 10
        self.size = 15

        self.toggler = None
        self.backRect = None

        self.props = {}
        self.props_gui = {}

        # colors
        self.bg = (0, 0, 0)
        self.labelcolor = (200, 200, 200)
        self.labelbg = (0, 0, 0)
        self.valueColor = (0, 200, 0)
        self.valuebg = (0, 0, 0)

        # initilize the rect
        self.refershRender()

        pass

    def selectedGate(self, gateObj, attributes):
        self.props = {attr: getattr(gateObj, attr) for attr in attributes}
        self.doubleRefresh()
        pass

    def loadWindow(self, window):
        self.window = window
        self.backRect.loadWindow(window)

        for gui in self.props_gui.values():
            gui[0].loadWindow(window)
            gui[1].loadWindow(window)

    def renderEvent(self, event):
        pass

    def renderWindow(self):
        self.backRect.renderWidget()

        for gui in self.props_gui.values():
            gui[0].renderWidget()
            gui[1].renderWidget()
        pass

    def doubleRefresh(self):
        self.refershRender()
        self.refershRender()

    def refershRender(self):

        max_width = 0
        for i, (prop, value) in enumerate(self.props.items()):
            label_gui = wid.TextMesh(f"{prop} :>", self.x + self.padding, self.y + self.gap * i + self.padding,
                                     self.labelbg, self.labelcolor, self.size)
            value_gui = wid.TextMesh(f" {value}", self.x + label_gui.TextRect.width + self.padding,
                                     self.y + self.gap * i + self.padding, self.valuebg, self.valueColor, self.size)

            if self.window:
                label_gui.loadWindow(self.window)
                value_gui.loadWindow(self.window)

            self.props_gui[prop] = [label_gui, value_gui]
            max_width = max(max_width, label_gui.TextRect.width + value_gui.TextRect.width + self.padding)
        pass

        if self.window:
            if self.layout == 'b':
                self.x = self.window.get_width() - max_width
                self.y = self.window.get_height() - (len(self.props_gui) * (self.gap) + self.padding)

        # make backrgrounf
        self.backRect = wid.textureRect(self.x, self.y, max_width, len(self.props_gui) * (self.gap) + self.padding,
                                        self.bg)
        if self.window: self.backRect.loadWindow(self.window)
