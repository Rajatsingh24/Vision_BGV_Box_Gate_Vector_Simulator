import pygame
import pygwidgets
import vectors


class TextMesh():
    def __init__(self, text, x, y, bg, fg, size, font="fonts/QueensidesMedium.ttf") -> None:
        self.text = text
        self.x = x
        self.y = y
        self.bg = bg
        self.fg = fg
        self.size = size
        self.font = font
        self.window = None
        self.TextRect = None
        self.visible = True

        # here we are doing this inital assignment

        font = pygame.font.Font(self.font, self.size)
        text = font.render(self.text, True, self.fg, self.bg)
        rect = text.get_rect()
        self.TextRect = rect

        pass

    def setPos(self, x, y):
        self.x, self.y = x, y

    def setText(self, text):
        self.text = text

    def loadWindow(self, window):
        self.window = window
        pass

    def renderWidget(self):
        if self.visible:
            font = pygame.font.Font(self.font, self.size)
            text = font.render(self.text, True, self.fg, self.bg)
            rect = text.get_rect()
            self.TextRect = rect
            rect.center = (self.x + rect.width / 2, self.y + rect.height / 2)
            self.window.blit(text, rect)
        pass


# a little modifies class

class TextMeshPro(TextMesh):
    def __init__(self, text, x, y, bg, fg, size, font="freesansbold.ttf", rectdim=(10, 10)) -> None:
        super().__init__(text, x, y, bg, fg, size, font)
        self.rectdim = rectdim

    def loadWindow(self, window):
        return super().loadWindow(window)

    def renderWidget(self):
        font = pygame.font.Font(self.font, self.size)
        text = font.render(self.text, True, self.fg, self.bg)
        rect = text.get_rect()
        self.TextRect = rect
        rect.center = (self.x, self.y)
        rect.width, rect.height = self.rectdim
        self.window.blit(text, rect)


# here we have a button class inheritesfrom textmesh class

class Button(TextMesh):
    def __init__(self, text, x, y, bg, fg, size, trigger, font="fonts/QueensidesMedium.ttf") -> None:
        super().__init__(text, x, y, bg, fg, size, font)
        # here we have exta
        self.mouseevent = "out"
        self.triggerFunctions = [lambda: 0, lambda: 0, lambda: 0, lambda: 0]
        self.triggerFunction = trigger

        self.onenterbg = (255, 255, 255)
        self.onleavebg = (0, 0, 0)
        self.onenterfg = (255, 0, 0)
        self.onleavefg = (255, 255, 255)
        self.ontriggerbg = (255, 255, 255)
        self.ontriggerfg = (0, 0, 0)

        self.active = True

        # retention proopeties // deplicates
        # //
        self.retainBg = self.bg
        self.retainFg = self.fg

    def loadWindow(self, window):
        return super().loadWindow(window)

    def renderWidget(self):
        return super().renderWidget()

    def eventRender(self, event):

        if event == None or not self.active:
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.TextRect.collidepoint(event.pos):
                self.triggerFunction()

                self.bg = self.ontriggerbg
                self.fg = self.ontriggerfg

                pass

        if event.type == pygame.MOUSEMOTION:
            if self.mouseevent == "out":
                if self.TextRect.collidepoint(event.pos):
                    # on mouse enter

                    self.triggerFunctions[0]()

                    # setitng the buttons proerties
                    self.bg = self.onenterbg
                    self.fg = self.onenterfg

                    self.mouseevent = "in"
                    pass

            if self.mouseevent == "in":
                if not self.TextRect.collidepoint(event.pos):
                    # on mouse leave
                    self.triggerFunctions[1]()

                    self.bg = self.onleavebg
                    self.fg = self.onleavefg

                    self.mouseevent = "out"
                    pass

            if self.mouseevent == "in":

                if self.TextRect.collidepoint(event.pos):
                    self.triggerFunctions[2]()

                    # mouse hover
                    self.mouseevent = "in"
                    pass

            if event.type == pygame.MOUSEMOTION:

                if self.mouseevent == "out":
                    if not self.TextRect.collidepoint(event.pos):
                        # on mouse enter

                        # setitng the buttons proerties
                        self.bg = self.retainBg
                        self.fg = self.retainFg

                        self.mouseevent = "out"
                        pass

        pass


class textInput():
    def __init__(self, x, y, width, fontsize=30, textcolor=(0, 0, 0), bg=(255, 255, 255), placeholder="",
                 triggger=lambda: 0) -> None:
        self.window = None
        self.x = x
        self.y = y
        self.width = width
        self.fontsize = fontsize
        self.textColor = textcolor
        self.bg = bg
        self.placeholder = placeholder
        self.InputField = None
        self.value = None
        self.trigger = triggger
        self.visible = True
        self.active = True
        pass

    def loadWindow(self, window):

        self.window = window
        self.InputField = pygwidgets.InputText(self.window, (self.x, self.y), self.placeholder, None, self.fontsize,
                                               self.width, self.textColor, self.bg)

        pass

    def renderWidget(self):
        if self.visible:
            self.InputField.draw()

        pass

    def eventHandler(self, event):
        if self.active:
            if self.InputField.handleEvent(event):
                self.value = self.InputField.getValue()
                self.trigger()
        pass


class textureRect():
    def __init__(self, x, y, width, height, color=(200, 200, 200), borders=[-1, -1, -1, -1]) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.borders = borders
        self.visible = True

        self.window = None

        self.innerRect = pygame.Rect(self.x, self.y, self.width, self.height)
        pass

    def loadWindow(self, window):
        self.window = window
        pass

    def renderWidget(self):
        if self.visible:
            self.innerRect = pygame.Rect(self.x, self.y, self.width, self.height)
            pygame.draw.rect(self.window, self.color, self.innerRect, border_top_left_radius=self.borders[0],
                             border_top_right_radius=self.borders[1], border_bottom_left_radius=self.borders[2],
                             border_bottom_right_radius=self.borders[3])

        pass


class Navbar():
    def __init__(self, x, y, rate=1, limits=100, width=100, outVisible=False) -> None:
        self.x = x
        self.y = y
        self.movY = y
        self.limits = limits
        self.width = width
        self.scollerRate = rate
        self.motion = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.limits)

        pass

    def renderEvent(self, event):
        # scolling the in sidethe boundry

        if self.rect.collidepoint(pygame.mouse.get_pos()):
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.motion += self.scollerRate
                    self.movY += self.scollerRate

                else:
                    self.motion -= self.scollerRate
                    self.movY -= self.scollerRate

    def renderTheWidgetByMotion(self, widget):
        # here we need tolimit this motion
        widget.y += self.motion

        pass

    def renderTheWidgetByPOs(self, widget):
        widget.y = self.movY
        pass

    def setDefaultPos(self):
        # making the things default
        self.movY = self.y

    def setDefaultMotion(self):
        self.motion = 0

    def outboundRenderHandle(self, widget):
        if widget.y < self.y or widget.y > self.y + self.limits:
            # cant reder the widget
            pass
        else:
            # render the widget
            widget.renderWidget()
            pass


class interactiveShape():
    LAYER_ID = 0
    EVENT_COUNT = 0

    def __init__(self, x, y, shape, attrib=None, triggerFunc=None, toggle=False, normal_col=(0, 0, 0)) -> None:
        self.x = x
        self.y = y
        self.shape = shape
        self.attrib = attrib
        self.triggerFunc = triggerFunc
        self.window = None
        self.toggle = toggle

        self.vectors = [[1, 1, 1]
            , [1, -1, 1],
                        [-1, -1, 1],
                        [-1, 1, 1]]
        self.transform = None

        self.colors = {'click': (200, 0, 0), 'enter': (200, 200, 200), 'leave': (200, 200, 200), 'normal': (0, 0, 0)}
        self.color = self.colors['normal']
        # internal state
        self.triggerRect = None
        self.mouseevent = 'out'

        interactiveShape.LAYER_ID += 1

        pass

    def apply_transform(self, posvec, trs):
        return [sum([v1 * v2 for v1, v2 in zip(posvec, vec)]) for vec in trs]

    def setTransform(self, trans):
        self.transform = trans

    def loadWindow(self, window):
        self.window = window
        self.renderWindow()
        pass

    def renderEvent(self, event):

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            interactiveShape.EVENT_COUNT += 1

            if self.triggerRect.collidepoint(event.pos):
                self.triggerFunc()
                self.color = self.colors['click']

        if event.type == pygame.MOUSEMOTION:
            if self.mouseevent == "out":
                if self.triggerRect.collidepoint(event.pos):
                    # on mouse enter

                    # setitng the buttons proerties
                    self.color = self.colors['enter']
                    self.mouseevent = "in"
                    pass

            if self.mouseevent == "in":
                if not self.triggerRect.collidepoint(event.pos):
                    # on mouse leave

                    self.color = self.colors['leave']
                    self.mouseevent = "out"
                    pass

            if self.mouseevent == "in":
                if self.triggerRect.collidepoint(event.pos):
                    # mouse hover

                    self.mouseevent = "in"
                    pass

            if event.type == pygame.MOUSEMOTION:

                if self.mouseevent == "out":
                    if not self.triggerRect.collidepoint(event.pos):
                        # on mouse enter
                        self.color = self.colors['normal']
                        # setitng the buttons proerties
                        self.mouseevent = "out"
                        pass

        pass

    def renderWindow(self):
        if self.shape == "circle":
            self.triggerRect = pygame.draw.circle(self.window, self.color, (self.x, self.y), self.attrib[0])
            pass
        if self.shape == "rect":
            self.triggerRect = pygame.draw.rect(self.window, self.color,
                                                pygame.Rect(self.x, self.y, self.attrib[0], self.attrib[1]))
            pass

        if self.shape == "rectB":
            self.triggerRect = pygame.draw.rect(self.window, self.color,
                                                pygame.Rect(self.x, self.y, self.attrib[0], self.attrib[1]),
                                                self.attrib[2])
            pass

        if self.shape == "rectBB":
            color1 = (200, 200, 0)
            color2 = (0, 0, 0)
            if (self.color == color1):
                self.color = color2
            else:
                self.color = color2
            self.triggerRect = pygame.draw.rect(self.window, self.color,
                                                pygame.Rect(self.x, self.y, self.attrib[0], self.attrib[1]),
                                                self.attrib[2])
            pass

        if self.shape == "rectT":
            points = [self.apply_transform(vec, self.transform)[:-1] for vec in self.vectors]
            self.triggerRect = pygame.draw.polygon(self.window, self.color, points)

            pass

        if self.shape == "rectU":
            self.triggerRect = pygame.draw.rect(self.window, self.color,
                                                pygame.Rect(self.x, self.y, self.attrib[0], self.attrib[1]),
                                                self.attrib[2])
            pass
        if self.shape == "triangle":
            points = self.__triangle((self.x, self.y), self.attrib[0], self.attrib[1])
            self.triggerRect = pygame.draw.polygon(self.window, self.color, points)
            pass

        if self.shape == "line":
            self.triggerRect = pygame.draw.line(self.window, self.color, (self.x, self.y),
                                                (self.attrib[0], self.attrib[1]), self.attrib[2])

        if self.shape == "Zline":

            cord1, cord2 = self.attrib
            stepx = (cord2[0] - cord1[0]) / 2
            stepy = (cord2[1] - cord1[1]) / 2
            if (abs(stepx) >= abs(stepy)):

                points = [cord1,
                          [cord1[0] + stepx, cord1[1]],
                          [cord2[0] - stepx, cord2[1]],
                          cord2,
                          [cord2[0] - stepx, cord2[1]],
                          [cord1[0] + stepx, cord1[1]]]

            else:
                points = [cord1,
                          [cord1[0], cord1[1] + stepy]
                    , [cord2[0], cord2[1] - stepy]
                    , cord2
                    , [cord2[0], cord2[1] - stepy]
                    , [cord1[0], cord1[1] + stepy]]

            self.triggerRect = pygame.draw.polygon(self.window, self.color, points, 3)

        if self.shape == "poly":
            points = [vectors.vector2D(*v) + vectors.vector2D(self.x, self.y) for v in self.attrib]
            points = [v.toArray() for v in points]
            self.triggerRect = pygame.draw.polygon(self.window, self.color, points)

        # here can add mosre shape

        pass

    pass

    def __triangle(self, origin, direction, mag):
        originVec = vectors.vector2D().fromArray(origin)
        vec = vectors.vector2D().fromArray(direction).normalized()
        vec2 = vec.rotateBy(120)
        vec3 = vec2.rotateBy(120)

        polypoints = [(vec * mag), vec2 * mag, vec3 * mag, vec * mag]

        return [(v + originVec).toArray() for v in polypoints]
        pass

    def setColor(self, to, color, baseColor=None):
        self.colors[to] = color
        if baseColor:
            self.color = baseColor
        pass

    def currentColor(self, col):
        self.color = col

    def getPos(self):
        return (self.x, self.y)


class ImageButton():
    def __init__(self, imagePath, x, y, BoxWidth, triggerFunc=lambda: 0) -> None:
        self.x = x
        self.y = y
        self.imagePath = imagePath
        self.BoxWidth = BoxWidth
        self.window = None
        self.baseRect = None
        self.image = None
        self.triggerFunc = triggerFunc

        self.refresh()
        pass

    def set_image(self, path):
        self.imagePath = path
        self.refresh()

    def setPos(self, pos):
        self.x, self.y = pos
        self.refresh()

    def loadWindow(self, window):
        self.window = window

    def eventRender(self, event):
        self.image.set_alpha(255)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.baseRect.collidepoint(event.pos):
                self.triggerFunc()
        if event.type == pygame.MOUSEMOTION:
            if self.baseRect.collidepoint(event.pos):
                self.image.set_alpha(150)
                pass

    def renderWidget(self):
        if self.image:
            self.window.blit(self.image, [self.x, self.y])
        pass

    def refresh(self):

        self.image = pygame.image.load(self.imagePath)
        w, h = self.image.get_size()
        ratio = h / w
        self.image = pygame.transform.scale(self.image, [self.BoxWidth, self.BoxWidth * ratio])
        self.baseRect = pygame.Rect(self.x, self.y, self.image.get_width(), self.image.get_height())

        pass


class InputFieldPro():
    def __init__(self, x, y, width, height, placeholder, size=15, fill=False, color=(0, 0, 0), borderColor=(0, 0, 0),
                 fillColor=(100, 100, 100), trigger=lambda x: 0, trigger2=lambda x: 0) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.window = None
        self.Text = ""
        self.placeHolder = placeholder
        self.baseRectBorder = None
        self.baseRectFill = None
        self.fill = fill
        self.borderColor = borderColor
        self.fillColor = fillColor
        self.color = color
        self.size = size
        self.trigger = trigger
        self.trigger2 = trigger2

        self.font = "fonts/QueensidesMedium.ttf"
        self.lineGap = 25
        self.activeColor = (200, 0, 0)
        self.isactive = False
        self.line = 0

        self.refresh()
        pass

    def setPos(self, x, y):
        self.x, self.y = x, y
        self.refresh()
        pass

    def setAtt(self, linegap=None, size=None):
        if linegap: self.lineGap = linegap
        if size: self.size = size

    def loadWindow(self, window):
        self.window = window

    def eventRender(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.baseRectBorder.collidepoint(*pygame.mouse.get_pos()):
                self.isactive = True
                print("true")
            else:
                self.isactive = False

        if self.isactive and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.Text = self.Text[:-1]
                self.trigger2(self.Text)
                return

            if event.key == pygame.K_RETURN:
                self.trigger(self.Text)
                self.Text = ""

            else:
                if self.lines(self.Text + "X", self.width) <= int(self.height / 25):
                    self.Text += event.unicode
                    self.trigger2(self.Text)

        pass

    def renderWindow(self):

        pygame.draw.rect(self.window, self.fillColor, self.baseRectFill)
        pygame.draw.rect(self.window, (self.activeColor if (self.isactive) else self.borderColor), self.baseRectBorder,
                         2)

        if not self.isactive and not self.Text:
            font = pygame.font.Font(self.font, self.size)
            text = font.render(self.placeHolder, True, self.color)
            self.window.blit(text, (self.x, self.y))

        # self.window.blit(text , self.baseRectBorder )
        self.line = 0
        charLimit = int(self.width / 8)
        for index in range(0, len(self.Text)):
            if (index) % charLimit == 0:
                text = TextMesh(self.Text[charLimit * self.line:charLimit * (self.line + 1)], self.x,
                                self.y + self.lineGap * self.line, None, self.color, self.size)
                text.loadWindow(self.window)
                text.renderWidget()
                self.line += 1

        pass

    def lines(self, text, width):
        line = 0
        charLimit = int(width / 8)
        for index in range(0, len(text)):
            if (index) % charLimit == 0:
                line += 1
        return line

    def refresh(self):
        self.baseRectBorder = pygame.Rect(self.x, self.y, self.width, self.height)
        self.baseRectFill = pygame.Rect(self.x, self.y, self.width, self.height)
        pass


class LabelList():

    def __init__(self, x, y, width, height, bg=(0, 0, 0), deafultcolor=(200, 200, 200), defSize=20) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.internlBias = 10
        self.height = height
        self.bg = bg
        self.defaultColor = deafultcolor
        self.defaultSize = defSize

        self.lineGap = 25
        self.queue = []

        self.window = None
        self.font = "fonts/QueensidesMedium.ttf"

        pass

    def setPos(self, x, y):
        self.x, self.y = x, y

    def addLabel(self, label, color=None, size=None):
        if color == None: color = self.defaultColor
        if size == None: size = self.defaultSize

        self.queue.append([label, color, size])
        pass

    def removeLabel(self, index=0):

        if self.queue:
            index = min(max(0, index), len(self.queue))
            return self.queue.pop(index)

    def loadWindow(self, window):
        self.window = window

    def eventRender(self, event):

        pass

    def renderWindow(self):

        pygame.draw.rect(self.window, self.bg, pygame.Rect(self.x, self.y, self.width, self.height + self.internlBias))
        q = self.queue.copy()
        q.reverse()
        currentHeight = 0
        widthRatio = 25
        lines = 0
        for textItem in q:
            if lines < self.height / widthRatio:
                text, color, size = textItem
                lines += self.renderline(self.x, self.y + lines * widthRatio, text, self.width, color, size)

        # self.height = lines * widthRatio

        pass

    def renderline(self, x, y, text, width, color, size):
        line = 0
        lineHeight = 0
        charLimit = int(width / 12)
        for index in range(0, len(text)):

            if (index) % charLimit == 0:
                t = TextMesh(text[charLimit * line:charLimit * (line + 1)], x, y + self.lineGap * line, None, color,
                             size)
                lineHeight += t.TextRect.height
                t.loadWindow(self.window)
                t.renderWidget()
                line += 1

        return line

    # def renderlinePro(self , x , y ,  text , width  , color  , size):
    #     line = 0
    #     lineHeight = 0
    #     lastIndex = -1
    #     for index in range(0 , len(text)):
    #         font = pygame.font.Font(self.font , size)
    #         txt = font.render(text[0:index+1] , True , (0,0,0))
    #         if  txt.get_size()[0] % width==0 :

    #             t = TextMesh(text[lastIndex+1 : index] , x , y  + lineHeight , None , color  ,size )
    #             lastIndex = index
    #             lineHeight +=t.TextRect.height
    #             t.loadWindow(self.window)
    #             t.renderWidget()
    #             line +=1

    #     t = TextMesh(text[lastIndex+1 :] , x , y  + lineHeight , None , color  ,size )
    #     lineHeight +=t.TextRect.height
    #     t.loadWindow(self.window)
    #     t.renderWidget()
    #     line +=1

    #     return lineHeight


class LoadingGui():
    # 100,100,arcSize=90,size=100 , incr=50 , borderW=5
    def __init__(self, x, y, color=(0, 20, 200), arcSize=90, size=100, borderW=5, incr=30, clk=0.1):
        from util_functions import Timer
        import math
        self.__math = math

        self.x = x
        self.y = y
        self.loadingRect = None
        self.size = size
        self.start = 0
        self.stop = arcSize
        self.incr = incr

        self.color = color
        self.borderW = borderW
        self.window = None

        self.delta = 0

        self.is_loading = False

        self.timer = Timer(clk, self.__process, True)
        pass

    def setPos(self, pos):
        self.x, self.y = pos

    def loadWindow(self, window):
        self.window = window
        pass

    def __process(self):
        self.delta += self.incr
        pass

    def startLoading(self):
        self.timer.start()
        self.is_loading = True
        pass

    def stopLoading(self):
        self.timer.pause()
        self.is_loading = False
        pass

    def againStartLoading(self):
        self.timer.startAgain()
        self.is_loading = True

    def renderWidget(self):
        if self.is_loading:
            self.loadingRect = pygame.draw.arc(self.window, self.color,
                                               pygame.Rect(self.x - self.size / 2, self.y - self.size / 2, self.size,
                                                           self.size), self.__math.radians(self.start + self.delta),
                                               self.__math.radians(self.stop + self.delta), self.borderW)

        pass
