class LinearPathForm():
    def __init__(self, start, end, smoothness, callback, loop=False, clk=2) -> None:
        self.start = start
        self.end = end
        self.smoothness = smoothness
        self.loop = loop
        self.callback = callback

        # internal states
        self.STATE = -1
        self.do_start = False

        self.clock = 0
        self.LIMIT = clk

        self.div = 10

        # fath formulation
        self.points = []

        self.makePath()

    def makePath(self):
        x0, y0 = self.start
        x1, y1 = self.end

        dx = x1 - x0
        dy = y1 - y0

        if dx:
            m = dy / dx
            i = dx / (self.div * self.smoothness)
            self.points = [[x0 + xx * i, y0 + m * (xx * i)] for xx in range(int(self.div * self.smoothness) + 1)]
        else:
            m = dx / dy
            i = dy / (self.div * self.smoothness)
            self.points = [[x0 + m * (xx * i), y0 + xx * i] for xx in range(int(self.div * self.smoothness) + 1)]

        self.indexer = 0

        pass

    def startPath(self):
        self.do_start = True
        pass

    def restartPath(self):
        self.do_start = True
        self.indexer = 0
        self.STATE = -1
        pass

    def resetPath(self):
        self.indexer = 0
        self.STATE = -1
        self.do_start = False
        self.callback(self.points[self.indexer])

    def path(self):
        if self.clock == 0:
            if self.STATE == -1:
                self.STATE = 0

            if self.STATE == 0:
                self.callback(self.points[self.indexer])
                self.indexer += 1
                if self.indexer == len(self.points):
                    self.STATE = 1

            if self.STATE == 1:
                if self.loop:
                    self.STATE = -1
                    self.indexer = 0
                pass
        pass

    def pathClock(self):
        if self.do_start:
            self.path()
            self.clock = (self.clock + 1) % self.LIMIT

        pass
