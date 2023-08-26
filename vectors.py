import math


class vector2D():
    def __init__(self, vx=0, vy=0) -> None:
        self.vx = vx
        self.vy = vy
        pass

    def normalized(self):
        z = complex(self.vx, self.vy)
        mod = math.sqrt((z * z.conjugate()).real)
        if mod == 0: return vector2D(z.real, z.imag)

        z = z / mod
        return vector2D(z.real, z.imag)

        pass

    def magnutude(self):
        return math.sqrt(self.dot(self))
        pass

    def dot(self, vector):
        return self.vx * vector.vx + self.vy * vector.vy
        pass

    def rotateBy(self, angle):
        conf = math.pi / 180
        vector = complex(math.cos(angle * conf), -math.sin(angle * conf))
        rotated = vector * self.toComplex()
        return vector2D().fromComplex(rotated)

    def angleWith(self, vector):
        mg = self.magnutude() * vector.magnutude()
        radians = math.acos(round(self.dot(vector) / mg, 6))
        return radians * 180 / math.pi

    def angleWithX(self):
        vector = vector2D(1, 0)
        mg = self.magnutude()
        radians = math.acos(round(self.dot(vector) / mg, 6))
        if self.vy >= 0:
            return radians * 180 / math.pi
        else:
            return 360 - radians * 180 / math.pi

    def inBetween(self, vector1, vector2):
        return self.angleWith(vector2) < vector1.angleWith(vector2)

    def toArray(self):
        return [self.vx, self.vy]

    def fromArray(self, arr):
        return vector2D(arr[0], arr[1])

    def toComplex(self):
        return complex(self.vx, self.vy)

    def fromComplex(self, com):
        return vector2D(com.real, com.imag)

    # overloaded operations

    def __add__(self, vector):
        return vector2D(self.vx + vector.vx, self.vy + vector.vy)
        pass

    def __sub__(self, vector):
        return vector2D(self.vx - vector.vx, self.vy - vector.vy)

    def __mul__(self, const):
        return vector2D(self.vx * const, self.vy * const)
        return
