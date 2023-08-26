from gui import InputFieldPro, LabelList, TextMesh
from util_functions import suggestions, naiveSuggestion, dicToStruct, listToStruct, parser, dicRef
import pygame


class shell():
    def __init__(self, x, y, inf_size, label_size) -> None:

        self.inf_size = inf_size
        self.label_size = label_size
        self.window = None
        self.inputFld = InputFieldPro(x, y, inf_size[0], inf_size[1], ">>>", trigger=self.getCommand,
                                      trigger2=self.hinter)
        self.inputFld.activeColor = (200, 0, 0)
        self.labelLst = LabelList(x, y + inf_size[1], label_size[0], label_size[1])

        self.hint = TextMesh("", 0, 0, (0, 0, 0), (0, 200, 0), 15)

        self.referObjects = {}
        self.isactive = False

        self.commands = ['get', 'set', 'add', "clr", "rem", "type"]

        self.recentCommands = ""
        pass

    def toggleActivity(self):
        self.isactive = not self.isactive
        print(self.isactive)

    def setRefObj(self, name, obj):
        self.referObjects[name] = obj

    def getCommand(self, text):
        self.recentCommands = text

        PREFIX = ">>> "
        self.hint.setText("")

        # here is teh command revalution
        t, c, s = self.commandEval(text)
        if t:
            self.labelLst.addLabel(PREFIX + t, c, s)
        self.renderWindow()
        pass

    def hinter(self, text):
        tokens = text.split(' ')

        if (len(tokens) == 1):

            sglist = naiveSuggestion(tokens[0], self.commands)
            self.hint.setText(" :: ".join(sglist))

            pass
        elif (len(tokens) == 2):
            sglist = naiveSuggestion(tokens[1], self.referObjects.keys())
            self.hint.setText(" :: ".join(sglist))

            pass
        elif (len(tokens) == 3):
            if tokens[1] in self.referObjects.keys():
                k = vars(self.referObjects[tokens[1]]).copy()
                sglist = naiveSuggestion(tokens[2], dict(k).keys())
                self.hint.setText(" :: ".join(sglist))

            pass
        self.hint.setPos(self.inputFld.x + len(text), self.inputFld.y - self.hint.TextRect.height)
        if self.inputFld.Text == "":
            self.hint.setText("")
        pass

    def setPos(self, x, y):
        self.inputFld.setPos(x, y)
        self.labelLst.setPos(x, y)

    def loadWindow(self, window):
        self.window = window
        self.inputFld.loadWindow(window)
        self.labelLst.loadWindow(window)
        self.hint.loadWindow(window)
        pass

    def eventRender(self, event):
        if self.isactive:
            if pygame.key.get_pressed()[pygame.K_DOWN]:
                if self.recentCommands:
                    self.inputFld.Text = self.recentCommands

            self.inputFld.eventRender(event)

        pass

    def renderWindow(self):
        if self.isactive:
            self.labelLst.renderWindow()
            self.inputFld.renderWindow()
            self.hint.renderWidget()
        pass

    def commandEval(self, cmd):
        tokens = cmd.split(' ')
        fxn = tokens[0]
        data = tokens[1:]

        SUCCESS = (0, 200, 0)
        FAIL = (200, 0, 0)
        CAUTION = (200, 200, 0)

        NORMSIZE = 15
        MEDSIZE = 16
        BIGSIZE = 17

        ERRROCODE1 = "INVALID REFERNCE"
        ERRORCODE2 = "INVALID COMMAND"
        ERRORCODE3 = "INVAILD ATRRIBUTE"
        ERRORCODE4 = "PRIMITIVE SET VALID ONLY [int , str]"
        ERRORCODE5 = "TYPE ERROR <str , int , float..>"

        EMPTY_RES = ["", None, None]

        if (fxn == 'get' and len(tokens) == 3):

            if (data[0] in self.referObjects.keys()):
                try:
                    obj = getattr(self.referObjects[data[0]], data[1])

                    print("type checking", type(obj), type(obj) is str or type(obj) is int, type(obj) is str,
                          type(obj) is int)

                    if type(obj) is str or type(obj) is int:

                        return [str(obj), SUCCESS, NORMSIZE]

                    elif type(obj) is list:
                        self.setRefObj(data[1], listToStruct(obj, data[1]))
                        return EMPTY_RES
                        pass

                    elif type(obj) is dict:
                        self.setRefObj(data[1], dicToStruct(obj, data[1]))
                        return EMPTY_RES
                        pass
                    else:
                        self.setRefObj(data[1], obj)
                        return EMPTY_RES

                except:

                    return [ERRORCODE3, FAIL, BIGSIZE]
            else:
                return [ERRROCODE1, FAIL, BIGSIZE]

        if (fxn == 'set' and len(tokens) == 4):
            if (data[0] in self.referObjects.keys()):
                try:

                    obj = getattr(self.referObjects[data[0]], data[1])
                    if type(obj) is int or type(obj) is str:
                        setattr(self.referObjects[data[0]], data[1], data[2])
                        return [f"{data[0]} set to {data[2]}", SUCCESS, NORMSIZE]

                    elif obj == None:
                        return ["UNABLE TO DECIDE THE TYPE", CAUTION, MEDSIZE]
                    else:
                        return [ERRORCODE4, FAIL, BIGSIZE]
                except:

                    return [ERRORCODE3, FAIL, BIGSIZE]
            else:
                return [ERRROCODE1, FAIL, BIGSIZE]

            pass

        if (fxn == 'add' and len(tokens) == 4):

            if (data[0] in self.referObjects.keys()):
                obj = self.referObjects[data[0]]

                if ("thisName" in dict(vars(obj)).keys()):
                    print("yes")
                    objname = obj.thisName
                    if (obj.type == 'list'):
                        print("yeah")
                        value = parser(data[1], data[2])
                        if value:
                            obj.this.append(value)
                            del self.referObjects[objname]
                            self.referObjects[objname] = listToStruct(obj.this, objname)
                            return [f"{objname} --> {obj.this} ", SUCCESS, MEDSIZE]
                        else:
                            return [ERRORCODE5, FAIL, BIGSIZE]

                else:

                    return [ERRROCODE1, FAIL, BIGSIZE]
            else:
                return [ERRROCODE1, FAIL, BIGSIZE]

            pass

        if (fxn == 'rem' and len(tokens) in [3]):

            if (data[0] in self.referObjects.keys()):
                obj = self.referObjects[data[0]]

                if ("thisName" in dict(vars(obj)).keys()):
                    print("yes")
                    objname = obj.thisName
                    if (obj.type == 'list'):
                        print("yeah")
                        try:
                            obj.this.pop(int(data[1]))
                            del self.referObjects[objname]
                            self.referObjects[objname] = listToStruct(obj.this, objname)
                            return [f"{objname} --> {obj.this} ", SUCCESS, MEDSIZE]
                        except:
                            return ["INVALID INDEX", FAIL, BIGSIZE]

                    if (obj.type == 'dict'):
                        print("yeah")
                        try:
                            del obj.this[dicRef(data[1])]
                            del self.referObjects[objname]
                            self.referObjects[objname] = dicToStruct(obj.this, objname)
                            return [f"{objname} --> {obj.this} ", SUCCESS, MEDSIZE]

                        except:

                            return ["INVALID INDEX", FAIL, BIGSIZE]




                else:

                    return [ERRROCODE1, FAIL, BIGSIZE]
            else:
                return [ERRROCODE1, FAIL, BIGSIZE]

            pass

        # some otehr coomamdns
        if (fxn == 'clr' and len(tokens) == 1):
            while (self.labelLst.queue):
                self.labelLst.removeLabel()
            return EMPTY_RES

        # some otehr coomamdns

        else:
            return [ERRORCODE2, FAIL, BIGSIZE]
