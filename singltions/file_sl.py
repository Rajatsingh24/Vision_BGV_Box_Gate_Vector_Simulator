from functools import partial
import gui as wid
import pygame
import pickle
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile, asksaveasfile


class saverLoader():
    def __init__(self, x, y, width=100, height=100) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = 100
        self.bg = (0, 0, 0)
        self.fg = (200, 200, 200)

        self.window = None

        # interanls
        self.active = False

        # saving
        self.basicSave = basicSaving()
        self.loadedFile = None
        self.savedObject = None

        self.loadOnFile = False

        self.loadingObject = None

        self.texture = wid.textureRect(self.x, self.y, self.width, self.height, color=(0, 0, 0))

        self.saveButton = wid.Button("SAVE", 0, self.y, self.bg, self.fg, 20, self.__saveFile)
        self.saveButton.x = self.x + self.width / 2 - self.saveButton.TextRect.width / 2
        self.saveButton.y = self.y + self.height / 2 - self.saveButton.TextRect.height - 10

        self.loadButton = wid.Button("LOAD", 0, self.y + self.width / 2, self.bg, self.fg, 20, self.__loadFile)
        self.loadButton.x = self.x + self.width / 2 - self.loadButton.TextRect.width / 2
        self.loadButton.y = self.y + self.height / 2 + 10

        self.toggleButton = wid.ImageButton("gateRes/save_load_icon.png", 0, 0, 32, self.__toggleActivity)

    def __saveFile(self):
        if self.savedObject:
            self.basicSave.saveBrowse()
            file = self.basicSave.savedFileHandle
            if file:
                self.basicSave.saveObj(file.name, self.savedObject)
        pass
        self.__toggleActivity()

    def __loadFile(self):
        self.basicSave.BrowseFilename()
        self.loadedFile = self.basicSave.browsedFile
        if self.loadOnFile and self.loadingObject:
            self.loadFromFile(self.loadingObject)
        self.__toggleActivity()

    def loadOnBoard(self, obj):
        self.loadOnFile = True
        self.loadingObject = obj

    def loadFromFile(self, obj):
        if self.loadedFile:
            self.basicSave.loadObj(self.loadedFile.name, obj)

    def saveIntoFile(self, obj):
        self.savedObject = obj

        pass

    def __toggleActivity(self):
        self.active = not self.active
        pass

    def setAttributes(self, x=None, y=None, width=None, height=None, bg=None, fg=None):
        if x: self.x = x
        if y: self.y = y
        if width: self.width = width
        if height: self.height = height
        if bg: self.bg = bg
        if fg: self.fg = fg
        pass

    def refeshButtons(self):
        self.x = self.window.get_width() / 2 - self.width / 2
        self.y = self.window.get_height() / 2 - self.height / 2

        self.texture = wid.textureRect(self.x, self.y, self.width, self.height, color=(0, 0, 0))
        self.saveButton = wid.Button("SAVE", 0, self.y, self.bg, self.fg, 20, self.__saveFile)
        self.saveButton.x = self.x + self.width / 2 - self.saveButton.TextRect.width / 2
        self.saveButton.y = self.y + self.height / 2 - self.saveButton.TextRect.height - 10

        self.loadButton = wid.Button("LOAD", 0, self.y + self.width / 2, self.bg, self.fg, 20, self.__loadFile)
        self.loadButton.x = self.x + self.width / 2 - self.loadButton.TextRect.width / 2
        self.loadButton.y = self.y + self.height / 2 + 10

        self.toggleButton = wid.ImageButton("gateRes/save_load_icon.png", 64, self.window.get_height() - 64, 32,
                                            self.__toggleActivity)

        # loading the window
        self.loadWindow(self.window)

    def loadWindow(self, window):
        self.window = window
        self.texture.loadWindow(window)
        self.saveButton.loadWindow(window)
        self.loadButton.loadWindow(window)
        self.toggleButton.loadWindow(window)

        pass

    def renderEvent(self, event):
        self.toggleButton.eventRender(event)
        if self.active:
            self.saveButton.eventRender(event)
            self.loadButton.eventRender(event)

        pass

    def renderWindow(self):

        self.toggleButton.renderWidget()

        if self.active:
            self.texture.renderWidget()
            self.saveButton.renderWidget()
            self.loadButton.renderWidget()

        pass


class basicSaving():
    def __init__(self) -> None:

        self.browsedFile = None
        self.loadedFileDic = None
        self.savedFileHandle = None

        pass

    def load(self, filename):
        fp = open(filename, "rb")
        value = pickle.load(fp)
        fp.close()
        return value
        pass

    def save(self, filename, dic):

        with open(filename, "wb") as fp:
            pickle.dump(dic, fp)
        pass

    def saveObj(self, filename, object):
        obj_attrib_dic = vars(object)
        self.save(filename, obj_attrib_dic)

    def loadObj(self, filename, obj):
        for k, v in self.load(filename).items():
            setattr(obj, k, v)

    def BrowseFilename(self):
        win = Tk()
        win.geometry("200x100")
        win.title("SELECT")
        win.iconphoto(False, PhotoImage(file="gateRes/icon.png"))

        def open_file():
            file = filedialog.askopenfile(mode='r', filetypes=[("All Files", "*.*")])
            if file:
                self.browsedFile = file
                win.destroy()
                # return file.name

        ttk.Button(win, text="Browse", command=open_file).pack(pady=20)
        win.mainloop()

    def loadBrowse(self):
        win = Tk()
        win.geometry("200x100")
        win.title("LOAD")
        win.iconphoto(False, PhotoImage(file="gateRes/icon.png"))

        def open_file():
            file = filedialog.askopenfile(mode='r', filetypes=[("All Files", "*.*"), ("Text Documents", "*.txt")])
            if file:
                self.loadedFileDic = self.load(file.name)
                win.destroy()

        ttk.Button(win, text="Browse", command=open_file).pack(pady=20)
        win.mainloop()

    def saveBrowse(self):

        win = Tk()
        win.geometry("200x100")
        win.title("SAVE")
        win.iconphoto(False, PhotoImage(file="gateRes/icon.png"))

        def on_closing():
            win.destroy()

        def save_file():
            f = filedialog.asksaveasfile(initialfile='Untitled.txt', defaultextension=".txt",
                                         filetypes=[("All Files", "*.*"), ("Text Documents", "*.txt")])
            self.savedFileHandle = f
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_closing)
        btn = Button(win, text="Save", command=lambda: save_file())
        btn.pack(pady=10)
        win.mainloop()
