import ctypes

myappid = 'mycompany.myproduct.subproduct.version'  # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

import pygame
from pygame.locals import *  # constants
import sys
from singltions.file_sl import basicSaving, saverLoader
from gui import InputFieldPro, LabelList, ImageButton, LoadingGui
from shells.shells_base import shell
from singltions.vision_pointer import vision_pointer
import webbrowser
from util_functions import Timer
import threading

# vision
from gesture_recognizer import GestureRec

# gesture control
gesture = GestureRec(True)

# active event
_MY_ACTIVE_EVENT_ = pygame.USEREVENT + 1
active_event = pygame.event.Event(_MY_ACTIVE_EVENT_)

# handle of the gesture thread and flag to break the while loop within
handle = None
flag = None

# vision button
vision_but = None
vision_lg = None
is_on_vision = False


# vision toggler

def toggleVision():
    global handle, flag, vision_but, is_on_vision
    if handle and handle.is_alive():
        flag.set()
        handle.join()
        vision_but.set_image("gateRes/vision_inactive_icon.png")
        is_on_vision = False

    else:
        # stop flag
        flag = threading.Event()
        handle = threading.Thread(target=gesture.main, args=[flag, active_event])
        handle.start()
        # vision_but.set_image("gateRes/vision_pending_icon.png")
        vision_lg.againStartLoading()
        is_on_vision = True


# web
COMMUNITY_LINK = "https://sample-app-71dda.web.app"

BLACKCOLOR = (150, 150, 150)
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FRAMES_PER_SECOND = 70

# cell size limits
MAX_CELL_SIZE = 64
MIN_CELL_SIZE = 16
NORMAL_CELL_SIZE = 32

# varible s
cellsize = NORMAL_CELL_SIZE
origin_shift = [0, 0]

# some call backs and small fucntipns

# pyagme game smothing

pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("BGV SIMULATOR")
pygame.image.load("gateRes/icon.png")
pygame.display.set_icon(pygame.image.load("gateRes/icon.png"))
clock = pygame.time.Clock()

import BSV as bsv
import singltions.quick_bar_ as menu

manager = bsv.gateManager()

# mamager obj saver
sl = saverLoader(WINDOW_WIDTH - 100, 0, 200, 400)
sl.loadWindow(window)
sl.refeshButtons()

# gui mamnager
guimanager = bsv.gateGuiManager(manager)
guimanager.loadWindow(window)
sl.loadOnBoard(manager)

# communitybutton

com_but = ImageButton("gateRes/feedback_icon.png", 16, WINDOW_HEIGHT - 64, 32,
                      lambda: webbrowser.open_new_tab(COMMUNITY_LINK))
com_but.loadWindow(window)

# visin
vision_lg = LoadingGui(x=64 + 32 / 2, y=WINDOW_HEIGHT - 128 + 32 / 2, size=50, color=(200, 0, 0), clk=0.02, borderW=5,
                       arcSize=300)
vision_but = ImageButton("gateRes/vision_inactive_icon.png", 64, WINDOW_HEIGHT - 128, 32, toggleVision)
vision_but.loadWindow(window)
vision_lg.loadWindow(window)

# vision pointer
vp = vision_pointer()
vp.loadWindow(window)


# fgrid maker
def gridMaker(center, cellSize, color, limits):
    lx, ly = limits
    clx = cellSize
    cx, cy = center
    normalcolor = color

    for i in range(int(lx / (clx))):
        if i == 0:
            color = (200, 0, 0)
        else:
            color = normalcolor

        pygame.draw.line(window, color, (cx + i * clx, 0), (cx + i * clx, ly))
        pygame.draw.line(window, color, ((cx - i * clx), 0), ((cx - i * clx), ly))

        pygame.draw.line(window, color, (0, cy + i * clx), (lx, cy + i * clx))
        pygame.draw.line(window, color, (0, cy - i * clx), (lx, cy - i * clx))


def apply_transform(posvec, trs):
    return [sum([v1 * v2 for v1, v2 in zip(posvec, vec)]) for vec in trs]


# heres is teh shell
shl = shell(0, WINDOW_HEIGHT / 3.7, (200, 50), (200, 300))
shl.loadWindow(window)
shl_lg = LoadingGui(x=16 + 32 / 2, y=WINDOW_HEIGHT - 128 + 32 / 2, size=40, color=(0, 0, 0), clk=0.02, borderW=5,
                    arcSize=290)
shl_lg.loadWindow(window)


def toggleShell():
    shl.toggleActivity()
    shl_lg.againStartLoading()
    Timer(0.5, shl_lg.stopLoading).start()


shl_but = ImageButton("gateRes/shell_icon.png", 16, WINDOW_HEIGHT - 128, 32, toggleShell)
shl_but.loadWindow(window)

# get the obj refernce to shell
shl.setRefObj('manager', manager)
shl.setRefObj('gui', guimanager)

while True:

    # saving the stsus of the mamger object
    sl.saveIntoFile(manager)

    # window setting

    # clearing the screen
    window.fill(BLACKCOLOR)

    gridMaker((WINDOW_WIDTH / 2 + origin_shift[0], WINDOW_HEIGHT / 2 + origin_shift[1]), cellsize, (100, 100, 100),
              [WINDOW_WIDTH, WINDOW_HEIGHT])

    for event in pygame.event.get():  # event loop
        if event.type == pygame.QUIT:

            # bs.saveObj("saved/test.pkl" , manager)

            if is_on_vision: toggleVision()

            pygame.quit()
            sys.exit()

        if pygame.key.get_pressed()[pygame.K_r]:
            guimanager = bsv.gateGuiManager(manager)
            guimanager.loadWindow(window)

        if event.type == pygame.VIDEORESIZE:
            WINDOW_HEIGHT, WINDOW_WIDTH = window.get_height(), window.get_width()

            sl.setAttributes(x=WINDOW_WIDTH - 100)
            sl.refeshButtons()

            shl_but.setPos((16, WINDOW_HEIGHT - 128))
            com_but.setPos((16, WINDOW_HEIGHT - 64))

            vision_but.setPos((64, WINDOW_HEIGHT - 128))
            vision_lg.setPos((64 + 32 / 2, WINDOW_HEIGHT - 128 + 32 / 2))
            shl_lg.setPos((16 + 32 / 2, WINDOW_HEIGHT - 128 + 32 / 2))
            pass

        # here we are scalling the cellesize
        if event.type == pygame.MOUSEWHEEL:
            if event.y < 0:
                cellsize = min(cellsize + 1, MAX_CELL_SIZE)
            else:
                cellsize = max(cellsize - 1, MIN_CELL_SIZE)

            guimanager.setScales(cellsize / NORMAL_CELL_SIZE)
            print(cellsize / NORMAL_CELL_SIZE)

        # here is the origin shift

        if pygame.key.get_pressed()[pygame.K_LSHIFT] and pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            origin_shift = [mx - WINDOW_WIDTH / 2, my - WINDOW_HEIGHT / 2]
            guimanager.setOrigins(origin_shift, globalOrigin=[WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2])
            pass

            # event for vision loading
        if event.type == _MY_ACTIVE_EVENT_:  # vision loaded
            vision_lg.stopLoading()
            vision_but.set_image("gateRes/vision_active_icon.png")

        guimanager.renderevent(event)
        sl.renderEvent(event)

        com_but.eventRender(event)
        vision_but.eventRender(event)

        shl.eventRender(event)
        shl_but.eventRender(event)

        vp.eventRenderer(event)

    shl.renderWindow()
    shl_but.renderWidget()
    guimanager.renderGates()
    sl.renderWindow()
    shl_lg.renderWidget()

    com_but.renderWidget()
    vision_but.renderWidget()
    vision_lg.renderWidget()

    if handle and handle.is_alive():
        vp.follow(pygame.mouse.get_pos())
        vp.renderWidget()

    pygame.display.update()
    clock.tick(FRAMES_PER_SECOND)
