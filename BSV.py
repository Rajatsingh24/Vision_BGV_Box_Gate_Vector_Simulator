from functools import reduce, partial
import gui
import pygame
import singltions.quick_bar_ as menu
import singltions.selection_tool as selection
import singltions.path_formulator as LPF
import singltions.prop_bar as propbar
from collections import defaultdict
from util_functions import Timer

# ssome conatants
TERMINAL_INP = 0
TERMINAL_OUT = 1
GATE_BOX = 2

# some prime gate vectors
AND_GATE_VEC = [0, 0, 0, 1]
OR_GATE_VEC = [0, 1, 1, 1]
NOT_GATE_VEC = [1, 0]
XOR_GATE_VEC = [0, 1, 1, 0]
NAND_GATE_VEC = [1, 1, 1, 0]
NOR_GATE_VEC = [1, 0, 0, 0]


class GATEBOX():

    def __init__(self, gateType=TERMINAL_INP, inputs=0, outputs=1, gateVector=[0, 1]) -> None:

        self.inputs = inputs
        self.outputs = outputs
        self.gatetype = gateType
        self.gateLabel = "INP"

        # inputa outputs
        self.input_id_array = [None for _ in range(inputs)]
        self.output_id = []

        # gatevector
        self.gateVector = gateVector
        # output vector
        self.outputVector = [0, 1] if gateType == TERMINAL_INP else None

        # inputs output bits
        self.intputBits = [None for _ in range(inputs)]
        self.outputBit = 0 if gateType == TERMINAL_INP else None

        # run lfag
        self.is_run_complete = False

        # here we have some positionl variables
        self.pos = [100, 100]
        self.size = [70, 30]

        pass

    def setOutBit(self):
        self.outputBit = 1

    def resetOutBit(self):
        self.outputBit = 0

    def setLabel(self, label):
        self.gateLabel = label

    def setPosition(self, pos):
        self.pos = pos

    def runGate(self, gatemanager):
        if self.gatetype == TERMINAL_INP:
            return True

        iter = [-1 if _ == self.inputs - 1 else 0 for _ in range(self.inputs)]
        limits = [len(gatemanager.gate_dic[id].outputVector) for id in self.input_id_array]
        iter_index = len(iter) - 1

        self.outputVector = []
        # copy the prevous output bits
        self.intputBits = [gatemanager.gate_dic[id].outputBit for id in self.input_id_array]

        while (True):
            if (iter[iter_index] == limits[iter_index] - 1):
                iter[iter_index] = 0
                if (iter_index == 0): break
                iter_index -= 1
                continue

            iter[iter_index] += 1
            if not iter_index == len(iter) - 1 and iter[iter_index + 1] == 0:
                iter_index += 1

            # here goesth dial of the input vectors
            # first getting the binary index
            # print(">>>>>>>>>" ,self.input_id_array , iter , [gatemanager.gate_dic[id].outputVector  for id  , i in zip(self.input_id_array , iter)])
            binary_index = [gatemanager.gate_dic[id].outputVector[i] for id, i in zip(self.input_id_array, iter)]
            binary_index.reverse()

            dial_index = self.__util_binaryConverter(binary_index)

            self.outputVector.append(self.gateVector[dial_index])

        self.outputBit = self.gateVector[self.__util_binaryConverter(self.intputBits)]

        for id in self.input_id_array:
            if gatemanager.gate_dic[id].gateLabel != "INP" and gatemanager.gate_dic[id].gateLabel != "CLK":
                gatemanager.gate_dic[id].outputVector = None

        print("---", self.intputBits, self.outputVector, self.outputBit)

    def runGate_feedback(self, gatemanager):
        if self.gatetype == TERMINAL_INP:
            return True

        iter = [-1 if _ == self.inputs - 1 else 0 for _ in range(self.inputs)]
        limits = [len(gatemanager.gate_dic[id].outputVector) for id in self.input_id_array]
        iter_index = len(iter) - 1

        self.outputVector = self.gateVector

        # copy the prevous output bits

        self.intputBits = [gatemanager.gate_dic[id].outputBit for id in self.input_id_array]

        self.outputBit = self.gateVector[self.__util_binaryConverter(self.intputBits)]

        print(self.intputBits, self.outputBit, self.outputVector)
        for id in self.input_id_array:
            if gatemanager.gate_dic[id].gateLabel != "INP" and gatemanager.gate_dic[id].gateLabel != "CLK":
                gatemanager.gate_dic[id].outputVector = None

    def isRunnable(self, gatemanager):
        if self.gatetype == TERMINAL_INP: return True
        for id in self.input_id_array:
            if (id == None): return False
            if gatemanager.gate_dic[id].outputVector == None: return False
        return True

    def isConnected(self):
        if self.gatetype == TERMINAL_INP: return True
        for id in self.input_id_array:
            if (id == None): return False
        return True

    def __util_binaryConverter(self, binary_index):
        dial_index = 0
        weight = 1
        for i in binary_index:
            dial_index += weight * i
            weight *= 2
        return dial_index

    def clearOutput(self):
        self.outputBit = None
        self.outputVector = None
        pass


class gateManager():
    def __init__(self) -> None:
        self.id_init = 0
        self.gate_dic = {}
        pass

    def setpos(self, id, pos):
        if (id in self.gate_dic.keys()):
            self.gate_dic[id].pos = pos

    def setlabel(self, id, label):
        if (id in self.gate_dic.keys()):
            self.gate_dic[id].gateLabel = label

    def addGate(self, gateObj):
        self.id_init += 1
        self.gate_dic.update({self.id_init: gateObj})
        return self.id_init

    def removeGate(self, gateId):

        # output to input connection removals
        for id in self.gate_dic[gateId].input_id_array:
            if id:
                if self.gate_dic[id].output_id:
                    self.gate_dic[id].output_id.remove(gateId)  # we are removing the id directly
                    # because the whole gate is removed

        # input connection removal from output
        for index in range(len(self.gate_dic[gateId].input_id_array)):
            self.gate_dic[gateId].input_id_array[index] = None

            # for the output connection
        for id in self.gate_dic[gateId].output_id:
            print("---->", id, gateId)
            for index, inp_id in enumerate(self.gate_dic[id].input_id_array):
                if (gateId == inp_id):
                    self.gate_dic[id].input_id_array[index] = None

                    # finally remove the gate
        del self.gate_dic[gateId]

    def setTerminalGate(self, id):
        if (self.gate_dic[id].gatetype == TERMINAL_INP):
            self.gate_dic[id].outputBit = 1

    def resetTerminalGate(self, id):
        if (self.gate_dic[id].gatetype == TERMINAL_INP):
            self.gate_dic[id].outputBit = 0

    def getTerminalOutput(self, id):
        if (self.gate_dic[id].gatetype == TERMINAL_INP):
            return self.gate_dic[id].outputBit
        return None

    def connectGatesToInput(self, gateId_1, gateId_2, input_index):
        self.gate_dic[gateId_2].input_id_array[input_index] = gateId_1
        if gateId_2 not in self.gate_dic[gateId_1].output_id:
            self.gate_dic[gateId_1].output_id.append(gateId_2)
        pass

    def disconnectGatesToInput(self, gateId_1, gateId_2, input_index):
        self.gate_dic[gateId_2].input_id_array[input_index] = None
        for id in self.gate_dic[gateId_2].input_id_array:
            if id == gateId_1:
                return
        # otherwise we there dont exist another connections
        self.gate_dic[gateId_1].output_id.remove(gateId_2)
        # self.gate_dic[gateId_1].output_id.

    def clearGateOutputs(self):
        for gateids in self.outputGateIds():
            self.gate_dic[gateids].clearOutput()
        pass

    def simulate(self):
        if len(self.gate_dic) == 0: return
        for outIds in self.outputGateIds():
            pending_stack = []
            gateid_iter = outIds
            while True:
                if not self.gate_dic[gateid_iter].isConnected(): break

                if (self.gate_dic[gateid_iter].isRunnable(self)):
                    self.gate_dic[gateid_iter].runGate(self)
                    if (pending_stack):
                        gateid_iter = pending_stack.pop()
                    else:
                        break
                else:
                    pending_stack.append(gateid_iter)
                    for inpgateId in self.gate_dic[gateid_iter].input_id_array:
                        # gate is not coonected
                        if (self.gate_dic[inpgateId].outputVector == None):
                            gateid_iter = inpgateId
                            break

    def simulateFeed(self):
        if len(self.gate_dic) == 0: return

        for outIds in self.outputGateIds():
            dead_lock = defaultdict(int)
            pending_stack = []
            gateid_iter = outIds
            deadLockBit = self.detectDeadLock(outIds)
            print("deadted_deadLocks: ", self.detectDeadLock(outIds))
            while True:
                if not self.gate_dic[gateid_iter].isConnected(): break

                if (self.gate_dic[gateid_iter].isRunnable(self)):

                    if deadLockBit:
                        self.gate_dic[gateid_iter].runGate_feedback(self)
                    else:
                        self.gate_dic[gateid_iter].runGate(self)

                    if (pending_stack):
                        gateid_iter = pending_stack.pop()
                    else:
                        break
                else:
                    pending_stack.append(gateid_iter)
                    dead_lock[gateid_iter] += 1

                    if (dead_lock[gateid_iter] > 1):
                        print(pending_stack, " ", dead_lock[gateid_iter])

                        self.gate_dic[gateid_iter].outputBit = 0
                        self.gate_dic[gateid_iter].outputVector = []
                        pending_stack.pop()

                        gateid_iter = pending_stack.pop()

                        continue
                    for inpgateId in self.gate_dic[gateid_iter].input_id_array:
                        # gate is not coonected
                        if (self.gate_dic[inpgateId].outputVector == None):
                            gateid_iter = inpgateId
                            break

    def detectDeadLock(self, gateIdOutput):

        dead_lock = defaultdict(int)
        pending_stack = []
        gateid_iter = gateIdOutput

        saved_outs = {
            id: [self.gate_dic[id].outputBit, self.gate_dic[id].outputVector]
            for id in self.outputGateIds()
        }

        self.clearGateOutputs()

        while True:
            if not self.gate_dic[gateid_iter].isConnected(): return None
            if self.gate_dic[gateid_iter].isRunnable(self):

                for id in self.outputGateIds():
                    setattr(self.gate_dic[id], 'outputBit', saved_outs[id][0])
                    setattr(self.gate_dic[id], 'outputVector', saved_outs[id][1])

                return False
            else:
                pending_stack.append(gateid_iter)
                dead_lock[gateid_iter] += 1
                if (dead_lock[gateid_iter] > 1):

                    for id in self.outputGateIds():
                        setattr(self.gate_dic[id], 'outputBit', saved_outs[id][0])
                        setattr(self.gate_dic[id], 'outputVector', saved_outs[id][1])

                    return True
                for inpgateId in self.gate_dic[gateid_iter].input_id_array:
                    # gate is not coonected
                    if (self.gate_dic[inpgateId].outputVector == None):
                        gateid_iter = inpgateId
                        break

    def simulationEngine(self):

        pass

    def getFinalGateVectors(self):
        # here you can extract the final output
        outgate_vectors = {}
        for k, v in self.gate_dic.items():
            if (v.gatetype == TERMINAL_OUT):
                outgate_vectors.update({k: v.outputVector})
                pass
        return outgate_vectors

    def getFinalGateOutputs(self):
        return {gateId: self.gate_dic[gateId].outputBit for gateId in self.outputGateIds()}

    def outputGateIds(self):
        outgate_ids = []
        for k, v in self.gate_dic.items():
            if (v.gatetype == TERMINAL_OUT):
                outgate_ids.append(k)
                pass
        return outgate_ids


class gateGuiManager():

    def __init__(self, gatemanger) -> None:

        self.gatemanager = gatemanger
        self.window = None

        # internal state
        self.selected_gate = None
        self.displacement_vector = [0, 0]

        self.recent_selected = None
        self.selected_input = [None, None]

        self.selected_connection = None
        self.selected_out_in = [None, [None, None]]

        self.scale = 1

        # buffer que to handlle alls the pending opeartions
        self.operationBuffer = []

        self.gateGUI_dic = {id: gateGUI(gate, partial(self.gate_request, id)) for (id, gate) in
                            gatemanger.gate_dic.items()}
        self.gateConnectionGUi = []

        # initlixeting the connections
        self.getConnectionGui()

        # quick bar
        self.btn_l = ["(+) ADD GATE", "(-) DELETE GATE",
                      "(~) DISCONNECT", "(||) DUPLICATE",

                      "(<<)CLEAR", "(*) SIMULATE", "(><)COMPRESS", "(C-)CLOCK"]

        sub_btn_l = ["AND", "OR", "NOT", "XOR", "NAND", 'NOR', "INPUT", "OUTPUT", "CLOCK"]

        self.menu_bar = menu.quickBar((0, 0), 20, buttonLabels=self.btn_l, trigger=self.menu_button_pressed)

        self.sub_menu_bar = menu.quickBar((0, 0), 20, buttonLabels=sub_btn_l, trigger=self.sub_menu_button_pressed)
        self.sub_menu_bar.modeBit = [0, 1]

        # selection tool

        self.selection_tool = selection.selectionTool(self.__callback_selection, self.__callback_deselection)

        # here we have the

        self.propert_bar = propbar.propertyBar(0, 0)

        # here we hav animation for the thing
        self.info_toggler = gui.ImageButton("gateRes/help_icon.png", 0, 0, 32, self.toggle_info_bar)
        self.liner_anim = LPF.LinearPathForm([0, -200], [0, 0], 1, self.info_anim, False, 1)
        self.info = propbar.propertyBar(0, -200, layout=None)

        self.info.props = {
            'ctrl+RMB': 'ON/OFF selected terminal INP',
            'LSHIFT+RMB': 'MENU > ADD , DELETE... GATES',
            'LMB': 'select/pick gate',
            'LSHIFT + LMB DRAG': 'change grid origin',
            'MOUSE WHEEL': 'scale the grid'
        }
        self.info.labelcolor = (200, 0, 0)
        self.info.valueColor = (200, 200, 200)
        self.info.refershRender()

        # connection guis
        self.selected_connection_color = (0, 200, 0)
        self.noraml_connection_color = (0, 0, 0)

        # timer for clk
        self.timer = None

    def toggle_info_bar(self):
        if self.liner_anim.do_start == False:
            self.liner_anim.startPath()
        else:
            self.liner_anim.resetPath()
        pass

    def info_anim(self, loc):
        self.info.x, self.info.y = list(map(int, loc))
        self.info.refershRender()

    def __callback_selection(self, List):
        self.__callback_deselection(List)
        for id in List:
            self.gateGUI_dic[id].gate_group_selected = True
            self.gateGUI_dic[id].getGateGui()
            self.gateGUI_dic[id].loadWindow(self.window)

        pass

    def __callback_deselection(self, List):

        for id in self.gateGUI_dic.keys():
            self.gateGUI_dic[id].gate_group_selected = False
            self.gateGUI_dic[id].getGateGui()
            self.gateGUI_dic[id].loadWindow(self.window)

        pass

    def setOrigins(self, origin, globalOrigin):
        for gate in self.gateGUI_dic.values():
            gate.setOrigin(origin, globalOrigin)

        self.getConnectionGui()
        for connection in self.gateConnectionGUi:
            connection[3].loadWindow(self.window)

    def setScales(self, scale):
        for gate in self.gateGUI_dic.values():
            gate.setScale(scale)
            self.scale = scale
        self.getConnectionGui()
        for connection in self.gateConnectionGUi:
            connection[3].loadWindow(self.window)

    def loadWindow(self, window):

        self.window = window
        self.menu_bar.loadWinow(window)
        self.sub_menu_bar.loadWinow(window)
        self.selection_tool.loadwindow(window)
        self.propert_bar.loadWindow(window)

        # hrer we are also settin the pos which needs the window
        self.info_toggler.loadWindow(window)
        self.info_toggler.setPos([128, self.window.get_height() - self.info_toggler.BoxWidth - 32])

        self.info.loadWindow(window)

        for gate in self.gateGUI_dic.values():
            gate.loadWindow(window)

        for connection in self.gateConnectionGUi:
            connection[3].loadWindow(window)

    def renderevent(self, event):

        self.sub_menu_bar.renderEvent(event)
        self.menu_bar.renderEvent(event)
        self.info_toggler.eventRender(event)

        self.selection_tool.renderEvent(event)

        for gate in self.gateGUI_dic.values():
            gate.renderevent(event)

        for connection in self.gateConnectionGUi:
            connection[3].renderEvent(event)

        # handking teh gate selceting
        # here we handle all user interations
        # handling the gate position on selection

        if self.selected_gate:

            self.gateGUI_dic[self.selected_gate].proceduralGate.pos = [a + b for a, b in zip(pygame.mouse.get_pos(),
                                                                                             self.displacement_vector)]

            # refresh the gui buffer for the selected gate gui
            self.gateGUI_dic[self.selected_gate].getGateGui()
            self.gateGUI_dic[self.selected_gate].loadWindow(self.window)

            # refresing the gate buffer for the connections
            self.getConnectionGui()
            for connection in self.gateConnectionGUi:
                connection[3].loadWindow(self.window)

            pass

        # some events
        # desecting the connections
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and not pygame.key.get_pressed()[pygame.K_LSHIFT]:
            self.selected_connection = None
            self.selected_out_in = [None, [None, None]]

            self.getConnectionGui()
            for connection in self.gateConnectionGUi:
                connection[3].loadWindow(self.window)
            pass

        # toggling the state of the selected gate output bit // only for terminal box
        shortcut = event.type == pygame.KEYDOWN and event.key == pygame.K_RALT
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and pygame.key.get_pressed()[
            pygame.K_LCTRL] or shortcut:
            if self.recent_selected:

                Bitstatus = self.gatemanager.getTerminalOutput(self.recent_selected)

                if (Bitstatus == 0):
                    self.gatemanager.setTerminalGate(self.recent_selected)
                elif (Bitstatus == 1):
                    self.gatemanager.resetTerminalGate(self.recent_selected)

                # setting the gui
                self.gateGUI_dic[self.recent_selected].getGateGui()
                self.gateGUI_dic[self.recent_selected].loadWindow(self.window)

        # here is the clock runnung prop
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and pygame.key.get_pressed()[pygame.K_LCTRL]:
            if self.recent_selected and self.gatemanager.gate_dic[self.recent_selected].gateLabel == "CLK":
                self.operationBuffer.append(['SIM', 20, self.recent_selected])
            pass

        # here isthe event of window ressing
        if event.type == pygame.VIDEORESIZE:
            # propert y bar
            self.propert_bar.doubleRefresh()
            # toggle buttons
            self.info_toggler.setPos([128, self.window.get_height() - self.info_toggler.BoxWidth - 32])

        # vision key short
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RSHIFT:
            self.operationBuffer.append(["SIM"])

    def renderGates(self):
        # handling the opeariong buffer queue
        iter_len = len(self.operationBuffer)
        while iter_len > 0:
            iter_len -= 1
            operation = self.operationBuffer.pop(0)

            if (operation[0] == 'DEL'):
                # removing thegates from the work becnk
                if self.recent_selected:
                    self.gatemanager.removeGate(self.recent_selected)
                    del self.gateGUI_dic[self.recent_selected]
                    # refesing the connections

                    # desecting the coonection incase it was coonected to deledted gate
                    self.selected_connection = None

                    self.getConnectionGui()
                    for connection in self.gateConnectionGUi:
                        connection[3].loadWindow(self.window)

                    self.recent_selected = None

                pass

            if (operation[0] == 'DIS'):
                if self.selected_connection:
                    # removing the logoical connection
                    index, connection = self.selected_connection
                    g1, g2, i = connection
                    self.gatemanager.disconnectGatesToInput(g1, g2, i)

                    # removing the gui connection
                    self.gateConnectionGUi.pop(index)

                    # this conenction has been removed
                    self.selected_connection = None

                    # refreshing th gui conenctions
                    self.getConnectionGui()
                    for connection in self.gateConnectionGUi:
                        connection[3].loadWindow(self.window)

            if operation[0] == "SIM":
                # try:
                self.gatemanager.simulateFeed()
                for id, bit in self.gatemanager.getFinalGateOutputs().items():
                    self.gateGUI_dic[id].getGateGui()
                    self.gateGUI_dic[id].loadWindow(self.window)

            # except:
            # print("CLEAR THE BITS INCASE OF FEEDBACK CIRCUITS")

            # if operation[0] == "SIMF":
            #     self.gatemanager.simulateFeed()
            #     for id , bit in self.gatemanager.getFinalGateOutputs().items():
            #         self.gateGUI_dic[id].getGateGui()
            #         self.gateGUI_dic[id].loadWindow(self.window)

            if operation[0] == "CLR":
                # gateIds = list(self.gateGUI_dic.keys())

                # slection prioprity
                gateIds = self.selection_tool.selectionSet
                if gateIds == []: gateIds = list(self.gateGUI_dic.keys())

                for gateid in gateIds:
                    self.gatemanager.removeGate(gateid)
                    del self.gateGUI_dic[gateid]

                    # refesing the connections
                    # desecting the coonection incase it was coonected to deledted gate
                    self.selected_connection = None

                    self.getConnectionGui()
                    for connection in self.gateConnectionGUi:
                        connection[3].loadWindow(self.window)

                    self.recent_selected = None

            # dupicating the gates here
            if operation[0] == "DUP":
                print("key", operation[1])
                mapToIndex = {'AND': 0, 'OR': 1, 'NOT': 2, 'XOR': 3, 'NAND': 4, 'NOR': 5, 'INP': 6, 'OUT': 7}
                gateLabel = self.gateGUI_dic[operation[1]].proceduralGate.gateLabel
                self.sub_menu_button_pressed(mapToIndex[gateLabel])

        # here are conenctions
        for connection in self.gateConnectionGUi:
            connection[3].renderWindow()

        # here ist herendering of the gates
        for gate in self.gateGUI_dic.values():
            gate.render()

        # connectecing tha gates
        if self.selected_out_in[0]:
            # gui
            x, y = self.gateGUI_dic[self.selected_out_in[0]].gateTerminal_gui[-1].getPos()
            pygame.draw.line(self.window, self.noraml_connection_color, (x, y), pygame.mouse.get_pos(), 3)
            if self.selected_out_in[1][0]:
                # here the connection is formed
                self.connectGates(self.selected_out_in[0], self.selected_out_in[1][0], self.selected_out_in[1][1])
                self.selected_out_in = [None, [None, None]]

        self.menu_bar.renderWindow()
        self.sub_menu_bar.renderWindow()

        self.selection_tool.renderWindow()
        self.selection_tool.renderConsiderationSet(self.gatemanager.gate_dic.values(), ['pos'],
                                                   list(self.gatemanager.gate_dic.keys()))

        self.propert_bar.renderWindow()

        # rendeign the info
        self.info_toggler.renderWidget()
        self.info.renderWindow()

        # animtion clocks
        self.liner_anim.pathClock()

    def setPos(self, id, pos):
        self.gateGUI_dic[id].proceduralGate.pos = pos

    def serachConnection(self, id1, id2, index):
        for i, connection in enumerate(self.gateConnectionGUi):
            print(connection[:-1], [id1, id2, index])
            if (connection[:-1] == [id1, id2, index]):
                return i

    def gate_request(self, id, selected_part, index=-1):

        # here hangle the gate requests
        if (selected_part == "G"):
            if self.selected_gate == id:
                print(id, selected_part, index, self.selected_gate, self.recent_selected)
                # if some gate was recently slected change its gui to noraml
                if self.recent_selected:
                    print("vector: = ", self.gatemanager.gate_dic[self.recent_selected].outputVector)
                    self.gateGUI_dic[self.recent_selected].selected = False
                    self.gateGUI_dic[self.recent_selected].getGateGui()
                    self.gateGUI_dic[self.recent_selected].loadWindow(self.window)

                self.recent_selected = self.selected_gate
                self.selected_gate = None

                # changet he gui of the new recent selected
                self.gateGUI_dic[self.recent_selected].selected = True
                self.gateGUI_dic[self.recent_selected].getGateGui()
                self.gateGUI_dic[self.recent_selected].loadWindow(self.window)

            else:
                self.selected_gate = id
                x, y = self.gateGUI_dic[self.selected_gate].proceduralGate.pos
                # selected displacement
                mx, my = pygame.mouse.get_pos()
                self.displacement_vector = [x - mx, y - my]
                self.propert_bar.selectedGate(self.gatemanager.gate_dic[self.selected_gate],
                                              ['inputs', 'outputs', 'gatetype', 'gateLabel',
                                               'input_id_array', 'gateVector', 'intputBits',
                                               'outputBit', 'pos', 'size'])

        if selected_part == "I":
            self.selected_out_in[1] = [id, index]

            pass
        if selected_part == "O":
            self.selected_out_in[0] = id
            pass

    def connection_request(self, info):

        # if already selected coonection exists
        if self.selected_connection:
            self.gateConnectionGUi[self.selected_connection[0]][3].setColor("normal", self.noraml_connection_color)

        # setting the buffer fir the connection
        self.selected_connection = [self.serachConnection(*info), info]

        # chaning the gui of newly slectied connection
        self.gateConnectionGUi[self.selected_connection[0]][3].setColor("normal", self.selected_connection_color)

        pass

    def menu_button_pressed(self, index):
        if (index == 0):
            # adding the gate
            self.sub_menu_bar.setPos(self.menu_bar.pos)
            self.sub_menu_bar.showWindow()

            pass
        if (index == 1):
            # delting the gate
            self.operationBuffer.append(['DEL'])
            pass
        if (index == 2):
            # disconnecting the gate
            self.operationBuffer.append(['DIS'])
            pass
        if (index == 3):
            # duplicating the gates
            if self.recent_selected:
                self.operationBuffer.append(['DUP', self.recent_selected])
            pass
        if (index == 4):
            self.operationBuffer.append(['CLR'])
            pass

        if (index == 5):
            self.operationBuffer.append(['SIM'])
            pass
        if (index == 6):
            if self.recent_selected and self.gatemanager.gate_dic[self.recent_selected].gateLabel == "OUT":
                ov = self.gatemanager.gate_dic[self.recent_selected].outputVector
                import math
                ins = math.log2(len(ov))
                self.sub_menu_button_pressed(9, [int(ins), 1, ov])
        if index == 7:
            # clock
            if self.timer:
                self.timer.destroy()
                self.timer = None
                self.btn_l[7] = "(-)CLOCK ON"
            else:
                self.timer = Timer(1, lambda: self.operationBuffer.append(['SIM']), True)
                self.timer.start()
                self.btn_l[7] = "(-)CLOCK OFF"
            pass

        pass

    def sub_menu_button_pressed(self, index, gateinfo=[]):

        gate = None

        if index == 0:  # and gate
            gate = GATEBOX(GATE_BOX, 2, 1, AND_GATE_VEC)
            gate.gateLabel = "AND"

        if (index == 1):  # or ate
            gate = GATEBOX(GATE_BOX, 2, 1, OR_GATE_VEC)
            gate.gateLabel = "OR"
            pass
        if (index == 2):
            gate = GATEBOX(GATE_BOX, 1, 1, NOT_GATE_VEC)
            gate.gateLabel = "NOT"

            pass
        if (index == 3):
            gate = GATEBOX(GATE_BOX, 2, 1, XOR_GATE_VEC)
            gate.gateLabel = "XOR"
            pass
        if index == 4:
            gate = GATEBOX(GATE_BOX, 2, 1, NAND_GATE_VEC)
            gate.gateLabel = "NAND"
            pass

        if (index == 5):
            gate = GATEBOX(GATE_BOX, 2, 1, NOR_GATE_VEC)
            gate.gateLabel = "NOR"
            pass

        if (index == 6):
            gate = GATEBOX()
            gate.gateLabel = "INP"
            pass

        if (index == 7):
            gate = GATEBOX(TERMINAL_OUT, 1)
            gate.gateLabel = "OUT"
            pass
        if index == 8:
            gate = GATEBOX()
            gate.gateLabel = "CLK"

        if index == 9:  # internal call
            gatein, gateouts, gatevec = gateinfo
            gate = GATEBOX(GATE_BOX, gatein, gateouts, gatevec)
            gate.gateLabel = "X"

        gate.pos = self.window.get_width() / 2, self.window.get_height() / 2
        id = self.gatemanager.addGate(gate)
        self.gateGUI_dic.update({id: gateGUI(gate, partial(self.gate_request, id))})
        self.gateGUI_dic[id].loadWindow(self.window)
        self.setScales(self.scale)

    def connectGates(self, id1, id2, inp_index):
        # logical conenctio
        self.gatemanager.connectGatesToInput(id1, id2, inp_index)
        # gui connection
        x, y = self.gateGUI_dic[id2].gateTerminal_gui[inp_index].getPos()
        x1, y1 = self.gateGUI_dic[id1].gateTerminal_gui[-1].getPos()
        connect_gui = gui.interactiveShape(x, y, 'line', [x1, y1, 2],
                                           partial(self.connection_request, [id1, id2, inp_index]))
        connect_gui.loadWindow(self.window)
        connect_gui.setColor('normal', [0, 0, 0])
        connect_gui.setColor('enter', (200, 0, 0))
        self.gateConnectionGUi.append([id1, id2, inp_index, connect_gui])

        pass

    def getConnectionGui(self):

        # flusig the array first
        self.gateConnectionGUi = []

        for id, gate in self.gateGUI_dic.items():
            for input_index, inp_id in enumerate(gate.proceduralGate.input_id_array):

                # some gaurad code
                if (not inp_id): continue
                # here im goint o mkathe gui
                x, y = self.gateGUI_dic[id].gateTerminal_gui[input_index].getPos()
                x1, y1 = self.gateGUI_dic[inp_id].gateTerminal_gui[-1].getPos()

                connect_gui = gui.interactiveShape(0, 0, 'Zline', [[x, y], [x1, y1]],
                                                   partial(self.connection_request, [inp_id, id, input_index]))
                connect_gui.setColor('normal', [0, 0, 0])
                connect_gui.setColor('enter', (200, 0, 0))

                # here is the severe problem
                # self.gateConnectionGUi.update({inp_id:[id , input_index , connect_gui ]})
                self.gateConnectionGUi.append([inp_id, id, input_index, connect_gui])

        # chnaing the selected connection gui
        if self.selected_connection:
            print(self.selected_connection)
            self.gateConnectionGUi[self.selected_connection[0]][3].setColor("normal", self.selected_connection_color)


class gateGUI():
    def __init__(self, gateObj, gate_request) -> None:
        self.window = None
        self.proceduralGate = gateObj
        self.origin = [0, 0]
        self.globalOrigin = [0, 0]
        self.scale = 1
        self.terminalSize = [10, 10]
        self.gate_request = gate_request

        # selection buffer
        self.selected = False
        self.connection_selected = [False, ]
        # gates
        self.selected_gate_color = (200, 0, 0)
        self.normal_gate_color = (150, 150, 150)

        # gate gui features
        self.gateImage = None
        self.baseRect = None
        self.gateTerminal_gui = []

        # groups
        self.gate_group_selected = False
        self.gate_group_selection_rect = None
        self.gate_group_selection_gap = 20

        self.output_gate_color_1 = (0, 255, 0)
        self.output_gate_color_0 = (255, 0, 0)

        self.input_gate_color_1 = (0, 255, 0)
        self.input_gate_color_0 = (255, 0, 0)

        # initilizer
        self.getGateGui()

    def setOrigin(self, origin, globalOrigin):
        self.origin = origin
        self.globalOrigin = globalOrigin
        self.getGateGui()
        self.loadWindow(self.window)
        pass

    def setScale(self, scale):
        self.scale = scale
        self.getGateGui()
        self.loadWindow(self.window)
        pass

    def __useScale(self, pos):
        WINDOW_WIDTH = 1000
        WINDOW_HEIGHT = 700
        # if self.window:
        #     O = [self.window.get_width()/2 + self.origin[0] , self.window.get_height()/2 + self.origin[1]]
        # else:

        O = [WINDOW_WIDTH / 2 + self.origin[0], WINDOW_HEIGHT / 2 + self.origin[1]]

        posx, posy = pos
        return [(posx) * self.scale + (1 - self.scale) * O[0],
                (posy) * self.scale + (1 - self.scale) * O[1]]

        pass

    def loadWindow(self, window):

        self.window = window
        # // herim loading the window for each gui object
        for GUI in self.gateTerminal_gui:
            GUI.loadWindow(window)

        self.baseRect.loadWindow(window)

        if self.gate_group_selection_rect and self.gate_group_selected:
            self.gate_group_selection_rect.loadWindow(window)

    def renderevent(self, event):

        if self.gate_group_selection_rect and self.gate_group_selected:
            self.gate_group_selection_rect.renderEvent(event)

        self.baseRect.renderEvent(event)
        for GUI in self.gateTerminal_gui:
            GUI.renderEvent(event)

        pass

    def render(self):
        # #refesing the status
        # self.getGateGui(self.proceduralGate.gateLabel)
        # self.loadWindow(self.window)

        if self.gate_group_selection_rect and self.gate_group_selected:
            self.gate_group_selection_rect.renderWindow()

        # base rect rendering
        self.baseRect.renderWindow()
        # image rendering
        posx, posy = self.proceduralGate.pos

        self.window.blit(self.gateImage, [posx + self.origin[0], posy + self.origin[1]])
        # gui rendering
        for gui in self.gateTerminal_gui:
            gui.renderWindow()

        pass

    def getGateGui(self):

        type_ = self.proceduralGate.gateLabel

        self.gateTerminal_gui = []

        if (type_ == "AND"):
            self.gateImage = pygame.image.load("gateRes/new_gate/and_gate.png")
        if (type_ == "OR"):
            self.gateImage = pygame.image.load("gateRes/new_gate/or_gate.png")
            pass
        if (type_ == "NOT"):
            self.gateImage = pygame.image.load("gateRes/new_gate/not_gate.png")
            pass
        # terminal gates box
        if (type_ == "INP"):
            self.gateImage = pygame.image.load("gateRes/input_gate.png")
        if (type_ == "CLK"):
            self.gateImage = pygame.image.load("gateRes/clock_gate.png")
            pass
        if (type_ == "OUT"):
            self.gateImage = pygame.image.load("gateRes/out_gate.png")
            pass
        # non basics
        if (type_ == "XOR"):
            self.gateImage = pygame.image.load("gateRes/new_gate/xor_gate.png")
            pass
        if type_ == "NAND":
            self.gateImage = pygame.image.load("gateRes/new_gate/nand_gate.png")

        if type_ == "NOR":
            self.gateImage = pygame.image.load("gateRes/new_gate/nor_gate.png")

        if type_ == "X":
            self.gateImage = pygame.image.load("gateRes/x_gate.png")

        w, h = self.gateImage.get_size()
        self.size = self.proceduralGate.size
        self.size = [self.size[0] * self.scale, self.size[1] * self.scale]
        self.size[1] = self.size[0] * (h / w)
        self.gateImage = pygame.transform.scale(self.gateImage, [self.size[0], self.size[1]])

        # here we are making te interactive part of the gate
        posx, posy = self.proceduralGate.pos

        pad = 1
        self.baseRect = gui.interactiveShape(posx + self.origin[0] - pad, posy + self.origin[1] - pad, "rectB",
                                             [self.size[0] + 2 * pad, self.size[1] + 2 * pad, 1],
                                             partial(self.gate_request, "G"))

        if (type_ == "OUT" and self.proceduralGate.outputBit != None):
            self.baseRect.shape = "rect"
            if self.proceduralGate.outputBit:
                self.baseRect.setColor('normal', self.output_gate_color_1, self.output_gate_color_1)
            else:
                self.baseRect.setColor('normal', self.output_gate_color_0, self.output_gate_color_0)
            pass
        elif (type_ == "INP" or type_ == "CLK"):
            self.baseRect.shape = "rect"
            if self.proceduralGate.outputBit:
                self.baseRect.setColor('normal', self.input_gate_color_1, self.input_gate_color_1)
            else:
                self.baseRect.setColor('normal', self.input_gate_color_0, self.input_gate_color_0)
            pass
        else:
            if self.selected:
                self.baseRect.setColor('normal', self.selected_gate_color)
            else:
                self.baseRect.setColor('normal', self.normal_gate_color)

        if self.window:
            self.baseRect.loadWindow(self.window)
            self.baseRect.renderWindow()

        # inputs
        posy += self.terminalSize[1] / 2
        for i in range(self.proceduralGate.inputs + 1):
            if i == self.proceduralGate.inputs:
                # output
                self.gateTerminal_gui.append(gui.interactiveShape(posx + self.size[0] + self.origin[0],
                                                                  self.proceduralGate.pos[1] + self.size[1] / 2 -
                                                                  self.terminalSize[1] / 2 + self.origin[1], "rect",
                                                                  self.terminalSize, partial(self.gate_request, "O")))
            else:
                # single input handling
                if self.proceduralGate.inputs == 1:
                    self.gateTerminal_gui.append(gui.interactiveShape(posx - self.terminalSize[0] + self.origin[0],
                                                                      posy - self.terminalSize[0] + self.size[1] / 2 +
                                                                      self.origin[1], "rect", self.terminalSize,
                                                                      partial(self.gate_request, "I", i)))
                    continue

                # inputs
                self.gateTerminal_gui.append(gui.interactiveShape(posx - self.terminalSize[0] + self.origin[0],
                                                                  posy - self.terminalSize[0] / 2 + self.origin[1],
                                                                  "rect", self.terminalSize,
                                                                  partial(self.gate_request, "I", i)))
                step = (self.size[1] - self.terminalSize[1]) / max((self.proceduralGate.inputs - 1), 1)
                posy += (step)

            self.gateTerminal_gui[i].setColor('normal', [200, 0, 0])

        # loading the window
        # group selction rect
        if self.gate_group_selected:
            posx, posy = self.proceduralGate.pos
            self.gate_group_selection_rect = gui.interactiveShape(posx + self.origin[0] - self.gate_group_selection_gap,
                                                                  posy + self.origin[1] - self.gate_group_selection_gap,
                                                                  "rectB",
                                                                  [
                                                                      self.gateImage.get_width() + 2 * self.gate_group_selection_gap,
                                                                      self.gateImage.get_height() + 2 * self.gate_group_selection_gap,
                                                                      1], lambda: 0)

            self.gate_group_selection_rect.setColor('normal', (0, 0, 0))
