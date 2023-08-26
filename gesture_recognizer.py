class GestureRec:
    def __init__(self, video: bool = True):
        # ---------------------------------- Imports ---------------------------------------------
        import cv2
        import time
        import copy
        import torch
        import pickle
        import winsound
        import win32api
        import win32con
        import threading
        import itertools
        import mediapipe as mp
        from collections import Counter, deque
        import pygame

        self.cv2 = cv2
        self.time = time
        self.copy = copy
        self.torch = torch
        self.winsound = winsound
        self.win32api = win32api
        self.win32con = win32con
        self.threading = threading
        self.itertools = itertools
        self.mp = mp
        self.Counter = Counter
        self.deque = deque
        self.pygame = pygame

        # --------------------------- gesture Recognizer ----------------------------------------
        self.mp_hand = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # -------------------- gesture Recognizer classification model --------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Device GPU or CPU
        # self.device = 'cpu'
        self.model = pickle.load(open(f"model/model_{self.device}.pk", "rb")).to(self.device)  # Classification Model

        # ---------------------------- Some constants -------------------------------------------
        self.i = 0  # Iterator
        self.fps = 0  # Frames per Sec.
        self.simu = True
        self.ctrlR = True
        self.shiftR = True
        self.clickL = True
        self.clickR = True
        self.changefps = True
        self.video = video  # Bool value for on/off vidio option
        self.length = 16
        self.center = [0, 0]  # Center of the cursor
        self.smoothening = 10  # mouse movement smoothening constant
        self.sensitivity = 20  # sensitivity of the hand
        self.screen = self.win32api.GetSystemMetrics(0), self.win32api.GetSystemMetrics(1)  # Screen size

    def main(self, flag=None, event=None):
        # -------------------------------- Initialization  -------------------------------------
        self.i = 0
        plocX, plocY = 0, 0  # previous loc.
        clocX, clocY = 0, 0  # current loc.
        self.center = [0, 0]  # center of the  hand (hand position in camera )
        self.sensitivity = 20  # control the width of camera ration and screen
        buffer_output = self.deque(maxlen=self.length)  # buffer of the output classification
        # For webcam input:
        cap = self.cv2.VideoCapture(0)  # video
        cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, 960)  # resizing of the width
        cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 540)  # resizing of the height

        with (self.mp_hand.Hands(
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                max_num_hands=1) as hands):
            # should be active event
            if event:
                self.pygame.event.post(event)

            while cap.isOpened():
                # _________________flag on thread termination______________#
                if flag and flag.is_set():
                    break

                start_time = self.time.time()
                success, image = cap.read()  # Input a frame success(bool) image (matrix)
                if not success:  # if the camera is not present
                    print("ignoring empty camera frame.")
                    continue
                # pass by reference.
                image.flags.writeable = False
                image = self.cv2.flip(image, 1)  # flip the image
                image1 = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
                results = hands.process(image1)  # processing the image for gesture location
                image.flags.writeable = True

                # -------------------------- If it find the hand in frame ----------------------------------
                if results.multi_hand_landmarks:  # Draw the hand annotations on the image.
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        landmark_list = self.calc_landmark_list(image, hand_landmarks)  # landmark list
                        pre_processed_landmark_list = self.pre_process_landmark(
                            landmark_list)  # making the hand landmarks ir-respect of position and size

                        # ------------------------------------- Classification Model -----------------------
                        inp = self.torch.tensor(pre_processed_landmark_list).to(
                            self.device)  # input to classification model
                        y_logit = self.model(inp)  # Logit
                        arg = int(self.torch.argmax(y_logit, dim=0))
                        soft = self.torch.softmax(y_logit, dim=0)  # if you need prediction probability
                        buffer_output.append(arg)  # Buffer
                        most_common = self.Counter(buffer_output).most_common()[0][0]  # output gesture

                        # ----------------------------- Recenter ------------------------------------------
                        if most_common == 6:  # to recenter the mouse location
                            self.center = [landmark_list[5][0], landmark_list[5][1]]  # new center
                            # print("new center : ", self.center)  # print statement for new center
                            X = [landmark_list[i][0] for i in range(len(landmark_list))]
                            self.sensitivity = 1.5 * (max(X) - min(X))  # changing the sensitivity area
                        co_ord = self.cord(float(landmark_list[5][0]),
                                           float(landmark_list[5][1]))  # cord pc screen (0, 1)
                        x = int((self.screen[0] + self.smoothening + self.smoothening - 1) * co_ord[0])  # PC cord
                        y = int((self.screen[1] + self.smoothening + self.smoothening - 1) * co_ord[1])  # PC cord

                        # -----------------------------  Smoothen cursor location -------------------------
                        clocX = int(plocX + (x - plocX) / self.smoothening)
                        clocY = int(plocY + (y - plocY) / self.smoothening)
                        self.win32api.SetCursorPos((clocX, clocY))  # Move the cursor to clocX, clocY
                        plocX, plocY = clocX, clocY  # storing location for next iter
                        print(most_common)
                        if most_common == 2 and self.clickL:  # left click // independent of fps
                            self.threading.Thread(target=self.sound, args=[most_common]).start()
                            self.clickL, self.changefps, self.i = False, False, 0  # Reset the variables
                            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_LEFTUP, 0, 0)
                            print("left click")

                        elif most_common == 3 and self.clickR:
                            self.threading.Thread(target=self.sound, args=[most_common]).start()
                            self.clickR, self.changefps, self.i = False, False, 0  # Reset the variables
                            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
                            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_RIGHTUP, 0, 0)
                            print("Right click")


                        elif most_common == 4 and self.shiftR:  # Add gates right control //independent of fps
                            self.threading.Thread(target=self.sound, args=[most_common]).start()
                            self.shiftR, self.changefps, self.i = False, False, 0  # Reset the variables
                            # self.win32api.keybd_event(self.win32con.VK_SHIFT, 0, 0, 0)  # Press the shift key
                            # self.win32api.mouse_event(self.win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0,
                            #                           0)  # Right-click the mouse
                            # self.win32api.mouse_event(self.win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)
                            # self.win32api.keybd_event(self.win32con.VK_SHIFT, 0, self.win32con.KEYEVENTF_KEYUP,
                            #   0)  # Release the shift key
                            ctrlEvent = self.pygame.event.Event(768, {'unicode': '', 'key': 1073742052, 'mod': 4224,
                                                                      'scancode': 228, 'window': None})
                            self.pygame.event.post(ctrlEvent)
                            print("menu")

                        elif most_common == 5 and self.ctrlR:  # toggle input (ctrl + mouse right click)
                            self.threading.Thread(target=self.sound, args=[most_common]).start()
                            self.ctrlR, self.changefps, self.i = False, False, 0  # Reset the variables
                            # self.win32api.keybd_event(self.win32con.VK_CONTROL, 0, 0, 0)  # Press the Ctrl key
                            # self.win32api.mouse_event(self.win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0,
                            #                           0)  # Right-click the mouse
                            # self.win32api.mouse_event(self.win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)
                            # self.win32api.keybd_event(self.win32con.VK_CONTROL, 0, self.win32con.KEYEVENTF_KEYUP,
                            #   0)  # Release the Ctrl key
                            altEvent = self.pygame.event.Event(768, {'unicode': '', 'key': 1073742054, 'mod': 4672,
                                                                     'scancode': 230, 'window': None})
                            self.pygame.event.post(altEvent)
                            print("toggle")

                        # 6 for recenter
                        elif most_common == 7 and self.simu:  # for simulate right shift
                            self.threading.Thread(target=self.sound, args=[most_common]).start()
                            self.simu, self.changefps, self.i = False, False, 0  # Reset the variables
                            # self.win32api.keybd_event(self.win32con.VK_RSHIFT, 0, 0, 0)  
                            # self.win32api.keybd_event(self.win32con.VK_RSHIFT, 0, self.win32con.KEYEVENTF_KEYUP, 0)
                            shiftEvent = self.pygame.event.Event(768, {'unicode': '', 'key': 1073742053, 'mod': 4098,
                                                                       'scancode': 229, 'window': None})
                            self.pygame.event.post(shiftEvent)
                            print("simulate")
                        else:
                            # print("pass")
                            pass

                        self.mp_drawing.draw_landmarks(image, hand_landmarks,  # Draw landmark
                                                       self.mp_hand.HAND_CONNECTIONS,  # connection
                                                       self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                                       # style
                                                       self.mp_drawing_styles.get_default_hand_connections_style())

                    if self.i > self.fps:  # loc for 1 sec. for single click
                        self.clickL, self.simu, self.shiftR = True, True, True
                        self.ctrlR, self.clickR, self.i = True, True, 0
                    else:
                        self.i += 1

                key = self.cv2.waitKey(10)
                end_time = self.time.time()
                if self.changefps:
                    self.fps = 1 / (end_time - start_time)  # fps
                # if not self.changefps:  # print all fps values
                #     print(1 / (end_time - start_time))
                # print(f"current fps : {self.fps}")  # printing the fps
                self.changefps = True
                if self.video:  # option for the showing the video live
                    self.cv2.imshow("MediaPipe Hands", image)  # show the video
                    if key == ord('q') or key == 27 or self.cv2.getWindowProperty('MediaPipe Hands',
                                                                                  self.cv2.WND_PROP_VISIBLE) < 1:
                        break
                else:
                    self.cv2.destroyAllWindows()
                    if key == ord('q') or key == 27:
                        break

            cap.release()
            self.cv2.destroyAllWindows()

    def calc_landmark_list(self, image, landmarks):  # convert the mediapipe results to landmark points
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_z = landmark.z
            landmark_point.append([landmark_x, landmark_y, landmark_z])

        return landmark_point

    def pre_process_landmark(self, landmark_list):  # Convert to relative coordinates
        temp_landmark_list = self.copy.deepcopy(landmark_list)
        base_x, base_y, base_z = 0, 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
            temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            self.itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list

    def cord(self, cx, cy):  # converting the camera coord to screen range (0,1)
        x, y = cx - self.center[0], cy - self.center[1]
        if x > self.sensitivity:
            x = self.sensitivity
        elif x < - self.sensitivity:
            x = - self.sensitivity
        if y > self.sensitivity:
            y = self.sensitivity
        elif y < - self.sensitivity:
            y = - self.sensitivity
        return x / (2 * self.sensitivity) + 0.5, y / (2 * self.sensitivity) + 0.5

    def isvideo(self, val: bool):
        self.video = val

    def sound(self, num: int = 1):
        if num == 3:
            self.winsound.PlaySound('model/simulate.wav', self.winsound.SND_FILENAME)
        else:
            self.winsound.PlaySound('model/click.wav', self.winsound.SND_FILENAME)


if __name__ == "__main__":
    gus = GestureRec()
    gus.main()
