import cv2
import numpy as np
import math
import threading
from BallHoleDetector import *
import copy
import time
import CMDcontrol

class Robot():
    def __init__(self):
        self.detector = BallHoleDetector("model/Leju_8classes_aug_v2.param","model/Leju_8classes_aug_v2.bin") 
        #self.ball_hole_detector = BallHoleDetector("model/Leju_ball_hole_aug.param","model/Leju_ball_hole_aug.bin") 
        
        self.classNames = ["background","boom","sill","door","ball","hole","step","bridge","danger","bridgedd"]
        self.detect_thresholds = [0,0.1,0,0,0.3,0.3,0.1,0.1,0.2,0]
        self.DEBUG = True
        
        self.image = None
        """
        def get_Img(self):
            image_reader = ImgConverter()
            while True:
                _, chest_frame = image_reader.chest_image()
                _, self.head_frame = image_reader.head_image()
                frame = cv2.resize(chest_frame, (320,320))
                frame = np.rot90(frame)
                frame = cv2.blur(frame,(3,3))
                self.image = frame
                time.sleep(0.05)

        th1 = threading.Thread(target=get_Img)
        th1.setDaemon(True)     # 设置为后台线程，这里默认是False，设置为True之后则主线程不用等待子线程
        th1.start()

        """
        self.camera = cv2.VideoCapture(2)
        #self.head = cv2.VideoCapture(0)
        tmp = 0
        while not self.camera.isOpened():
            self.camera = cv2.VideoCapture(tmp)
            tmp += 1 
            if tmp > 6:
                break
        if tmp > 6:
            raise ValueError("video open failed")
        else:
            print(f"Video{tmp - 1} is opened successed")
        self.h,self.w = 320,320

        self.pictures = []
        self.color_dist = {'Lower':np.array([5,8,46]),'Upper':np.array([180,45,180])}
        self.picture_list_pointer = 0
        #self.use_camera = True
        
        self.step_num_60cm = 6
        self.stair_nums = 3
        
        # mations
        
        
        self.headMotions = ['HeadTurnMM', 'HeadTurn140', 'HeadTurn060', 'HeadTurn180', 'HeadTurn020']
        self.headPositions = [0, -30, 30, -60, 60]
        
        self.turnMotionsRight = ['turn001R','turn005R','turn010R','turn90R']
        self.turnMotionsLeft = ['turn001L','turn005L','turn010L','turn90L']
        self.turnDegrees = [6,30,45,90]
        self.min_turnDegree = min(self.turnDegrees)
        
        while True:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.resize(frame, (320,320))
                self.image = frame            
                break
        
                    
        self.init_image_thread()
        self.init_action_thread()
    # detect_object()

    def switch_camera(self, chest = True):
        self.use_camera = False
        if self.camera is not None:
            self.camera.release()

        if chest: 
            self.camera = cv2.VideoCapture(2)
        else:
            self.camera = cv2.VideoCapture(0)

        _, self.image = self.camera.read()
        self.use_camera = True

        #time.sleep(0.5)


    def detect_ball_hole(self):
        self.switch_camera(chest = False)

        ret = self.ball_hole_detector.locate(self.image)

        self.switch_camera(chest = True)
        classNames = ["background","ball", "hole"]

        res = {
            "ball" : [],
            "hole" : [],
            "total" : 0,
        }
        if len(ret) == 0:
            print("No object detected")
            frame = copy.deepcopy(self.image)
        
            cv2.imshow("DEBUG",frame)
            cv2.waitKey(1)
            return res
        else:
            targetNum = len(ret) // 6
            
            for i in range(targetNum):
                label,x1,y1,x2,y2,score = ret[i*6:(i+1)*6]
                if score < self.detect_thresholds[label]:
                    continue

                x1, y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                merge_flag = False

                for i, [X1,Y1,X2,Y2,Score] in enumerate(res[classNames[label]]):
                    max_topleft = [max(x1,X1),max(y1,Y1)]
                    min_bottomright = [min(x2,X2),min(y2,Y2)]
                    if max_topleft[0]< min_bottomright[0] and max_topleft[1] < min_bottomright[1]:
                        res[classNames[label]][i] = [min(x1,X1),min(y1,Y1),max(x2,X2),max(y2,Y2),min(Score,score)]
                        merge_flag = True
                if not merge_flag == True:
                    res[classNames[label]].append([x1,y1,x2,y2,score])
                    res["total"] += 1
            if self.DEBUG:
                frame = copy.deepcopy(self.image)
                for key in res.keys():
                    if key != "total":
                        for x1,y1,x2,y2,score in res[key]:
                            
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),1,1,0)
                            text = key + "%.2f"%(score)
                            label_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                            cv2.putText(frame,text,(x1,y1+label_size[0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))
                cv2.imshow("DEBUG",frame)
                cv2.waitKey(1)
        return res

    def detect_object(self):
        res = {
            "boom" : [],
            "sill" : [],
            "door" : [],
            "ball" : [],
            "hole" : [],
            "step" : [],
            "bridge" : [],
            "danger" : [],
            "bridgedd" : [],
            "total" : 0,
        }

        ret = self.detector.locate(self.image)

        if len(ret) == 0:
            print("No object detected")
            frame = copy.deepcopy(self.image)
        
            cv2.imshow("DEBUG",frame)
            cv2.waitKey(1)
            return res
        else:
            targetNum = len(ret) // 6
            
            for i in range(targetNum):
                label,x1,y1,x2,y2,score = ret[i*6:(i+1)*6]
                if score < self.detect_thresholds[label]:
                    continue

                x1, y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                merge_flag = False

                for i, [X1,Y1,X2,Y2,Score] in enumerate(res[self.classNames[label]]):
                    max_topleft = [max(x1,X1),max(y1,Y1)]
                    min_bottomright = [min(x2,X2),min(y2,Y2)]
                    if max_topleft[0]< min_bottomright[0] and max_topleft[1] < min_bottomright[1]:
                        res[self.classNames[label]][i] = [min(x1,X1),min(y1,Y1),max(x2,X2),max(y2,Y2),min(Score,score)]
                        merge_flag = True
                if not merge_flag == True:
                    res[self.classNames[label]].append([x1,y1,x2,y2,score])
                    res["total"] += 1
            if self.DEBUG:
                frame = copy.deepcopy(self.image)
                for key in res.keys():
                    if key != "total":
                        for x1,y1,x2,y2,score in res[key]:
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),1,1,0)
                            text = key + "%.2f"%(score)
                            label_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                            cv2.putText(frame,text,(x1,y1+label_size[0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))
                cv2.imshow("DEBUG",frame)
                cv2.waitKey(1)
        return res
                            


    # image
    def get_image(self):
        while True:
            time.sleep(0.03)
            
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.resize(frame, (self.h,self.w))
                frame = np.rot90(frame)
                frame = cv2.blur(frame,(3,3))
                self.image = frame
                
            else:
                if not self.camera.isOpened():
                    print("Camera disconnected, trying ")
                    self.camera = cv2.VideoCapture(0)
                    tmp = 1
                    while not self.camera.isOpened():
                        self.camera = cv2.VideoCapture(tmp)
                        tmp += 1
                        if tmp > 6:
                            break
                    if tmp > 6:
                        raise ValueError("Video open failed")
                    else:
                        print("yes")
    
    def init_image_thread(self):
        th1 = threading.Thread(target = self.get_image)
        th1.setDaemon(True)
        th1.start()
        
    
    # motions 
    def thread_move_action(self):
        CMDcontrol.CMD_transfer()
    
    def action(self,action_name):
        print(f'finished {action_name}')
        CMDcontrol.action_append(action_name)
     
    def init_action_thread(self):
        th2 = threading.Thread(target = self.thread_move_action)
        th2.setDaemon(True)
        th2.start()
    
    def StepForward(self,n = 1):
        for _ in range(n):
            self.action("YiBu1")
            time.sleep(1)
    def Forwark(self,n = 1):
        for _ in range(n):
            self.action('Forwalk01')
            time.sleep(1)


    def Forwark_one(self,n = 1):
        
        for _ in range(n):
            self.action("finalwalk")
            time.sleep(1)

    def Forwark_half(self,n = 1):
        
        for _ in range(n):
            self.action("finalwalkhalf")
            time.sleep(1)
    def Xiao(self):
        for _ in range(4):
            self.action('Forwalk00')
        for _ in range(1):
            self.action("finalwalkhalf")
        for _ in range(4):
            self.action('Forwalk00')    
    def TurnLeft(self,n = 1):
        for _ in range(n):    
            self.action("turn001L")
            time.sleep(1)
    def TurnRight(self,n = 1):
        for _ in range(n):    
            self.action("turn001R")
            time.sleep(1)

    def TurnLeftBig(self,n=1):
        for _ in range(n):    
            self.action("turn010L")
            time.sleep(1)
    def TurnRightBig(self,n=1):
        for _ in range(n):    
            self.action("turn010R")
            time.sleep(1)
    def MoveLeft(self,n = 1):
        for _ in range(n):
            self.action("Left02move")
            
    def MoveRight(self,n = 1):
        for _ in range(n):
            self.action("Right02move")
            time.sleep(1)
    def MoveBack(self,n = 1):
        for _ in range(n):
            self.action("HouTui")
            time.sleep(1.5)
    def cross_obstracle(self):
        for _ in range(5):
            self.action("Forwalk00")
        self.action("RollRail")
        time.sleep(1)
        self.MoveBack(5)
        print('crossing obstracle')
        time.sleep(1)
        
    def up_stair(self):
        
        for _ in range(4):
            self.action("Forwalk00")
        for _ in range(2):
            self.action('stj')
            self.action("Forwalk00")

        self.action('stj')
        self.action('HouTui')
        
        for _ in range(2):
            self.action('DownBridge')
            self.action("Forwalk00")
        print('up_stairs')
        time.sleep(1)
    
    def LeftKill(self):
        self.action('LfootShot')
        print("kill")
        time.sleep(1)
    def TurnLeftBig(self):
        self.action('turn010L')
        time.sleep(1)
    def turn(self,degree):
        print(f"turning {degree} degree")

        turnRight = True
        if degree < 0:
            degree = abs(degree)
            turnRight = False
        if degree <= min(self.turnDegrees):
            res = [min(self.turnDegrees)]
        else:
            dp, path = [0] * (degree+1), [0] * (degree+1) # use dynamic programming(DP) to minimize turn steps
            for i in range(1, degree+1):
                minNum = i 
                for c in self.turnDegrees: 
                    if i >= c and minNum > dp[i-c]+1:
                        minNum, path[i] = dp[i-c]+1, i - c
                dp[i] = minNum 
            
            minStep = dp[-1]
            res = []
            while path[degree] != 0:
                res.append(degree-path[degree])
                degree = path[degree]

        for r in res:
            if turnRight:
                act = self.turnMotionsRight[self.turnDegrees.index(r)]
            else:
                act = self.turnMotionsLeft[self.turnDegrees.index(r)]
            self.action(act)
            
            
    def debug(self):
        if not self.DEBUG:
            print("not in debug mode,pleace check the flag of debug mode")
            return 
        while True:
            print(self.detect_object())
            
                
        
if __name__ == "__main__":
    robot = Robot()
    robot.debug()
    #robot.up_stair()
    #print(robot.detect_ball_hole())
    #robot.debug()
            
              
        
