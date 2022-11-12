from operator import le
from robot import Robot
import math
import cv2

import numpy as np
import time


class Solver():
    def __init__(self):
        self.robot = Robot()
        self.DEBUG = self.robot.DEBUG
        
        self.h,self.w = self.robot.image.shape[:2] # image shape      

        #-----------------------params-------------#
        self.origin_wh_proportion = {
            "boom" :[],
            "sill" :1.5,
            "door" :1.5,
            "ball" :1.5,
            "hole" :[],
            "step" :1.5,
            "bridge" :1.5,
            "danger" :1.5,
            "bridgedd" :[],
        }
        
        # section 1 danger
        self.pit_min_distance = 70
       
        self.pit_bias = 90
        
        
        # section 2 mine
        self.mine_turn = 1
        self.mine_min_shift_bias = 120
        self.mine_distance_bias = 100
        self.max_obstracle_proportion = 0.4
        self.sill_max_distance = 100
        
        
        # section 3 bridge
        
        self.bridge_distance = 50
        self.bridge_min_center = 150
        self.bridge_max_center = 165
        self.bridge_min_bias = -10
        self.bridge_max_bias = 5
        # section 4 ball
        self.waiting_count = 10
        self.maximum_step_kick = 6
        self.walk_coef = 100
        self.turn_coef = 7
        self.ball_dist = 120
        self.ball_min_bias = -5
        self.ball_max_bias = 5
        self.kick_min_shift_bias = -70
        self.kick_max_shift_bias = -40
        self.kick_max_distance_bias = 90
        self.kick_min_distance_bias = 40
        self.kick_angle_bias = 4
        self.kick_max_turn_bias = 10
        
        # section 5 stairs
        self.step_distance = 100
        self.step_min_bias = -6
        self.step_max_bias = 6
        self.step_min_center = 150
        self.step_max_center = 170        
        
    def get_size(self,x1,y1,x2,y2):
        """
        input x1,y1,x2,y2,return the area of the target
        """

        return (x2-x1) * (y2 - y1)

    def get_prop(self,x1,y1,x2,y2):
    
        return self.get_size(x1,y1,x2,y2) / self.h / self.w
    
    
    def analyze_target(self,target,target_type):
        """
        判断目标在机器人的方位 rho,theta
        :param target : 预测的物体
        :param axis: 预测的朝向
        """
        x1,y1,x2,y2 = target
        width_height_proportion = (x2 - x1)/(y2 - y1)
        print(f"width_height_proportion = {width_height_proportion}")
        origin_proportion = self.origin_wh_proportion[target_type]
        
        center = [(x1+x2)//2,(y1+y2)//2]
        bias = center[0] - self.w//2
        
        if width_height_proportion < origin_proportion:
            if bias > 0:
                turn = self.robot.min_turnDegree
            else:
                turn = -self.robot.min_turnDegree
        else:
            turn = 0
        
        return bias, turn
    
    def select_section(self):
        targets = self.robot.detect_object()
        if targets["total"] == 0:
            print("no target find")
            self.robot.Forwark_one()
        else:
            pass 
    # section 1 (pit)
    def section_pit(self):
        
        #detect_cnt = 0
        
        while True:
            targets = self.robot.detect_object()
            bridge = targets["bridge"]
            pits = targets["danger"]    
            if len(pits) > 0:
                #detect_cnt = 0
                x1,y1,x2,y2 = pits[0][:4]
                distance = self.h - y2
                
                #print(distance)
                bias,turn = self.analyze_target(pits[0][:4],"danger")
                #print("bias:",bias)
                if distance > self.pit_min_distance:
                   self.robot.StepForward()
                   
                if  bias < self.pit_bias and distance < self.pit_min_distance:
                    self.robot.MoveLeft()
                if bias > self.pit_bias:
                    self.robot.TurnRight()
                    self.robot.Forwark_one(1)
                    #self.robot.MoveRight(6)
                    self.robot.TurnRight()
                    #self.robot.Forwark_one()
                    #self.robot.TurnRightBig()
                    #self.robot.Forwark_one()
                    #self.robot.cross_obstracle()
                    break
            if len(pits) == 0 and len(bridge) > 0:
                x1,y1,x2,y2 = bridge[0][:4]
                distance = self.h - y2
                bias,turn = self.analyze_target(bridge[0][:4],"danger")
                #print("bias:",bias)
                if distance > self.pit_min_distance:
                   self.robot.StepForward()
                   
                if  bias < self.pit_bias and distance < self.pit_min_distance:
                    self.robot.MoveLeft()
                    
                  
            #else:
                #self.robot.MoveLeft()

    def section_grass(self):
        time.sleep(3)
        cd = {"yellow": [(17, 70, 70), (34, 255, 255)],  # YELLOW[(25, 30, 40), (34, 255, 255)]
        "black": [(0, 0, 0), (180, 255, 35)]}

        thresSection0 = 0.8
        while True:
            image = self.robot.image[:120,:,:]
            yellow = cv2.inRange(image, cd["yellow"][0],cd["yellow"][1])
            black = cv2.inRange(image, cd["black"][0],cd["black"][1])
            bio = np.array(cv2.bitwise_or(yellow, black), dtype=np.float)
            #cv2.imshow("res", np.hstack([image, cv2.cvtColor(np.array(bio,dtype = np.uint8),cv2.COLOR_GRAY2BGR)]))
            #cv2.waitKey(1)
            bio/=255
            yellow = np.array(yellow, dtype = np.float)/255
            weightY = int(np.sum(np.sum(bio, axis=1) * np.array(range(bio.shape[0]))) / (np.sum(bio) + 1e-8))
            roiArea = bio[weightY-4:weightY+5, :].sum()

            print(bio.sum() * thresSection0 < roiArea - 10, yellow.sum())
            if bio.sum() * thresSection0 < roiArea - 10 or yellow.sum() > 200:
                continue
            break

        self.robot.Forwark_one()
        
    def mine_debug(self):
        targets = self.robot.detect_object()
        obstracles = targets['sill']
        if len(obstracles) > 0:
            obstracles.sort(key = lambda x:x[4],reverse = True)
            bias, turn = self.analyze_target(obstracles[0][:4],"sill")
            distance = self.h - obstracles[0][3]
            print(f"distance = {distance}")
            print(f"bias = {bias},degree = {turn}")
                    
            x1,y1,x2,y2,_ = obstracles[0]
            obstracle_proportion = self.get_prop(x1,y1,x2,y2)
            print(f"obstracle_proportion = {obstracle_proportion}")
    
    def section_mine(self):
        """
        Two Steps:
         Crossing mine zone: move left & right & forward to cross, only turn if has to.
            Adjust, Get over the obstracle, than turn to face the door
        """
        sill_distance = 100000
        cnt = 1
        while True:
            """
            if facing the wrong direction, turn
            else if a mine in the close front, check the obstracle to decide whether to move left or right
            else step forward
            """
            targets = self.robot.detect_object()
            obstracles = targets['sill']
            bridge = targets['bridge']
            if cnt >= 3:
                time.sleep(1)
                cnt = 0
                if len(obstracles) > 0:
                    obstracles.sort(key = lambda x:x[4],reverse = True)
                    bias, turn = self.analyze_target(obstracles[0][:4],"sill")
                    sill_distance = self.h - obstracles[0][3]
                    print(f"bias = {bias},degree = {turn}")
                    if bias > 30:
                        self.robot.TurnRight(4)
                    if bias < -30:
                        self.robot.TurnLeft(4)
                    x1,y1,x2,y2,_ = obstracles[0]
                    obstracle_proportion = self.get_prop(x1,y1,x2,y2)
                    print(f"obstracle_proportion = obstracle_proportion")
                    if abs(turn) > self.robot.min_turnDegree/2:
                        # print("")
                        pass
                else:
                    self.robot.MoveLeft()
            else:
                obstracle_proportion = 0
                bias = 0
            # print(sill_distance)
            # avoiding
            mines = targets["boom"]

            skip_flag = False
            for x1,y1,x2,y2,_ in mines:
                mine_center = [(x1+x2)//2,(y1+y2)//2]
                print("mine_center",abs(mine_center[0] - self.w//2))
                if abs(mine_center[0] - self.w//2) < self.mine_min_shift_bias and abs(mine_center[1] - self.h) < self.mine_distance_bias:
                  if mine_center[0] -self.w // 2 > 0:
                        self.robot.MoveLeft()
                  else:
                        self.robot.MoveRight()
                  skip_flag = True
                  break
            if skip_flag:
                continue
            

            # if close enough to the obstracle ,finetune
            if sill_distance < self.sill_max_distance:
                self.robot.cross_obstracle()
                break
            else:
                self.robot.StepForward()
                cnt += 1
                        
    def section_bridge(self):
       
        while True:     
            targets = self.robot.detect_object()
            obstracles = targets['bridge']
        
            if len(obstracles) == 0:
                self.robot.TurnLeftBig()
            else:
                break
        while True:
            image = self.robot.image
            targets = self.robot.detect_object()
            obstracles = targets['bridge']
            doors = targets['door']
            """
            if len(doors) > 0:
                obstracles.sort(key = lambda x:x[4],reverse = True)
                
                x1,y1,x2,y2,_ = obstracles[0]
                
                door1_center = [(x1+x2)//2,(y1+y2)//2]
                #x1,y1,x2,y2,_ = obstracles[1]
                #door2_center = [(x1+x2)//2,(y1+y2)//2]
                
                print(f"door1 = {door1_center},")
            """
            if len(obstracles) > 0:
                """
                if len(doors) > 1:
                    #doors = doors[:2]

                    center = [(doors[0][0] + doors[0][2])//2, (doors[1][0] + doors[1][2])//2]
                    
                    if center[0] > center[1]:#is there any latency problem?
                        left_door, right_door = doors[1], doors[0]
                    else:
                        left_door, right_door = doors[0], doors[1]

                    left_image = image[left_door[1]:left_door[3], left_door[0]:left_door[2]]
                    right_image = image[right_door[1]:right_door[3], right_door[0]:right_door[2]]

                    left_image = cv2.Canny(left_image, 10,100)
                    right_image = cv2.Canny(right_image, 10,100)

                    
                    left_lines = cv2.HoughLines(left_image,1,np.pi/180,118)
                    right_lines = cv2.HoughLines(right_image,1,np.pi/180,118)"""

                obstracles.sort(key = lambda x:x[4],reverse = True)
                bias,turn =self.analyze_target(obstracles[0][:4],"bridge")
                
                print(f"bias = {bias},degree = {turn}")
                
                x1,y1,x2,y2,_ = obstracles[0]
                
                bridge_center = [(x1+x2)//2,(y1+y2)//2]
                bridge_distance = self.h - y2
                print(f"bridge_center = {bridge_center} distance = {bridge_distance}")    
                if bridge_distance < self.bridge_distance and abs(bias) < 10:
                    self.robot.Forwark_one()
                    self.robot.Forwark(3)
                    break
                else:
                    if bias < self.bridge_min_bias :
                        self.robot.TurnLeft()
                    elif bias > self.bridge_max_bias :
                        self.robot.TurnRight()
                        
                    elif bridge_center[0] < self.bridge_min_center and abs(bias) < self.bridge_max_bias:
                        self.robot.MoveLeft()
                    elif bridge_center[0] > self.bridge_max_center and abs(bias) < self.bridge_max_bias:
                        self.robot.MoveRight()
                    else:
                        self.robot.Forwark(3)     


    # section 3 door 
    def section_door(self):
        while True:     
            targets = self.robot.detect_object()
            obstracles = targets['danger']
        
            if len(obstracles) == 0:
                self.robot.TurnLeftBig()
                time.sleep(1)
            else:
                break
        while True:
            image = self.robot.image
            targets = self.robot.detect_object()
            obstracles = targets['danger']
            doors = targets['door']
            """
            if len(doors) > 0:
                obstracles.sort(key = lambda x:x[4],reverse = True)
                
                x1,y1,x2,y2,_ = obstracles[0]
                
                door1_center = [(x1+x2)//2,(y1+y2)//2]
                #x1,y1,x2,y2,_ = obstracles[1]
                #door2_center = [(x1+x2)//2,(y1+y2)//2]
                
                print(f"door1 = {door1_center},")
            """
            if len(obstracles) > 0:
                """
                if len(doors) > 1:
                    #doors = doors[:2]

                    center = [(doors[0][0] + doors[0][2])//2, (doors[1][0] + doors[1][2])//2]
                    
                    if center[0] > center[1]:#is there any latency problem?
                        left_door, right_door = doors[1], doors[0]
                    else:
                        left_door, right_door = doors[0], doors[1]

                    left_image = image[left_door[1]:left_door[3], left_door[0]:left_door[2]]
                    right_image = image[right_door[1]:right_door[3], right_door[0]:right_door[2]]

                    left_image = cv2.Canny(left_image, 10,100)
                    right_image = cv2.Canny(right_image, 10,100)

                    
                    left_lines = cv2.HoughLines(left_image,1,np.pi/180,118)
                    right_lines = cv2.HoughLines(right_image,1,np.pi/180,118)"""

                obstracles.sort(key = lambda x:x[4],reverse = True)
                bias,turn =self.analyze_target(obstracles[0][:4],"bridge")
                
                print(f"bias = {bias},degree = {turn}")
                
                x1,y1,x2,y2,_ = obstracles[0]
                
                bridge_center = [(x1+x2)//2,(y1+y2)//2]
                bridge_distance = self.h - y2
                print(f"bridge_center = {bridge_center} distance = {bridge_distance}")    
                if bridge_distance < self.bridge_distance and abs(bias) < 10:
                    #self.robot.Forwark_one()
                    #self.robot.Forwark(3)
                    self.section_pit()
                    break
                else:
                    if bias < self.bridge_min_bias :
                        self.robot.TurnLeft()
                    elif bias > self.bridge_max_bias :
                        self.robot.TurnRight()
                        
                    elif bridge_center[0] < self.bridge_min_center and abs(bias) < self.bridge_max_bias:
                        self.robot.MoveLeft()
                    elif bridge_center[0] > self.bridge_max_center and abs(bias) < self.bridge_max_bias:
                        self.robot.MoveRight()
                    else:
                        self.robot.Forwark(3) 
                    #self.robot.TurnL 
                #    self.robot.Turn    
                #    if bridge_distance > self.bridge_distance:
                    
    # section_ball
    def section_ball(self):
        """
        Two Step:
            first setp : get close to ball & hole,than lower head
            second step : fine turn 
        """     
        """
        targets = self.robot.detect_ball_hole()
        while len(targets["ball"]) == 0:
            self.robot.StepForward()
            targets = self.robot.detect_ball_hole()"""
        
        """
        targets = self.robot.detect_object()
        cnt = 0
        print("first_stage: getting close to ball & hole")
        while targets["total"] == 0:
            print("No target detected")
            targets = self.robot.detect_object()
            self.robot.StepForward()
            cnt += 1
            if cnt > self.waiting_count:
                if self.Debug:
                    raise ValueError("NO object")
                else:
                    self.robot.StepForward()
                    break

        while len(targets["ball"]) == 0:
            self.robot.StepForward()
            targets = self.robot.detect_object()

        bottomCenter = (self.w//2,self.h)

        x1,y1,x2,y2,_ = targets["ball"][0]
        ball_center = [(x1+x2)//2,(y1+y2)//2]
        degree = - int(math.atan((ball_center[0] - bottomCenter[0])/(ball_center[1] -bottomCenter[1]))/math.pi*180)

        for _ in range(int(degree/self.turn_coef)):
            if degree > 0:
                self.robot.TurnRight()
            else:
                self.robot.TurnLeft()"""

        self.robot.Forwark_one()
        """
        cnt = 0
        while len(targets["ball"]) < 1 or len(targets["hole"]) < 1:
            cnt += 1
            if cnt < self.maximum_step_kick:
                self.robot.StepForward()
                targets = self.robot.detect_object()
            else:
                for turnDegree in [-45,45]:
                    pass
        
        
        while True:
            targets = self.robot.detect_object()
            if len(targets['ball']) == 0:
                self.robot.StepForward()
            else:
                break"""
        bottomCenter = (self.w//2,self.h)
        step_cnt1 = 0
        while True:
            time.sleep(0.5)
            targets = self.robot.detect_object()
            if len(targets['ball']) == 0 or len(targets['hole']) == 0:
                self.robot.StepForward()
                step_cnt1 +=1
                if step_cnt1 >=4:
                    return
            else:
                
                x1,y1,x2,y2,_ = targets["ball"][0]
                ball_center = [(x1+x2)//2,(y1+y2)//2]
                #ball_size = (x2+y2 - x1 - y1)//2
                x1,y1,x2,y2,_ = targets["hole"][0]
                hole_center = [(x1+x2)//2,(y1+y2)//2]
                dist = math.sqrt((ball_center[0] - bottomCenter[0])**2 + (ball_center[1] - bottomCenter[1])**2)
                degree = - int(math.atan((hole_center[0]- ball_center[0])/(hole_center[1] - ball_center[1]))/math.pi*180)
                
                #bias, turn = self.analyze_target(targets["ball"][0][:4],"ball")
                print(dist)
                #print(bias)
                self.robot.StepForward()
                for _ in range(2):
                    self.robot.MoveLeft()
                """
                for _ in range(int(dist/self.walk_coef)):
                    # self.robot.StepForward()
                    pass"""
                    
                for _ in range(int(degree/self.turn_coef)):
                    self.robot.TurnRight()
                    
                break
                    
                """
                if dist > 150:
                    self.robot.Forwark_one()
                degree = - int(math.atan((hole_center[0]- ball_center[0])/(hole_center[1] - ball_center[1]))/math.pi*180)
                if dist > self.ball_dist:
                    if bias < self.ball_min_bias :
                        self.robot.TurnLeft()
                    elif bias > self.ball_max_bias :
                        self.robot.TurnRight()
                    else:
                        self.robot.StepForward()
                else:
                    break """
                
                # print(f"ball_center = {ball_center}")
                # print(f"distance = {dist},walking{int(dist/self.walk_coef)}")
                # for _ in range(int(dist/self.walk_coef)):
                    #self.robot.StepForward()"""
        bottomCenter = (self.w//2,self.h)

        cnt = 0
                    
        while True:
           time.sleep(0.5)
           targets = self.robot.detect_object()
           #time.sleep(0.2)
           if len(targets["ball"]) < 1: # can't find the ball
               self.robot.MoveBack()
               cnt+=1
               if cnt >=2:
                   break
               continue
           if len(targets["hole"]) < 1: # can't find the hole
               self.robot.TurnRight()
               continue
           x1, y1, x2, y2, _ = targets["ball"][0]
           ball_center = [(x1+x2)//2, (y1+y2)//2]
           ball_size = (x2+y2 - x1-y1) //2

           x1, y1, x2, y2, _ = targets["hole"][0]
           hole_center = [(x1+x2)//2, (y1+y2)//2]
           print(ball_center[0] - bottomCenter[0])
           
           if ball_center[0] - bottomCenter[0] < self.kick_min_shift_bias: # correct left/right bias
                print(f'bias is {ball_center[0] - bottomCenter[0]}')
                self.robot.MoveLeft()
                continue

           if ball_center[0] - bottomCenter[0] > self.kick_max_shift_bias:
                print(f'bias is {ball_center[0] - bottomCenter[0]}')
                self.robot.MoveRight()
                continue

           distance = bottomCenter[1] - ball_center[1] # get closer to the ball
           print(f"distance = {distance}")
           if distance < self.kick_min_distance_bias:
               print(f'distance = {distance}, too close, steping backward')
               self.robot.MoveBack()
               #cnt += 1
               continue

           degree = -int(math.atan((hole_center[0]-ball_center[0])/(hole_center[1]-ball_center[1]))/math.pi*180) - self.kick_angle_bias # corect the angle bias
           print(f"degree = {degree}")
           if degree > self.kick_max_turn_bias :
                self.robot.TurnRight()
                continue
           if degree < -self.kick_max_turn_bias:
                self.robot.TurnLeft()
           
           if distance > self.kick_max_distance_bias:
               print(f'distance = {distance}, steping forward')
               self.robot.StepForward()
               continue
           """
           if distance < self.kick_max_distance_bias:# and distance > self.kick_min_distance_bias:
               self.robot.StepBackward()
               #self.robot.MoveLeft(2)
               #self.robot.TurnRight(5)
               #self.robot.LeftKill()
               #break
           """
           
               
           self.robot.LeftKill()# kiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiick!
           break
           
    
    
    def section_stairs(self):
        while True:     
            targets = self.robot.detect_object()
            obstracles = targets['step']
        
            if len(obstracles) == 0:
                self.robot.TurnLeftBig()
            else:
                break
        while True:
            targets = self.robot.detect_object()
            
            obstracles = targets['step']  
            if len(obstracles) == 0:
                obstracles = targets['sill']
            if len(obstracles) > 0:
                obstracles.sort(key = lambda x:x[4],reverse = True)
                bias,turn =self.analyze_target(obstracles[0][:4],"step")
                
                print(f"bias = {bias},degree = {turn}")     
                x1,y1,x2,y2,_ = obstracles[0]
                
                step_center = [(x1+x2)//2,(y1+y2)//2]
                step_distance = self.h - y2
                print(f"bridge_center = {step_center} distance = {step_distance}")
                if step_distance < self.step_distance and abs(bias) < 10:
                    self.robot.up_stair()
                    break
                else:
                    if bias < self.step_min_bias :
                        self.robot.TurnLeft()
                    elif bias > self.step_max_bias :
                        self.robot.TurnRight()
                        
                    elif step_center[0] < self.step_min_center and abs(bias) < self.step_max_bias:
                        self.robot.MoveLeft()
                    elif step_center[0] > self.step_max_center and abs(bias) < self.step_max_bias:
                        self.robot.MoveRight()
                    elif step_distance > 150 and abs(bias) < self.step_max_bias:
                        self.robot.Forwark_one()
                    else:
                        self.robot.StepForward(3) 
           
           
    # Debug
    def Debug(self):
        if not self.DEBUG:
            print("not in debug mode,pleace check the flag of debug mode")
            return
        while True:
            # self.select_section()
            self.section_grass()
            self.robot.Forwark_one()
            #self.section_bridge()
            self.section_mine()
            self.section_door()
            self.section_ball() # with pit embedded
            self.section_stairs()
            # self.robot.up_stair()
            self.robot.Xiao()
            # self.section_ball_debug()
if __name__ == "__main__":
    solver = Solver()
    
    solver.Debug()
    #solver.section_ball()
    # solver.section_door_debug()
        


