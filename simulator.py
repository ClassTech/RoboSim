#!/usr/bin/env python3
"""
Contains the main SubmarineSimulator class.
This class handles Pygame, rendering, physics, and the main game loop.
"""
import math
import random
import time
from typing import Tuple, Optional

import pygame
import numpy as np

from config import *
from world import Gate, PathMarker, SlalomPole, SubmarinePhysicsState
from data_structures import ThrusterCommands, MPU6050Readings, SensorSuite
from ai.submarine import Submarine
from ai.tasks import GateTask, SlalomTask
from ai.tasks.slalom_task import SlalomTaskState


class SubmarineSimulator:
    def __init__(self, submarine_ai: Submarine, width=1200, height=800):
        pygame.init()
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Autonomous Submarine Gate & Slalom Simulator")
        self.clock = pygame.time.Clock()
        self.config = SimulationConfig()
        self.scaleX = width * 0.7 / self.config.worldWidth
        self.scaleY = height * 0.8 / self.config.worldHeight
        self.font = pygame.font.Font(None, 36)
        self.smallFont = pygame.font.Font(None, 24)
        self.cameraSurface = pygame.Surface((320, 240))
        try:
            bg_img = pygame.image.load("image_9c266f.jpg").convert()
            h=480; w=int(bg_img.get_width()*(h/bg_img.get_height()))
            self.camera_background = pygame.transform.scale(bg_img, (w,h))
            self.camera_background_pano = pygame.Surface((w*2,h))
            self.camera_background_pano.blit(self.camera_background,(0,0)); self.camera_background_pano.blit(self.camera_background,(w,0))
        except pygame.error: self.camera_background, self.camera_background_pano = None, None
        self.subMass, self.subInertia = 4.0, 0.15
        self.netBuoyancyForce, self.thrusterMaxForce = -0.4, 0.8
        self.linearDragCoeff, self.angularDragCoeff = 1.5, 0.1
        
        self.submarineAI = submarine_ai
        
        self.resetSimulation()

    def resetSimulation(self):
        self.gate = Gate(x=15.0, center_y=7.5, z=random.uniform(0.6, 1.8))
        self.path_marker = PathMarker(x=self.gate.x+2.0, y=self.gate.center_y, z=self.config.worldDepth-0.2, heading=0)
        self.slalom_poles = []
        ssx, ssy, sst = self.gate.x+8.0, 1.524, 1.524*0.25
        last_y = self.config.worldHeight / 2
        
        sz = self.config.worldDepth - (0.9 + random.uniform(0.3, 0.6))
        
        for i in range(3):
            sx, y = ssx+i*4.0, last_y+random.uniform(-sst,sst) if i>0 else last_y
            y = np.clip(y, sst*2, self.config.worldHeight-(sst*2)); last_y = y
            self.slalom_poles += [
                SlalomPole(x=sx, y=y-ssy, z=sz, color=WHITE), 
                SlalomPole(x=sx, y=y, z=sz, color=RED), 
                SlalomPole(x=sx, y=y+ssy, z=sz, color=WHITE)
            ]
        
        start_heading = random.choice([0, 90, 270])
        self.subPhysics = SubmarinePhysicsState(x=self.gate.x-9.144, y=self.gate.center_y+random.uniform(-0.9144,0.9144), z=0.5, heading=start_heading)
        self.submarineAI.reset()
        self.startTime = time.time()
        self.running, self.paused = True, False
        self.lastThrusterCommands, self.last_imu_readings = ThrusterCommands(), MPU6050Readings()

    def skip_to_slalom(self):
        self.resetSimulation()
        
        slalom_task_instance = next((task for task in self.submarineAI.mission_plan if isinstance(task, SlalomTask)), None)
        slalom_task_index = next((i for i, task in enumerate(self.submarineAI.mission_plan) if isinstance(task, SlalomTask)), None)

        if slalom_task_instance and slalom_task_index is not None:
            self.submarineAI.current_task_index = slalom_task_index
            self.submarineAI.gateCompleted = True
            self.subPhysics.x, self.subPhysics.y, self.subPhysics.z = self.gate.x + 4.0, self.config.worldHeight / 2, slalom_task_instance.target_depth
            self.subPhysics.heading = 0.0
            
            slalom_task_instance.current_state = SlalomTaskState.SEARCHING
            slalom_task_instance.course_axis_heading, slalom_task_instance.scan_target_heading = 0.0, 0.0

    def worldToScreen(self, x, y): return int(x*self.scaleX+50), int((self.config.worldHeight-y)*self.scaleY+50)
    
    def handleInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: self.paused = not self.paused
                elif event.key == pygame.K_r: self.resetSimulation()
                elif event.key == pygame.K_s: self.skip_to_slalom()

    def applyPhysics(self, dt, commands):
        f = [c*self.thrusterMaxForce for c in [commands.h_port_bow,commands.h_starboard_bow,commands.h_port_aft,commands.h_starboard_aft, commands.v_bow, commands.v_aft]]
        ca, sa = math.cos(math.radians(45)), math.sin(math.radians(45))
        
        surge, sway, heave = (f[0]+f[1]+f[2]+f[3])*ca, (f[0]-f[1]-f[2]+f[3])*sa, f[4]+f[5]
        
        yaw_t = (f[0]-f[1]+f[2]-f[3])*ca*(self.config.submarineWidth/2)
        pitch_t = (f[4]-f[5])*(self.config.submarineLength/2)
        h,p = math.radians(self.subPhysics.heading), math.radians(self.subPhysics.pitch)
        ch,sh,cp,sp = math.cos(h),math.sin(h),math.cos(p),math.sin(p)
        fsh, fwv = surge*cp+heave*sp, heave*cp-surge*sp
        fx,fy,fz = fsh*ch-sway*sh, fsh*sh+sway*ch, fwv+self.netBuoyancyForce
        speed = math.hypot(self.subPhysics.velocity_x, self.subPhysics.velocity_y)
        dm = self.linearDragCoeff*speed*speed
        dx,dy,dz = -dm*(self.subPhysics.velocity_x/speed) if speed>0 else 0, -dm*(self.subPhysics.velocity_y/speed) if speed>0 else 0, -self.linearDragCoeff*2*self.subPhysics.velocity_z*abs(self.subPhysics.velocity_z)
        d_yaw, d_pitch = -self.angularDragCoeff*self.subPhysics.angular_velocity_z**2*np.sign(self.subPhysics.angular_velocity_z), -self.angularDragCoeff*self.subPhysics.angular_velocity_y**2*np.sign(self.subPhysics.angular_velocity_y)
        ax,ay,az = (fx+dx)/self.subMass, (fy+dy)/self.subMass, (fz+dz)/self.subMass
        self.last_imu_readings = MPU6050Readings(accel_x=-ax*sh+ay*ch, accel_y=ax*ch+ay*sh, accel_z=az, gyro_z=self.subPhysics.angular_velocity_z)
        self.subPhysics.velocity_x+=ax*dt; self.subPhysics.velocity_y+=ay*dt; self.subPhysics.velocity_z+=az*dt
        self.subPhysics.angular_velocity_z+=((yaw_t+d_yaw)/self.subInertia)*dt
        self.subPhysics.angular_velocity_y+=((pitch_t+d_pitch)/self.subInertia)*dt
        self.subPhysics.x+=self.subPhysics.velocity_x*dt; self.subPhysics.y+=self.subPhysics.velocity_y*dt; self.subPhysics.z+=self.subPhysics.velocity_z*dt
        self.subPhysics.heading=(self.subPhysics.heading+math.degrees(self.subPhysics.angular_velocity_z*dt))%360
        self.subPhysics.pitch=np.clip(self.subPhysics.pitch+math.degrees(self.subPhysics.angular_velocity_y*dt), -90, 90)
        margin=0.5
        self.subPhysics.x=np.clip(self.subPhysics.x,margin,self.config.worldWidth-margin); self.subPhysics.y=np.clip(self.subPhysics.y,margin,self.config.worldHeight-margin); self.subPhysics.z=np.clip(self.subPhysics.z,0.0,self.config.worldDepth-margin)
    
    def project3D(self, world_pos: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
        dx,dy,dz = world_pos[0]-self.subPhysics.x, world_pos[1]-self.subPhysics.y, world_pos[2]-self.subPhysics.z
        h,p = math.radians(-self.subPhysics.heading), math.radians(-self.subPhysics.pitch)
        ch,sh,cp,sp = math.cos(h),math.sin(h),math.cos(p),math.sin(p)
        x_yaw, y_yaw = dx*ch-dy*sh, dx*sh+dy*ch
        cz,cy,cx = x_yaw*cp+dz*sp, x_yaw*sp-dz*cp, y_yaw
        if cz < 0.2: return None
        w,h = self.cameraSurface.get_size()
        f = w/(2*math.tan(math.radians(self.config.cameraFov/2)))
        return int(w/2-f*(cx/cz)), int(h/2-f*(cy/cz)), math.hypot(dx,dy,dz)

    def generateCameraView(self):
        w,h = self.cameraSurface.get_size()
        if self.camera_background_pano:
            bg_w,bg_h = self.camera_background.get_size()
            x_off = (self.subPhysics.heading/360)*bg_w
            y_off = np.clip(((bg_h-h)/2)-(self.subPhysics.pitch*2.0), 0, bg_h-h)
            self.cameraSurface.blit(self.camera_background_pano, (-x_off, -y_off))
        else:
            self.cameraSurface.fill(WATER_COLOR)
            hp = self.project3D((self.subPhysics.x+20, self.subPhysics.y, self.config.worldDepth))
            if hp: pygame.draw.rect(self.cameraSurface, POOL_FLOOR_COLOR, (0,hp[1],w,h))
        
        drawable = []
        # The gate is now always drawn, regardless of sub position
        dw, ph, pz = self.gate.dividerWidth/2, 0.305, self.gate.z+0.1+(0.305/2)
        gp = {"tpt":(self.gate.x,self.gate.topPoleY,self.gate.z),"bpt":(self.gate.x,self.gate.bottomPoleY,self.gate.z),
              "tpb":(self.gate.x,self.gate.topPoleY,self.gate.z+self.gate.poleHeight),"bpb":(self.gate.x,self.gate.bottomPoleY,self.gate.z+self.gate.poleHeight),
              "dtl":(self.gate.x,self.gate.center_y-dw,self.gate.z),"dtr":(self.gate.x,self.gate.center_y+dw,self.gate.z),
              "dbl":(self.gate.x,self.gate.center_y-dw,self.gate.z+self.gate.dividerHeight),"dbr":(self.gate.x,self.gate.center_y+dw,self.gate.z+self.gate.dividerHeight),
              "sp":(self.gate.x,(self.gate.center_y+self.gate.topPoleY)/2,pz),"fp":(self.gate.x,(self.gate.center_y+self.gate.bottomPoleY)/2,pz)}
        proj = {k: self.project3D(v) for k,v in gp.items()}
        
        if proj["tpt"] and proj["bpt"]: drawable.append((proj["tpt"][2],'line',GRAY,proj["tpt"][:2],proj["bpt"][:2],5))
        if proj["bpt"] and proj["bpb"]: p1,p2=proj["bpt"][:2],proj["bpb"][:2]; my=(p1[1]+p2[1])/2; drawable.extend([(proj["bpt"][2],'line',RED,p1,(p1[0],my),8),(proj["bpt"][2],'line',BLACK,(p1[0],my),p2,8)])
        if proj["tpt"] and proj["tpb"]: p1,p2=proj["tpt"][:2],proj["tpb"][:2]; my=(p1[1]+p2[1])/2; drawable.extend([(proj["tpt"][2],'line',BLACK,p1,(p1[0],my),8),(proj["tpt"][2],'line',RED,(p1[0],my),p2,8)])
        d_keys = ["dtl","dtr","dbr","dbl"]
        if all(proj.get(p) for p in d_keys): drawable.append((sum(proj[p][2] for p in d_keys)/4,'polygon',RED,[proj[p][:2] for p in d_keys]))
        if proj["sp"]: x,y,d=proj["sp"]; s=min(120,max(10,int(800/(d+1)*ph))); drawable.append((d,'rect',SHARK_BLUE,(x-s/2,y-s/2,s,s)))
        if proj["fp"]: x,y,d=proj["fp"]; s=min(120,max(10,int(800/(d+1)*ph))); drawable.append((d,'rect',SAWFISH_GREEN,(x-s/2,y-s/2,s,s)))
        
        m_pts = [self.project3D((self.path_marker.x+c[0],self.path_marker.y+c[1],self.path_marker.z+c[2])) for c in [(-0.6,-0.075,0),(0.6,-0.075,0),(0.6,0.075,0),(-0.6,0.075,0)]]
        if all(m_pts): drawable.append((sum(p[2] for p in m_pts)/4, 'polygon', self.path_marker.color, [p[:2] for p in m_pts]))

        for pole in self.slalom_poles:
            tp,bp = self.project3D((pole.x,pole.y,pole.z)), self.project3D((pole.x,pole.y,pole.z+pole.height))
            if tp and bp: drawable.append(((tp[2]+bp[2])/2, 'line', pole.color, tp[:2], bp[:2], 8))
        
        drawable.sort(key=lambda x: x[0], reverse=True)
        for d in drawable:
            if d[1]=='polygon': pygame.draw.polygon(self.cameraSurface,d[2],d[3])
            elif d[1]=='line': pygame.draw.line(self.cameraSurface,d[2],d[3],d[4],d[5])
            elif d[1]=='rect': pygame.draw.rect(self.cameraSurface,d[2],d[3])

    def render(self):
        self.screen.fill(LIGHT_BLUE)
        pygame.draw.rect(self.screen, BLACK, (40,40,int(self.config.worldWidth*self.scaleX+20),int(self.config.worldHeight*self.scaleY+20)), 2)
        gate_color = GREEN if self.submarineAI.gateCompleted else GRAY
        tp,bp,cp=self.worldToScreen(self.gate.x,self.gate.topPoleY),self.worldToScreen(self.gate.x,self.gate.bottomPoleY),self.worldToScreen(self.gate.x,self.gate.center_y)
        pygame.draw.line(self.screen,gate_color,tp,bp,2)
        
        pole_r_top = pygame.Rect(0,0,16,16); pole_r_top.center = tp
        pygame.draw.arc(self.screen,BLACK,pole_r_top,0,math.pi,8)
        pygame.draw.arc(self.screen,RED,pole_r_top,math.pi,2*math.pi,8)
        
        pole_r_bot = pygame.Rect(0,0,16,16); pole_r_bot.center = bp
        pygame.draw.arc(self.screen,RED,pole_r_bot,0,math.pi,8)
        pygame.draw.arc(self.screen,BLACK,pole_r_bot,math.pi,2*math.pi,8)

        pygame.draw.circle(self.screen,RED,cp,6)
        
        marker_rect=pygame.Rect(0,0,self.path_marker.length*self.scaleX, self.path_marker.width*self.scaleY)
        marker_rect.center=self.worldToScreen(self.path_marker.x,self.path_marker.y)
        pygame.draw.rect(self.screen, self.path_marker.color, marker_rect)

        for pole in self.slalom_poles:
            pygame.draw.circle(self.screen, pole.color, self.worldToScreen(pole.x, pole.y), 5)
            pygame.draw.circle(self.screen, BLACK, self.worldToScreen(pole.x, pole.y), 5, 1)

        subPos = self.worldToScreen(self.subPhysics.x, self.subPhysics.y)
        hRad, cos_h, sin_h = math.radians(self.subPhysics.heading), math.cos(math.radians(self.subPhysics.heading)), math.sin(math.radians(self.subPhysics.heading))
        pvc_s = self.config.submarineWidth*self.scaleX/2
        corners = [(-pvc_s,-pvc_s), (pvc_s,-pvc_s), (pvc_s,pvc_s), (-pvc_s,pvc_s)]
        rotated = [(subPos[0]+dx*cos_h-dy*sin_h, subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in corners]
        pygame.draw.polygon(self.screen,YELLOW,rotated,4)
        box_w,box_l=0.127*self.scaleY/2,self.config.submarineLength*self.scaleX/2
        box_corners=[(-box_l,-box_w),(box_l,-box_w),(box_l,box_w),(-box_l,box_w)]
        rotated_box=[(subPos[0]+dx*cos_h-dy*sin_h,subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in box_corners]
        pygame.draw.polygon(self.screen,CONTROL_BOX_GRAY,rotated_box)
        arrow_pts = [(box_l,-box_w),(box_l,box_w),(box_l+0.2*self.scaleX,0)]
        rotated_arrow=[(subPos[0]+dx*cos_h-dy*sin_h, subPos[1]-(dx*sin_h+dy*cos_h)) for dx,dy in arrow_pts]
        pygame.draw.polygon(self.screen, YELLOW, rotated_arrow)
        
        self._renderUi()
        scaled_camera = pygame.transform.scale(self.cameraSurface, (400, 300))
        self.screen.blit(scaled_camera, (self.width-420, 20))
        pygame.draw.rect(self.screen, BLACK, (self.width-420, 20, 400, 300), 2)
        pygame.display.flip()

    def _drawThrusterBar(self, x, y, label, value):
        bar_w, bar_h = 25, 80
        center_y = y + bar_h / 2
        pygame.draw.rect(self.screen, GRAY, (x, y, bar_w, bar_h), 2)
        
        if value > 0:
            pygame.draw.rect(self.screen, GREEN, (x + 1, center_y - value * bar_h / 2, bar_w - 2, value * bar_h / 2))
        elif value < 0:
            pygame.draw.rect(self.screen, RED, (x + 1, center_y, bar_w - 2, abs(value) * bar_h / 2))
        
        pygame.draw.line(self.screen, BLACK, (x, center_y), (x + bar_w, center_y), 1)
        label_surf = self.smallFont.render(label, True, BLACK)
        self.screen.blit(label_surf, (x + bar_w / 2 - label_surf.get_width() / 2, y + bar_h + 5))

    def _renderUi(self):
        y = 20
        title = self.font.render("Autonomous Submarine Simulator", True, BLACK); self.screen.blit(title, (20,y)); y+=40
        speed = math.hypot(self.subPhysics.velocity_x, self.subPhysics.velocity_y)
        task,state = self.submarineAI.get_current_task_name(), self.submarineAI.get_current_state_name()
        stats=[f"Time: {time.time()-self.startTime:.1f}s", f"Task: {task}", f"State: {state}",
               f"Speed: {speed:.2f} m/s", f"Heading: {self.subPhysics.heading:.1f}°", 
               f"Pitch: {self.subPhysics.pitch:.1f}°", f"Depth: {self.subPhysics.z:.2f} m"]
        for s in stats: self.screen.blit(self.smallFont.render(s,True,BLACK),(20,y)); y+=20
        y+=10
        imu = self.last_imu_readings
        imu_stats=["IMU:", f" Accel Y(surge): {imu.accel_y: .2f} m/s²", f" Accel X(sway): {imu.accel_x: .2f} m/s²",
                   f" Accel Z(heave): {imu.accel_z: .2f} m/s²", f" Gyro Z(yaw): {math.degrees(imu.gyro_z): .1f}°/s"]
        for s in imu_stats: self.screen.blit(self.smallFont.render(s,True,BLACK),(20,y)); y+=18
        y = self.height - 80
        controls=["Controls:", "R - Reset", "SPACE - Pause", "S - Skip to Slalom"]
        for c in controls: self.screen.blit(self.smallFont.render(c,True,BLACK),(20,y)); y+=18
        tx,ty = self.width-420,350
        self.screen.blit(self.smallFont.render("Thruster Output:",True,BLACK),(tx,ty)); ty+=25
        tc=self.lastThrusterCommands
        h_labels=[("H-PB",tc.h_port_bow),("H-SB",tc.h_starboard_bow),("H-PA",tc.h_port_aft),("H-SA",tc.h_starboard_aft)]
        for i,(l,v) in enumerate(h_labels): self._drawThrusterBar(tx+i*50,ty,l,v)
        ty+=110
        v_labels=[("V-Bow",tc.v_bow),("V-Aft",tc.v_aft)]
        for i,(l,v) in enumerate(v_labels): self._drawThrusterBar(tx+i*50,ty,l,v)

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            if dt > 0.1: dt = 0.1
            self.handleInput()
            
            if self.paused: 
                self.render()
                continue
            
            self.generateCameraView()
            sensors = SensorSuite(camera_image=self.cameraSurface, depth=self.subPhysics.z, 
                                  heading=self.subPhysics.heading, pitch=self.subPhysics.pitch, 
                                  imu=self.last_imu_readings, x=self.subPhysics.x, y=self.subPhysics.y,
                                  velocity_x=self.subPhysics.velocity_x, velocity_y=self.subPhysics.velocity_y,
                                  angular_velocity_y=self.subPhysics.angular_velocity_y, velocity_z=self.subPhysics.velocity_z)
            
            thrusterCommands, vision_data = self.submarineAI.update(dt, sensors)
            
            if thrusterCommands.pause_simulation:
                self.paused = True
            
            self.lastThrusterCommands = thrusterCommands
            
            for pole in vision_data.potential_poles:
                pygame.draw.rect(self.cameraSurface, ORANGE, (pole['min_x'], pole['min_y'], pole['max_x']-pole['min_x'], pole['max_y']-pole['min_y']), 1)
            if vision_data.gate_is_visible:
                w,h = vision_data.max_x-vision_data.min_x, vision_data.max_y-vision_data.min_y
                pygame.draw.rect(self.cameraSurface, YELLOW, (vision_data.min_x, vision_data.min_y, w, h), 1)
            for pole in vision_data.visible_poles:
                w,h = pole['max_x']-pole['min_x'], pole['max_y']-pole['min_y']
                color = GREEN if pole.get('color') == 'white' else YELLOW
                pygame.draw.rect(self.cameraSurface, color, (pole['min_x'], pole['min_y'], w, h), 1)
            for pole in vision_data.selected_slalom_poles:
                w = pole['max_x'] - pole['min_x']
                h = pole['max_y'] - pole['min_y']
                pygame.draw.rect(self.cameraSurface, MAGENTA, (pole['min_x'], pole['min_y'], w, h), 3)
            # CORRECTED: Add visualization for poles being actively avoided
            for pole in vision_data.avoidance_poles:
                w = pole['max_x'] - pole['min_x']
                h = pole['max_y'] - pole['min_y']
                pygame.draw.rect(self.cameraSurface, ORANGE, (pole['min_x'], pole['min_y'], w, h), 3)
            
            self.applyPhysics(dt, thrusterCommands)
            self.render()
        pygame.quit()