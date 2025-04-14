import random
import numpy as np
from vpython import sphere, vector, rate, scene, arrow, color, mag

class PointEnv:
    def __init__(self, render=False, speed=1.0, threshold=2.0, startPosition=vector(5, 5, 5), targetPosition=vector(30, 30, 30), axisLength=25):
        self.render = render
        self.speed = speed
        self.threshold = threshold
        self.startPosition = startPosition
        self.targetInit = targetPosition 

        self.actionDict = {
            0: vector(self.speed, 0, 0),   # +x
            1: vector(-self.speed, 0, 0),  # -x
            2: vector(0, self.speed, 0),   # +y
            3: vector(0, -self.speed, 0),  # -y
            4: vector(0, 0, self.speed),   # +z
            5: vector(0, 0, -self.speed)   # -z
        }
        self.numActions = len(self.actionDict)
        self.maxSteps = 500 

        if self.render:
            self._initScene(axisLength)
        self.reset()

    def _initScene(self, axisLength):
        scene.title = "3D Point Movement with RL Control"
        scene.width = 800
        scene.height = 600
        scene.background = vector(0.2, 0.2, 0.2)
        self.x_axis = arrow(pos=vector(0, 0, 0), axis=vector(axisLength, 0, 0), color=color.red)
        self.y_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, axisLength, 0), color=color.green)
        self.z_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, axisLength), color=color.blue)
        self.pointSphere = sphere(pos=self.startPosition, radius=2, color=vector(1, 1, 0))
        self.targetSphere = sphere(pos=self.targetInit, radius=2, color=color.red)

    def reset(self):
        if self.render:
            self.pointSphere.pos = self.startPosition
            rate(60)
        self.pointPosition = vector(self.startPosition.x, self.startPosition.y, self.startPosition.z)
        self.targetPosition = vector(np.random.uniform(0, 40),
                                 np.random.uniform(0, 40),
                                 np.random.uniform(0, 40))
        if self.render:
            self.targetSphere.pos = self.targetPosition
        self.steps = 0
        self.prevDistance = mag(self.pointPosition - self.targetPosition)
        return self._getState()

    def _getState(self):
        return np.array([self.pointPosition.x, self.pointPosition.y, self.pointPosition.z,
                         self.targetPosition.x - self.pointPosition.x,
                         self.targetPosition.y - self.pointPosition.y,
                         self.targetPosition.z - self.pointPosition.z], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        move = self.actionDict[action]
        self.pointPosition += move
        if self.render:
            self.pointSphere.pos = self.pointPosition
            rate(120)

        distance = mag(self.pointPosition - self.targetPosition)
        mew = (self.prevDistance - distance) * 10.0
        choto = False

        if distance <= self.threshold:
            mew += 100.0
            choto = True
        elif self.steps >= self.maxSteps:
            mew -= 100.0
            choto = True

        self.prevDistance = distance
        nextState = self._getState()
        return nextState, mew, choto
