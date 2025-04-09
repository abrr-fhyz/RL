import gym
import numpy as np
from vpython import *
import vpython
import random
from gym import spaces

class Snake3DEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.directions = [vpython.vector(1, 0, 0), vpython.vector(-1, 0, 0),
                           vpython.vector(0, 1, 0), vpython.vector(0, -1, 0),
                           vpython.vector(0, 0, 1), vpython.vector(0, 0, -1)]
        self.grid_size = 20
        self.cell_size = 25
        self.max_steps = 1000
        self.snake = None
        self.food = None
        self.direction = None
        self.steps = 0
        self.score = 0
        self.prevDistance = None
        self._render = render
        if render:
            self._initScene()
        self.reset()

    def _initScene(self):
        self.scene = vpython.canvas(width=800, height=800, center=vpython.vector(0,0,0), background=vpython.color.gray(0.2))
        self.background_color = vpython.vector(0.3, 0.3, 0.3)
        self._drawBackgroundGrid()
        self._drawBorder()
        self.snake_render = []
        self.food_object = None
        self.score_label = label(pos=vector(0, self.grid_size*self.cell_size/2 + 40, 0), text='Score: 0', height=20, color=color.white, background=color.black)

    def reset(self):
        self.snake = [vpython.vector(0,0,0)]
        self.food = self._spawnFood()
        self.direction = vpython.vector(1,0,0)
        self.steps = 0
        self.score = 0
        self.prevDistance = vpython.mag(self.snake[0]-self.food)
        if self._render:
            self._initRenderObjects()
        return self._getState()

    def step(self, action):
        if not self._isValidDirection(self.directions[action]):
            action = self._getDefaultAction()
        self.direction = self.directions[action]
        new_head = self.snake[0] + self.direction * self.cell_size
        new_head = self._teleport(new_head)
        self.snake.insert(0, new_head)
        reward, done = self._rewardPolicy()
        self.steps += 1
        if self._render:
            self._renderMovement()
        return self._getState(), reward, done, {}

    def _rewardPolicy(self):
        currentDistance = vpython.mag(self.snake[0]-self.food)
        reward = (self.prevDistance - currentDistance) * 10 - 0.1
        done = False
        if self._checkFoodCollision():
            reward += 100
            self.score += 1
            self.food = self._spawnFood()
        else:
            self.snake.pop()
        if self._checkSelfCollision() or self.steps >= self.max_steps:
            reward -= 100
            done = True
        self.prevDistance = vpython.mag(self.snake[0]-self.food)
        return reward, done

    def _getState(self):
        head = self.snake[0]
        dx = head.x - self.food.x
        dy = head.y - self.food.y
        dz = head.z - self.food.z
        return np.array([dx, dy, dz], dtype=np.float32)

    def _spawnFood(self):
        return vpython.vector(random.randint(-10,10)*self.cell_size,
                              random.randint(-10,10)*self.cell_size,
                              random.randint(-10,10)*self.cell_size)

    def _teleport(self, pos):
        bound = (self.grid_size * self.cell_size) / 2
        if pos.x > bound:
            pos.x = -bound
        elif pos.x < -bound:
            pos.x = bound
        if pos.y > bound:
            pos.y = -bound
        elif pos.y < -bound:
            pos.y = bound
        if pos.z > bound:
            pos.z = -bound
        elif pos.z < -bound:
            pos.z = bound
        return pos

    def _checkFoodCollision(self):
        return vpython.mag(self.snake[0]-self.food) < self.cell_size

    def _checkSelfCollision(self):
        return any(vpython.mag(self.snake[0]-s) < self.cell_size*0.9 for s in self.snake[1:])

    def _isValidDirection(self, new_direction):
        return vpython.mag(self.direction+new_direction) > 1

    def _getDefaultAction(self):
        for i, d in enumerate(self.directions):
            if vpython.mag(self.direction-d) < 1e-3:
                return i
        return 0

    def _drawBackgroundGrid(self):
        for x in range(-self.grid_size//2, self.grid_size//2+1):
            for y in range(-self.grid_size//2, self.grid_size//2+1):
                for z in range(-self.grid_size//2, self.grid_size//2+1):
                    vpython.box(pos=vpython.vector(x*self.cell_size, y*self.cell_size, z*self.cell_size),
                                size=vpython.vector(self.cell_size-2, self.cell_size-2, self.cell_size-2),
                                color=self.background_color, opacity=0.1)

    def _drawBorder(self):
        border_thickness = 5
        border_color = vpython.vector(0.8,0.8,0.8)
        for axis in [vpython.vector(1,0,0), vpython.vector(0,1,0), vpython.vector(0,0,1)]:
            for sign in [-1,1]:
                vpython.box(pos=sign*axis*(self.grid_size*self.cell_size/2+border_thickness/2),
                            size=vpython.vector(self.grid_size*self.cell_size+border_thickness*2, border_thickness, border_thickness),
                            color=border_color, opacity=0.5)

    def _initRenderObjects(self):
        for obj in getattr(self, 'snake_render', []):
            obj.visible = False
        self.snake_render = []
        for i, pos in enumerate(self.snake):
            col = vpython.color.green if i == 0 else vpython.color.cyan
            seg = vpython.sphere(pos=pos, radius=self.cell_size/2, color=col)
            self.snake_render.append(seg)
        if self.food_object:
            self.food_object.pos = self.food
        else:
            self.food_object = vpython.sphere(pos=self.food, radius=self.cell_size/2, color=vpython.color.red, emissive=True)

    def _renderMovement(self):
        self.score_label.text = f'Score: {self.score}'
        if not hasattr(self, 'snake_render') or len(self.snake_render) != len(self.snake):
            self._initRenderObjects()
            return

        teleport_threshold = self.cell_size * 3

        interpolation_steps = 10 
        old_positions = [seg.pos for seg in self.snake_render]

        for i in range(1, interpolation_steps+1):
            t = i / interpolation_steps
            vpython.rate(30)
            for j, seg in enumerate(self.snake_render):
                start_pos = old_positions[j]
                target_pos = self.snake[j]
                if vpython.mag(target_pos - start_pos) > teleport_threshold:
                    seg.pos = target_pos
                else:
                    seg.pos = vpython.vector(start_pos.x + (target_pos.x - start_pos.x)*t,
                                             start_pos.y + (target_pos.y - start_pos.y)*t,
                                             start_pos.z + (target_pos.z - start_pos.z)*t)

    def close(self):
        if self._render:
            self.scene.delete()