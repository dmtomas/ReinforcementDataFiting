import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

def Ajustar(b, t):
    return b[0] + np.e**(t * b[1])

def Reward(a, b, action, inicial):
    ans = inicial - (b - a)**2
    if action == 0 and b < a:
        ans *= -1
    elif action == 1 and b > a:
        ans *= -1
    else:
        ans = 0
    return ans


class CustomEnv(gym.Env):
    def __init__(self, N_CHANNELS):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3) # N of discrete actions
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.infty, high=np.infty,
                                            shape=(N_CHANNELS,), dtype=np.float32)

    def reset(self):
        self.done = False
        self.buscado = [0.5, 0.5]
        self.intervalos = [[0, 1], [0, 1]]
        self.tiempo = np.linspace(0, 10, 200)
        self.delta = 0.1

        self.actual = [(self.intervalos[i][1] - self.intervalos[i][0]) / 2 for i in range(len(self.intervalos))]
        self.inicial = [(self.actual[i] - self.buscado[i])** 2 for i in range(len(self.intervalos))]

        self.steps = 0
        for i in range(0, len(self.actual)):
            self.steps += self.actual[i] * 2 / self.delta
        self.variable = 0
        self.resultado = Ajustar(self.actual, self.tiempo)
        self.observation = np.array([self.actual, self.resultado])
        return self.observation # Esta es la observación.


    def step(self, action):
        # Se intenta actualizar la siguiente variable.
        if self.variable < len(self.intervalos) - 1:
            self.variable += 1
        else:
            self.variable = 0
        self.steps -= 1
        if self.steps <= 0:
            self.done = True
        # Acción = 0 -> aumentá la variable.
        # Acción = 1 -> reducí la variable.
        # Acción = 2 -> no hagas nada con la variable.
        if action == 0:
            self.actual[self.variable] += self.delta
        elif action == 1:
            self.actual[self.variable] -= self.delta
        
        self.reward = Reward(self.actual[self.variable], self.buscado[self.variable], action, self.inicial[self.variable])
        self.observation = np.array([self.actual, self.resultado])
        info = {}
        return self.observation, self.reward, self.done, info

    def render(self):
        self.vals = Ajustar(self.actual, self.tiempo)
        plt.clf()
        plt.plot(self.tiempo, self.resultado, label="Buscado")
        plt.plot(self.tiempo, self.vals, label="Resultado Actual")
        plt.legend()
        plt.pause(0.01)
        return 0

if __name__=="__main__":
    enviroment = CustomEnv(202)
    enviroment.reset()
    for i in range(0, 10):
        action = int(input())
        enviroment.step(action)
        enviroment.render()
    plt.show()