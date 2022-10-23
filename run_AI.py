from enviroment import CustomEnv
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt
import keyboard

episodes = 1
models_dir = 'models/DQN'
model_path = f"{models_dir}/Bueno_fondo.zip"

env = CustomEnv(205)
model = DQN.load(model_path, env)

for ep in range(episodes):
    obs = env.reset()
    done  = False
    while not done:
        obs, reward, done, info = env.step(model.predict(obs)[0])
        env.render()
        if keyboard.is_pressed("p"):
            break
    print([obs[i] for i in range(1, 5)])
    plt.show()