# Generate a mspacman dataset by running through action decisions in
#	the openai gym MsPacman emulator.  This uses a DQN agent from
#	Sherjil Ozair that can be found https://github.com/sherjilozair

import gym
env_name = "MsPacman-v0"
env = gym.make(env_name)

agent = Agent(state_size=env.observation_space.shape,
              number_of_actions=env.action_space.n,
              save_name=env_name)

observation = env.reset()

gen_iters = 10000

for i in range(0, gen_iters):
    observation = env.reset()
    done = False
    agent.new_episode()
    total_cost = 0.0
    total_reward = 0.0
    frame = 0
    while not done:
        frame += 1
        action, values = agent.act(observation)
        #action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
