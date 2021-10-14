import gym
import numpy as np
from collections import deque
from HLAgent import HLA
from FEAgent import FEA
import warnings


env=gym.make('CartPole-v1')

def FastEnv(NUM_EPISODES, MAX_TIMESTEPS):
    #create fast operating envirnoment agent
    agent = FEA(env)
    #programme loop for the fast environment agent looping through episodes and time steps.
    for episode in range(NUM_EPISODES):
        #this will reset the environment ready for next episode
        observation = env.reset()
        #Iterating through time steps within an episode
        for t in range(MAX_TIMESTEPS):
            env.render()
            #reads in current environment and randomly,using epsilon, will return an optimum step or random(extort.v.explore)
            action = agent.select_action(observation)
            #temporarily saves environment before next save is made
            prev_obs = observation
            #next step (selected from select_action
            observation, reward, done, info = env.step(action)
            # Keep a store of the agent's experiences, appends variables to memory
            agent.remember(done, action, observation, prev_obs)
            #negative or positive reinforcemt from actions memory, works out rewards and values
            agent.experience_replay(20)
            # epsilon decay
            if done:
                # If the pole has tipped over, end this episode
                print('Episode {} ended after {} timesteps'.format(episode, t+1))
                #print(agent.layers[0].lr)
                break
    

def FastLearn(NUM_EPISODES, MAX_TIMESTEPS):

    #key variables for this agent
    count = 0
    eps = 0.9
    eps_min = 0.01
    decay = 0.995
    average_rewards=[]
        
    #create agent
    agent = HLA(states, actions)
    #for given episodes
    for episodes in range(NUM_EPISODES):
        #reset environment
        current_state = env.reset()
        #reset reward
        total_reward = 0
        for t in range(MAX_TIMESTEPS):
        
        ##while True:
            env.render()
           ## if count == 0:
           ##     action = env.action_space.sample()
           ## else:
            action = agent.action_select(current_state,eps)
            next_state , reward , done , _  = env.step(action)
            #increase global reward accumulated across all iterations
            total_reward+=reward
            #push agent state to memory
            agent.push([current_state, reward, action, next_state, done]) 
            current_state = next_state
            #if enough information gathered to learn
            if count > agent.batch_size:
                agent.learn()
            count+=1
            #calculate new epsilon value if required
            if eps > eps_min:
                eps*=decay
            #if pole fallen
            if done:
                average_rewards.append(total_reward)
                break
        if len(average_rewards) > 100:
            del average_rewards[0] 
        if np.mean(average_rewards)> 195:
            break
        print("Episode: ",episodes,"Reward: ",total_reward,"Average: ", np.mean(average_rewards))
        print("ended after timesteps: ", t+1)
    print('solved after ',episodes,' episodes')

        
choice = 0
print("Please choose your operating preference: ")
print("1. fast environment")
print("2. fast learning")
choice = input("please choose 1 or 2: ")


# Global variables
NUM_EPISODES = 1000
MAX_TIMESTEPS = 1000

actions = env.action_space.n
states = env.observation_space.shape[0]


if (choice == '1'):

    FastEnv(NUM_EPISODES, MAX_TIMESTEPS)

                
elif(choice == '2'):
    
    FastLearn(NUM_EPISODES, MAX_TIMESTEPS)


