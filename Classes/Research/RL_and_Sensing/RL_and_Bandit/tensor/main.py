import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import argparse
import wandb

from config import *
from environment import *
from agent import *
from model import *

def get_name(params):
    keys = [key for key in params.keys()]
    values = [params[key] for key in keys]

    name = ''
    for key, val in zip(keys, values):
        name += ''.join([i[0] for i in key.split('_')])+':'+str(val)+'_'
    return name

def plot_image(image,bres_map,real):
    fig,ax = plt.subplots(2,2)
    ax[0,0].imshow(image[0])
    ax[0,1].imshow(image[1])
    ax[1,0].imshow(bres_map)
    ax[1,1].imshow(real)
    plt.show()

class Simulation:
    def __init__(self, map_size=ENV_SIZE,lambda_=LAMBDA,sparsity=SPARSITY,learning_rate = ALPHA,buffer_size = BUFFER_SIZE,batch_size = BATCH_SIZE,max_steps = MAX_EPISODE_STEPS,update_every = UPDATE_EVERY,update_target_every = UPDATE_TARGET_EVERY,wandb_log =False):
        self.map_size = map_size
        self.lambda_ = lambda_
        self.sparsity = sparsity
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.update_every = update_every
        self.update_target_every = update_target_every
        self.wandb_log = wandb_log
        
        self.history = History(map_size=self.map_size, lambda_=self.lambda_)
        self.field = Field(size=self.map_size, sparsity=self.sparsity)
        self.q_agent = QAgent(
            img_size=self.map_size, num_actions=self.field.num_actions,
            learning_rate=self.learning_rate, buffer_size=self.buffer_size, batch_size=self.batch_size)
        self.q_agent.summary()

    def simulate(self, steps=1000):
        init_map = self.field.reset()
        epsilon = EPSILON_START
        
        for game in range(steps+1):
            self.history.reset()
            uncertainity, reconstruction = self.history.compute_uncertainity_reconstruction()
            state = np.array([uncertainity, reconstruction]
                             ).reshape(self.map_size, self.map_size, 2)
            done = False

            for episode in (range(self.max_steps+1)):
                action = self.q_agent.epsilon_greedy_action(state, epsilon=epsilon)
                bres_map, resultant_attenutaion = self.field.step(action)
                self.history.add(bres_map, resultant_attenutaion)
                uncertainity, reconstruction, rank = self.history.compute_uncertainity_reconstruction()
                next_state = np.array([uncertainity, reconstruction]).reshape(
                    ENV_SIZE, ENV_SIZE, 2)
                reward, done, facts = self.field.calculate_reward(
                    episode, action, rank, resultant_attenutaion)
                self.q_agent.replay_buffer.add(state,action,reward,next_state,done)
                
                
                if not episode % self.update_every and len(self.q_agent.replay_buffer) >= self.batch_size:
                    sample_batch = self.q_agent.replay_buffer.sample(self.batch_size)
                    td_update_loss = self.q_agent.td_update(
                        *sample_batch)
                    if self.wandb_log:
                        wandb.log({'loss': td_update_loss})
                        wandb.log({'reward': reward})
                        wandb.log({'step':facts['step']})
                        wandb.log({'manhattan_distance':facts['manhattan_distance']})
                        wandb.log({'rank': facts['rank']})
                        wandb.log({'mse': facts['mse']})
                    else:
                        # plot_image(state.reshape(2,self.map_size,self.map_size),bres_map,init_map)
                        print(f'game [{game}/{steps}]\t episode [{episode}/{self.max_steps}]\t loss [{td_update_loss:.4f}]\t reward [{reward:.4f}]\t step [{facts["step"]}]\t manhattan_distance [{facts["manhattan_distance"]}]\t rank [{facts["rank"]}]\t mse [{facts["mse"]:.4f}]\t epsilon [{epsilon:.4f}]')
                if not episode % self.update_target_every:
                    self.q_agent.update_target_model()
                    epsilon *= EPSILON_DECAY
                    epsilon = max(epsilon, EPSILON_MIN)
                
                if done:
                    break
                state = next_state
                
def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        run.name = get_name(config)
        
        sim = Simulation(
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            update_every=config.update_every,
            update_target_every=config.update_target_every
        )
        sim.simulate()
    
def main(sweep = False):
    if not sweep:
        parser = argparse.ArgumentParser()
        parser.add_argument('-size','--env_size', type=int, default=ENV_SIZE,help='Environment size')
        parser.add_argument('-steps','--steps', type=int, default=1000, help='Number of simulation steps')
        parser.add_argument('-lambda','--lambda_', type=float, default=LAMBDA, help='Lambda value for uncertainty regularization')
        parser.add_argument('-sparsity','--sparsity', type=float, default=SPARSITY, help='Sparsity of the environment')
        parser.add_argument('-lr','--learning_rate', type=float, default=ALPHA, help='Learning rate for the Q-network')
        parser.add_argument('-bs','--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
        parser.add_argument('-buffer','--buffer_size', type=int, default=BUFFER_SIZE, help='Replay buffer size')
        parser.add_argument('-max_steps','--max_steps', type=int, default=MAX_EPISODE_STEPS, help='Maximum number of steps per episode')
        parser.add_argument('-update_every','--update_every', type=int, default=UPDATE_EVERY, help='Frequency of training updates')
        parser.add_argument('-update_target_every','--update_target_every', type=int, default=UPDATE_TARGET_EVERY, help='Frequency of updating the target network')
        args = parser.parse_args()

        simulation = Simulation(
            map_size=args.env_size, lambda_=args.lambda_, sparsity=args.sparsity,
            learning_rate=args.learning_rate, buffer_size=args.buffer_size, batch_size=args.batch_size,
            max_steps=args.max_steps, update_every=args.update_every, update_target_every=args.update_target_every
        )
        simulation.simulate(steps=args.steps)
    else:
        sweep_config = {
            'method': 'bayes',
        }
        
        metric = {
            'name':'loss',
            'goal':'minimize',
        }
        
        sweep_config['metric'] = metric
        parameters_dict = {
            'learning_rate': {
                'values': [1e-4, 2e-4, 5e-4],
            },
            'batch_size': {
                'values': [32, 64, 128],
            },
            'buffer_size': {
                'values': [10000, 20000, 40000],
            },
            'update_every': {
                'values': [5, 10, 20],
            },
            'update_target_every': {
                'values': [50, 100, 200],
            },
        }
        
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project='reinforcement_learning')
        wandb.agent(sweep_id, function=train_sweep, count=10)


if __name__ == '__main__':
    main()
