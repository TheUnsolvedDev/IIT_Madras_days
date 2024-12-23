import jax
import jax.numpy as jnp
import numpy as np

from environment import *
from agent import *


class Simulation:
    def __init__(self):
        self.map_size = 5
        self.env = Field(jax.random.PRNGKey(0), size=self.map_size)
        self.history = History(size = self.map_size)

    def simulate(self, trials=100):
        start_action = np.random.choice(np.arange(self.env.num_total_actions))
        action_vector = self.env.convert_action_index_to_map(start_action)
        action = start_action
        uncertainity,theta_hat = self.history.compute_uncertainity_and_reconstruction()
        state = (theta_hat,uncertainity,action_vector)
        print(state)
        
        for i in range(trials):
            reward = self.env.step(action)
            self.history.add(action_vector, reward)
            
            
            next_state = (new_theta_hat,new_uncertainity,new_action_vector)    
            


if __name__ == "__main__":
    sim  = Simulation()
    sim.simulate()
