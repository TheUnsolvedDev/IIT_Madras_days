
from dm_control.locomotion import soccer as dm_soccer
from shimmy import DmControlMultiAgentCompatibilityV0

env = dm_soccer.load(team_size=2)
env = DmControlMultiAgentCompatibilityV0(env, render_mode="human")

observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(actions)
    env.render()
env.close()