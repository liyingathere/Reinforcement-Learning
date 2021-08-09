import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as path
import yaml
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

from reth.algorithm.util import calculate_discount_rewards_with_dones
from reth.buffer import DynamicSizeBuffer
from reth.presets.config import get_solver, get_worker, get_trainer

torch.manual_seed(10)
np.random.seed(10)

if __name__ == "__main__":
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    max_ts = config["common"]["max_ts"]
    gamma = config["common"]["gamma"]
    update_interval = config["common"]["batch_size"]
    env = gym.make('Pendulum-v0')
    env.seed(10)

    solver = get_solver(config_path, env = env)
    act_solver = get_solver(config_path, env = env)
    # shared solver
    
    worker = get_worker(config_path, env = env, solver=act_solver)
    trainer = get_trainer(config_path, env = env, solver=solver)
    
    all_ep_r = []
    # worker.s0 = worker.env.reset()
    # s = env.reset()
    act_solver.sync_weights(solver)
    for episode in range(2000):
        buffer_s, buffer_a, buffer_r,buffer_logprob, buffer_snext= [], [], [], [], []
        ep_r = 0
        # import pdb;pdb.set_trace()
        for t in range(update_interval):
            # s = worker.s0
            # import pdb;pdb.set_trace()
            a, logprob = act_solver.act(worker.s0)
            # print(a)
            # s_, r, done, _ = env.step(a)
            s0, a, r, s1, done = worker.step(a)
            buffer_s.append(s0)
            buffer_a.append(a)
            buffer_r.append((r +10) /10) 
            buffer_logprob.append(logprob)
            buffer_snext.append(s1)   
            ep_r += r
            # s = s_

            if (t + 1) % 32 == 0 or t == 200 - 1:
                # import pdb;pdb.set_trace()   
                buffer_s = np.array(buffer_s)
                buffer_a = np.array(buffer_a)
                buffer_r = np.array(buffer_r)
                buffer_logprob = np.array(buffer_logprob)
                # print(len(buffer_s))             
                loss = trainer.step([buffer_s, buffer_a, buffer_r, buffer_logprob,buffer_snext])
                # import pdb;pdb.set_trace()
                act_solver.sync_weights(solver)          
                buffer_s, buffer_a, buffer_r, buffer_logprob,buffer_snext = [], [], [], [],[]

        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1) 
        print('Episode: %i' % episode,"|Ep_reward: %i" % ep_r)
    
    print('r /10' )
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()   



