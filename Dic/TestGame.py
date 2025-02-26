from envs.stag_hunt.stag_hunt import StagHunt
from envs.lbf.foraging import ForagingEnv
from envs.overcooked.overcookedenv import OvercookedMultiEnv
import numpy as np
import yaml
import random
import os
import torch as th
from controllers import REGISTRY as mac_REGISTRY
from types import SimpleNamespace
from components.episode_buffer import MetaReplayBuffer
from components.transforms import OneHot
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Agent.RandomAgent import RandomAgent

class Test:
    def __init__(self, env_type:str, random=False,):

        self.env_type = env_type
        self.random = random
        if self.env_type == 'PP4a':
            self.env_name ='stag_hunt'
            self.test_yaml = 'test_PP.yaml'
            self.teammate_model_path = f'../saves/PP4a/PP4a_test_models/3/'
        elif self.env_type == 'LBF':
            self.env_name ='lbf'
            self.test_yaml = 'test_LBF.yaml'
            self.teammate_model_path = f'../saves/LBF/LBF_test_models/3/'
            # self.teammate_model_path = f'../saves/LBF/LBF_models/0/'
            # self.teammate_model_path = f'../saves/models/2/'
        elif self.env_type == 'overcooked':
            self.env_name ='overcooked'
            self.test_yaml = 'test_overcooked.yaml'
            self.teammate_model_path = f'../saves/overcooked/overcooked_test_models/3/'
            # self.teammate_model_path = f'../saves/models/3/'
        teammate_list = []
        for file_name in os.listdir(self.teammate_model_path):
            file_path = os.path.join(self.teammate_model_path, file_name)
            teammate_list.append(file_path)
        self.teammate_list = teammate_list

    
    # 定义一个函数来加载yaml文件
    def load_args_from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            args = yaml.safe_load(f)
        return args

    # 定义一个函数来合并两个字典
    def merge_dicts(self, base_dict, custom_dict):
        for key, value in custom_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self.merge_dicts(base_dict[key], value)
            else:
                base_dict[key] = value

    def init_game_setting(self):
        #if self.env_type == "PP4a":
        env_config_path = f'../src/config/envs/{self.env_name}.yaml'
        args_dict = self.load_args_from_yaml(f'../src/config/{self.test_yaml}')
        test_dict = args_dict
        default_dict = self.load_args_from_yaml('../src/config/default.yaml')
        env_args = self.load_args_from_yaml(env_config_path)

        game_args = env_args['env_args']
        game_args['seed'] = random.randint(1, 10000)
        if self.env_type == "PP4a":
            self.episode_limit = game_args['episode_limit']
        if self.env_type == "LBF":
            self.episode_limit = 50
        if self.env_type == "overcooked":
            self.episode_limit = 400

        self.merge_dicts(args_dict, default_dict)
        self.merge_dicts(args_dict, game_args)

        args = SimpleNamespace(**args_dict)

        args.agent_output_type = "q"
        args.device = "cuda"

        # 初始化环境
        if self.env_name =='stag_hunt':
            self.env = StagHunt(**game_args)
            args.n_actions = self.env.n_actions
        if self.env_name == 'lbf':
            self.env = ForagingEnv(**game_args)
            args.n_actions = self.env.n_actions
        if self.env_name == 'overcooked':
            self.env = OvercookedMultiEnv(**game_args)
            args.n_actions = self.env.action_space.n
        self.n_actions = args.n_actions
        env_info = self.env.get_env_info()
        args.n_agents = self.env.n_agents
        # print(args)
        groups = {
            "agents": args.n_agents
        }

        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
        }

        global_groups = {
            "agents": args.n_agents
        }
        self.buffer = MetaReplayBuffer(scheme, global_groups, 1024, env_info["episode_limit"] + 1,
                                    preprocess=preprocess,
                                    device=args.device)
        # 设置 explore agent 控制器
        if self.env_type == 'LBF':
            args.max_food = test_dict['env_args']['max_food']
            args.field_size = test_dict['env_args']['field_size']
            args.sight = test_dict['env_args']['sight']
            args.population_alg = 'vdn'

        self.mac = mac_REGISTRY[args.mac](self.buffer.scheme, groups, args)
        self.new_batch = partial(EpisodeBatch, self.buffer.scheme, groups, 1, self.episode_limit + 1,
                                    preprocess=preprocess, device=args.device)
    
    def test_game(self, test_episodes, agent, K=20):
        # 加载模型
        self.init_game_setting()

        if "filled" in self.buffer.scheme:
            del self.buffer.scheme["filled"]
        
        env = self.env
        episode = 0
        return_list = []
        # 测试实训
        with tqdm(total=len(range(test_episodes)), desc="testing") as pbar:
            for _ in range(test_episodes):
                o_list = []
                a_list = []
                g_list = []
                t_list = []
                # 随机选择一种队友
                teammate_model_idx = random.randint(0, len(self.teammate_list) - 1)
                self.mac.load_models(self.teammate_list[teammate_model_idx])
                self.mac.init_hidden(batch_size=1)
                batch = self.new_batch()
                obs, state = env.reset()
                avail_actions = env.get_avail_actions()
                
                # 初始化第一个时间步的数据
                pre_transition_data = {
                    "state": [[state]],
                    "avail_actions": [[avail_actions]],
                    "obs": [[obs]]
                }
                batch.update(pre_transition_data, ts=0)
                
                done = False
                step_count = 0
                total_reward = 0
                teammate_idx = random.randint(0, self.env.n_agents - 1)

                while not done and step_count < self.episode_limit:
                    if self.env_type == "overcooked":
                        dynamic_env_infos = env.get_dynamic_env_info()
                    
                    # 记录数据
                    o_list.append(obs[teammate_idx])
                    t_list.append(torch.tensor(step_count))
                    
                    # 维持列表长度
                    if len(o_list) > K:
                        o_list = o_list[-K:]
                        t_list = t_list[-K:]
                        a_list = a_list[-K+1:]
                        g_list = g_list[-K+1:]

                    # 队友选择动作
                    if self.env_type == "overcooked":
                        actions_tensor = self.mac.select_actions(batch, bs=[0], t_ep=step_count, t_env=1, test_mode=True, dynamic_env_infos=dynamic_env_infos)
                    else:
                        actions_tensor = self.mac.select_actions(batch, bs=[0], t_ep=step_count, t_env=1, test_mode=True)

                    actions = actions_tensor[0].numpy()

                    # adhoc agent选择动作
                    if self.random:
                        action_ad = agent.take_action()
                    else:
                        action_ad, S = agent.take_action(o_list, a_list, g_list, t_list, self.n_actions)
                        g_list.append(S)

                    # actions[teammate_idx] = action_ad[0]
                    a_list.append(actions[teammate_idx])

                    # 更新动作
                    actions_chosen = {
                        "actions": actions_tensor[0].unsqueeze(1).to("cuda")
                    }
                    batch.update(actions_chosen, bs=[0], ts=step_count, mark_filled=False)

                    # 执行动作
                    reward, done, info = env.step(actions)
                    state = env.get_state()
                    obs = env.get_obs()
                    avail_actions = env.get_avail_actions()

                    # 更新下一个时间步的数据
                    if not done:
                        pre_transition_data = {
                            "state": [[state]],
                            "avail_actions": [[avail_actions]],
                            "obs": [[obs]]
                        }
                        batch.update(pre_transition_data, ts=step_count + 1)

                    # 更新当前时间步的奖励和终止状态
                    post_transition_data = {
                        "reward": [[reward]],
                        "terminated": [[done]]
                    }
                    batch.update(post_transition_data, bs=[0], ts=step_count, mark_filled=False)

                    total_reward += reward
                    step_count += 1

                episode += 1
                print(total_reward)
                return_list.append(total_reward)
                pbar.update(1)
            
        print("Average Return:", sum(return_list)/ len(return_list))
        return sum(return_list)/ len(return_list), np.var(return_list)

    def test_game_dt(self, test_episodes, agent, K=20):
        # 加载模型
        self.init_game_setting()

        if "filled" in self.buffer.scheme:
            del self.buffer.scheme["filled"]
        

        env = self.env
        episode = 0
        return_list = []


        # 测试实训
        while episode < test_episodes:
            o_list = []
            a_list = []
            R_list = []
            t_list = []
            # 随机选择一种队友
            teammate_model_idx = random.randint(0, len(self.teammate_list) - 1)
            self.mac.load_models(self.teammate_list[teammate_model_idx])
            self.mac.init_hidden(batch_size=1)
            batch = self.new_batch()
            obs, state = env.reset()
            done = False
            step_count = 0
            total_reward = 0
            # 随机选择一个个体作为ad hoc agent
            teammate_idx = random.randint(0, self.env.n_agents - 1)
            # print(teammate_idx)
            # 开始游戏循环
            while not done and step_count < self.episode_limit:

                obs = env.get_obs()

                state = env.get_state()
                # print("states:", state)
                # avail_actions = env.get_avail_actions()
                # pre_transition_data["state"].append([state])
                # pre_transition_data["avail_actions"].append([avail_actions])
                # pre_transition_data["obs"].append([obs])
                # batch.update(pre_transition_data, bs=[0], ts=step_count)

                # 记录数据
                o_list.append(obs[teammate_idx])
                t_list.append(torch.tensor(step_count))
                if len(R_list) == 0:
                    if self.env_type == "PP4a":
                        R_list.append(80)
                    if self.env_type == "LBF":
                        R_list.append(0.5)
                    if self.env_type == "overcooked":
                        R_list.append(12)
                # 维持列表长度
                if len(o_list) > K:
                    o_list = o_list[-K:]
                    t_list = t_list[-K:]
                    a_list = a_list[-K+1:]
                    R_list = R_list[-K+1:]

                # 队友选择动作
                actions_tensor = self.mac.select_actions(batch, bs=[0],t_ep=step_count, t_env=step_count, test_mode=True)
                actions = actions_tensor[0].numpy()
                
                # adhoc agent选择动作
                action_ad = agent.take_action(o_list, a_list, R_list, t_list, self.n_actions)
                # action_ad = agent.take_action()
                # action_ad = np.random.randint(0, env.n_actions, size=(1,))
                # 拼接
                actions[teammate_idx] = action_ad[0]
                # actions = np.concatenate([actions[:-1], action_ad])
                # print("action:", actions)
                a_list.append(actions[teammate_idx])

                # 执行动作并获取下一步
                reward, done, info = env.step(actions)
                # print("reward:", reward)
                R_list.append(R_list[-1] - reward)

                post_transition_data = {
                    "actions": [],
                    "reward": [],
                    "terminated": [],
                }
                post_transition_data["actions"].append([actions])
                post_transition_data["reward"].append([reward])
                post_transition_data["terminated"].append([done])
                
                batch.update(post_transition_data, bs=[0], ts=step_count)

                # 累积奖励
                total_reward += reward
                step_count += 1
            episode += 1
            return_list.append(total_reward)
            #print(f"Episode {episode} return: {total_reward}")
        #print("Average Return:", sum(return_list)/ len(return_list))
        return sum(return_list)/ len(return_list), np.var(return_list)


    def test_game_prom_ok(self, test_episodes, agent, K, states_p, actions_p, rtg_p, prompt_len):
        # 加载模型
        self.init_game_setting()

        if "filled" in self.buffer.scheme:
            del self.buffer.scheme["filled"]
        

        env = self.env
        episode = 0
        return_list = []
        states_p = states_p[0, ...].tolist()
        actions_p = actions_p[0, ...].tolist()
        rtg_p = rtg_p[0, ...].tolist()
        t_p = list(range(len(rtg_p)))
        # 测试实训
        while episode < test_episodes:
            o_list = []
            a_list = []
            R_list = []
            t_list = []
            # 随机选择一种队友
            teammate_model_idx = random.randint(0, len(self.teammate_list) - 1)
            self.mac.load_models(self.teammate_list[teammate_model_idx])
            self.mac.init_hidden(batch_size=1)
            batch = self.new_batch()
            obs, state = env.reset()
            done = False
            step_count = 0
            total_reward = 0
            # 随机选择一个个体作为ad hoc agent
            teammate_idx = random.randint(0, self.env.n_agents - 1)
            # 开始游戏循环
            while not done and step_count < self.episode_limit:

                pre_transition_data = {
                    "state": [],
                    "avail_actions": [],
                    "obs": [],
                }

                obs = env.get_obs()
                state = env.get_state()
                avail_actions = env.get_avail_actions()
                pre_transition_data["state"].append([state])
                pre_transition_data["avail_actions"].append([avail_actions])
                pre_transition_data["obs"].append([obs])
                batch.update(pre_transition_data, bs=[0], ts=step_count)

                # 记录数据
                o_list.append(obs[teammate_idx])
                t_list.append(torch.tensor(step_count))
                if len(R_list) == 0:
                    if self.env_type == "PP4a":
                        R_list.append(80)
                    if self.env_type == "LBF":
                        R_list.append(0.5)
                    if self.env_type == "overcooked":
                        R_list.append(12)
                # 维持列表长度
                if len(o_list) > K:
                    o_list = o_list[-K:]
                    t_list = t_list[-K:]
                    a_list = a_list[-K+1:]
                    R_list = R_list[-K+1:]
                o_input = states_p + o_list
                a_input = actions_p + a_list
                r_input = rtg_p + R_list
                t_input = t_p + t_list
                # 队友选择动作
                actions_tensor = self.mac.select_actions(batch, bs=[0],t_ep=step_count, t_env=step_count, test_mode=True)
                actions = actions_tensor[0].numpy()
                
                # adhoc agent选择动作
                action_ad = agent.take_action(states_p, o_list, actions_p, a_list, rtg_p, R_list, t_p, t_list, self.n_actions)
                # action_ad = agent.take_action()
                # action_ad = np.random.randint(0, env.n_actions, size=(1,))
                # 拼接
                actions[teammate_idx] = action_ad[0]
                # actions = np.concatenate([actions[:-1], action_ad])
                
                a_list.append(actions[teammate_idx])

                # 执行动作并获取下一步
                reward, done, info = env.step(actions)

                R_list.append(R_list[-1] - reward)

                post_transition_data = {
                    "actions": [],
                    "reward": [],
                    "terminated": [],
                }
                post_transition_data["actions"].append([actions])
                post_transition_data["reward"].append([reward])
                post_transition_data["terminated"].append([done])
                
                batch.update(post_transition_data, bs=[0], ts=step_count)

                # 累积奖励
                total_reward += reward
                step_count += 1
            episode += 1
            return_list.append(total_reward)
            #print(f"Episode {episode} return: {total_reward}")
        #print("Average Return:", sum(return_list)/ len(return_list))
        return sum(return_list)/ len(return_list), np.var(return_list)
        
    def test_game_prom(self, test_episodes, agent, K, states_p, actions_p, rtg_p, prompt_len):
        # 加载模型
        self.init_game_setting()

        if "filled" in self.buffer.scheme:
            del self.buffer.scheme["filled"]
        

        env = self.env
        episode = 0
        return_list = []
        states_p = states_p[0, ...].tolist()
        actions_p = actions_p[0, ...].tolist()
        rtg_p = rtg_p[0, ...].tolist()
        t_p = list(range(len(rtg_p)))
        # 测试实训
        while episode < test_episodes:
            o_list = []
            a_list = []
            R_list = []
            t_list = []
            # 随机选择一种队友
            teammate_model_idx = random.randint(0, len(self.teammate_list) - 1)
            self.mac.load_models(self.teammate_list[teammate_model_idx])
            self.mac.init_hidden(batch_size=1)
            batch = self.new_batch()
            obs, state = env.reset()
            done = False
            step_count = 0
            total_reward = 0
            # 随机选择一个个体作为ad hoc agent
            teammate_idx = random.randint(0, self.env.n_agents - 1)
            # 开始游戏循环
            while not done and step_count < self.episode_limit:

                pre_transition_data = {
                    "state": [],
                    "avail_actions": [],
                    "obs": [],
                }

                obs = env.get_obs()
                state = env.get_state()
                avail_actions = env.get_avail_actions()
                pre_transition_data["state"].append([state])
                pre_transition_data["avail_actions"].append([avail_actions])
                pre_transition_data["obs"].append([obs])
                batch.update(pre_transition_data, bs=[0], ts=step_count)

                # 记录数据
                o_list.append(obs[teammate_idx])
                t_list.append(torch.tensor(step_count))
                if len(R_list) == 0:
                    if self.env_type == "PP4a":
                        R_list.append(80)
                    if self.env_type == "LBF":
                        R_list.append(0.5)
                    if self.env_type == "overcooked":
                        R_list.append(12)
                # 维持列表长度
                if len(o_list) > K:
                    o_list = o_list[-K:]
                    t_list = t_list[-K:]
                    a_list = a_list[-K+1:]
                    R_list = R_list[-K+1:]
                o_input = states_p + o_list
                a_input = actions_p + a_list
                r_input = rtg_p + R_list
                t_input = t_p + t_list
                # 队友选择动作
                actions_tensor = self.mac.select_actions(batch, bs=[0],t_ep=step_count, t_env=step_count, test_mode=True)
                actions = actions_tensor[0].numpy()
                
                # adhoc agent选择动作
                action_ad = agent.take_action(o_input, a_input, r_input, t_input, self.n_actions)
                # action_ad = agent.take_action()
                # action_ad = np.random.randint(0, env.n_actions, size=(1,))
                # 拼接
                actions[teammate_idx] = action_ad[0]
                # actions = np.concatenate([actions[:-1], action_ad])
                
                a_list.append(actions[teammate_idx])

                # 执行动作并获取下一步
                reward, done, info = env.step(actions)

                R_list.append(R_list[-1] - reward)

                post_transition_data = {
                    "actions": [],
                    "reward": [],
                    "terminated": [],
                }
                post_transition_data["actions"].append([actions])
                post_transition_data["reward"].append([reward])
                post_transition_data["terminated"].append([done])
                
                batch.update(post_transition_data, bs=[0], ts=step_count)

                # 累积奖励
                total_reward += reward
                step_count += 1
            episode += 1
            return_list.append(total_reward)
            #print(f"Episode {episode} return: {total_reward}")
        #print("Average Return:", sum(return_list)/ len(return_list))
        return sum(return_list)/ len(return_list), np.var(return_list)

    def test_game_odits(self, test_episodes, agent):
        # 加载模型
        self.init_game_setting()


        if "filled" in self.buffer.scheme:
            del self.buffer.scheme["filled"]
        

        env = self.env
        episode = 0
        return_list = []
        # 测试实训
        with tqdm(total=len(range(test_episodes)), desc="testing") as pbar:
            for _ in range(test_episodes):

                # 随机选择一种队友
                teammate_model_idx = random.randint(0, len(self.teammate_list) - 1)
                self.mac.load_models(self.teammate_list[teammate_model_idx])
                self.mac.init_hidden(batch_size=1)
                batch = self.new_batch()
                obs, state = env.reset()
                a = np.array(0)

                done = False
                step_count = 0
                total_reward = 0
                # 随机选择一个个体作为ad hoc agent
                teammate_idx = random.randint(0, self.env.n_agents - 1)
                # 开始游戏循环
                h = None
                while not done and step_count <= self.episode_limit:

                    pre_transition_data = {
                        "state": [],
                        "avail_actions": [],
                        "obs": [],
                    }
                    last_obs = obs[teammate_idx]
                    last_action = a
                    obs = env.get_obs()
                    state = env.get_state()
                    avail_actions = env.get_avail_actions()
                    pre_transition_data["state"].append([state])
                    pre_transition_data["avail_actions"].append([avail_actions])
                    pre_transition_data["obs"].append([obs])
                    batch.update(pre_transition_data, bs=[0], ts=step_count, mark_filled=True)

                    # 队友选择动作
                    actions_tensor = self.mac.select_actions(batch, bs=[0],t_ep=step_count, t_env=step_count, test_mode=True)
                    actions = actions_tensor[0].numpy()
                    
                    # adhoc agent选择动作
                    if self.random:
                        action_ad = agent.take_action()
                    else:
                        action_ad, h_new = agent.take_action(last_obs, obs[teammate_idx], last_action, h, self.n_actions)
                        h = h_new
                    # action_ad = agent.take_action()
                    # action_ad = np.random.randint(0, env.n_actions, size=(1,))
                    # 拼接
                    actions[teammate_idx] = action_ad[0]
                    a = action_ad[0]
                    # 执行动作并获取下一步
                    reward, done, info = env.step(actions)
                    actions_chosen = {
                        "actions": torch.tensor(actions).unsqueeze(1)
                    }
                    actions_chosen["actions"] = actions_chosen["actions"].to(batch.device)
                    batch.update(actions_chosen, bs=[0], ts=step_count, mark_filled=False)
                    post_transition_data = {
                        "reward": [],
                        "terminated": [],
                    }
                    #post_transition_data["actions"].append([actions])
                    post_transition_data["reward"].append([reward])
                    post_transition_data["terminated"].append([done])
                    
                    batch.update(post_transition_data, bs=[0], ts=step_count, mark_filled=False)
                    # 累积奖励
                    total_reward += reward
                    step_count += 1
                episode += 1
                return_list.append(total_reward)
                #pbar.set_description(f"Episode {episode} return: {total_reward}")
                #print(f"Episode {episode} return: {total_reward} ", self.teammate_list[teammate_model_idx])
                pbar.update(1)
        print("Average Return:", sum(return_list)/ len(return_list))
        return sum(return_list)/ len(return_list), np.var(return_list)

    def test_game_liam(self, test_episodes, agent):
        # 加载模型
        self.init_game_setting()

        if "filled" in self.buffer.scheme:
            del self.buffer.scheme["filled"]
        
        env = self.env
        episode = 0
        return_list = []
        # 测试实训
        with tqdm(total=len(range(test_episodes)), desc="testing") as pbar:
            for _ in range(test_episodes):

                # 随机选择一种队友
                teammate_model_idx = random.randint(0, len(self.teammate_list) - 1)
                self.mac.load_models(self.teammate_list[teammate_model_idx])
                self.mac.init_hidden(batch_size=1)
                batch = self.new_batch()
                obs, state = env.reset()
                a = np.array(0)

                done = False
                step_count = 0
                total_reward = 0
                # 随机选择一个个体作为ad hoc agent
                teammate_idx = random.randint(0, self.env.n_agents - 1)
                # 开始游戏循环
                h = None
                while not done and step_count <= self.episode_limit:

                    pre_transition_data = {
                        "state": [],
                        "avail_actions": [],
                        "obs": [],
                    }
                    last_obs = obs[teammate_idx]
                    last_action = a
                    obs = env.get_obs()
                    state = env.get_state()
                    avail_actions = env.get_avail_actions()
                    pre_transition_data["state"].append([state])
                    pre_transition_data["avail_actions"].append([avail_actions])
                    pre_transition_data["obs"].append([obs])
                    batch.update(pre_transition_data, bs=[0], ts=step_count, mark_filled=True)

                    # 队友选择动作
                    actions_tensor = self.mac.select_actions(batch, bs=[0],t_ep=step_count, t_env=step_count, test_mode=True)
                    actions = actions_tensor[0].numpy()
                    
                    # adhoc agent选择动作
                    if self.random:
                        action_ad = agent.take_action()
                    else:
                        action_ad, h_new = agent.take_action(obs[teammate_idx], last_action, h, self.n_actions)
                        h = h_new
                    # action_ad = agent.take_action()
                    # action_ad = np.random.randint(0, env.n_actions, size=(1,))
                    # 拼接
                    actions[teammate_idx] = action_ad[0]
                    a = action_ad[0]
                    # 执行动作并获取下一步
                    reward, done, info = env.step(actions)
                    actions_chosen = {
                        "actions": torch.tensor(actions).unsqueeze(1)
                    }
                    actions_chosen["actions"] = actions_chosen["actions"].to(batch.device)
                    batch.update(actions_chosen, bs=[0], ts=step_count, mark_filled=False)
                    post_transition_data = {
                        "reward": [],
                        "terminated": [],
                    }
                    #post_transition_data["actions"].append([actions])
                    post_transition_data["reward"].append([reward])
                    post_transition_data["terminated"].append([done])
                    
                    batch.update(post_transition_data, bs=[0], ts=step_count, mark_filled=False)
                    # 累积奖励
                    total_reward += reward
                    step_count += 1
                episode += 1
                return_list.append(total_reward)
                #pbar.set_description(f"Episode {episode} return: {total_reward}")
                #print(f"Episode {episode} return: {total_reward} ", self.teammate_list[teammate_model_idx])
                pbar.update(1)
        print("Average Return:", sum(return_list)/ len(return_list))
        return sum(return_list)/ len(return_list), np.var(return_list)
    
if __name__ == "__main__":
    test = Test(env_type="overcooked", random=True)
    randomAgent = RandomAgent(n_actions=5)
    test.test_game(5, randomAgent)