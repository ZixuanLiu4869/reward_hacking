import faulthandler
import json
import logging
import os
import signal
import tempfile
from datetime import datetime
from typing import Any, Dict
from scipy.optimize import root, least_squares, minimize
import math

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.typing import PolicyID
from ray.tune.registry import register_env
from sacred import SETTINGS, Experiment

from occupancy_measures.models.model_with_discriminator import ModelWithDiscriminator

from ..agents.orpo import ORPO, chi2_divergence
from ..utils.os_utils import available_cpu_count
from ..utils.training_utils import load_algorithm, load_algorithm_config
import pickle

os.environ["DISPLAY"] = ":99"

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
CURRENT_POLICY_ID = "current"
SAFE_POLICY_ID = "safe_policy0"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


faulthandler.register(signal.SIGUSR1)


ex = Experiment("traffic_eval", save_git_info=False)
logger = logging.getLogger(__name__)


@ex.config
def sacred_config(_log):
    num_cpus = available_cpu_count()
    config_updates = {}
    run = "PPO"  # noqa: F841
    generate_histogram = False  # noqa: F841
    checkpoint = ""  # noqa: F841
    episodes = 30  # noqa: F841
    evaluation_duration_unit = "episodes"  # noqa: F841
    experiment_name = ""  # noqa: F841
    policy_ids: list = [CURRENT_POLICY_ID]  # noqa: F841
    hist_x_label: str = ""  # noqa: F841
    seed = 17
    render = (
        False  # only possible for the traffic and tomato environments at the moment
    )
    render_dir_name = "tomato_rendering"

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(  # noqa: F841
        os.path.dirname(checkpoint), f"rollouts_{experiment_name}_{time_str}"
    )

    # For traffic environment eval, run with xvfb-run if running on a server
    original_config = load_algorithm_config(checkpoint)


    if original_config["env_config"].get("flow_params") is not None:
        from flow.utils.registry import make_create_env
        from flow.utils.rllib import FlowParamsEncoder, get_flow_params

        original_flow_params = get_flow_params(original_config.__dict__)
        original_reward_spec = original_config["env_config"]["reward_specification"]
        original_reward_fun = original_config["env_config"]["reward_fun"]
        original_reward_fun = "proxy"
        
        #original_reward_spec["true"][0]=('local_first',1)
        #original_reward_spec["true"][1]=("accel",0)
        #original_reward_spec["true"][2]=("headway",0)
        print(original_reward_spec)
        

        if render:
            original_sim = original_flow_params["sim"]
            render_mode = "drgb"
            original_sim.render = render_mode
            original_sim.restart_instance = True
            original_sim.save_render = True

            original_sim.force_color_updates = True

            original_flow_params["sim"] = original_sim

        flow_json = json.dumps(
            original_flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4
        )

        create_env, env_name = make_create_env(
            params=original_flow_params,
            reward_specification=original_reward_spec,
            reward_fun=original_reward_fun,
            path=out_dir,
        )
        register_env(env_name, make_multi_agent(create_env))
        config_updates["env"] = env_name
        config_updates["env_config"] = {"flow_params": flow_json}
    elif "tomato" in original_config["env"] and render:
        config_updates["env_config"] = original_config["env_config"]
        config_updates["env_config"]["rendering_filepath"] = os.path.join(
            os.path.dirname(checkpoint), f"{render_dir_name}_{time_str}"
        )

        config_updates["env_config"]["render_mode"] = "rgb_array"
    is_multiagent = False
    per_policy_config_updates: Dict[PolicyID, Any] = {
        policy_id: {} for policy_id in policy_ids
    }
    if is_multiagent:
        for policy_id in policy_ids:
            per_policy_config_updates[policy_id].setdefault("multiagent", {})
            policy_mapping_fn = (
                lambda agent_id, *args, policy_id=policy_id, **kwargs: policy_id
            )
            per_policy_config_updates[policy_id]["multiagent"][
                "policy_mapping_fn"
            ] = policy_mapping_fn
            per_policy_config_updates[policy_id][
                "policy_mapping_fn"
            ] = policy_mapping_fn
            per_policy_config_updates[policy_id]["policies_to_train"] = [policy_id]
    num_workers = 1
    # Could use dataset
    train_batch_size = original_config["train_batch_size"]
    num_envs_per_worker = original_config["num_envs_per_worker"]
    rollout_fragment_length = original_config["rollout_fragment_length"]

    updates = {  # noqa: F841
        "_enable_rl_module_api": False,
        "_enable_learner_api": False,
        "enable_connectors": False,
        "seed": seed,
        "evaluation_num_workers": num_workers,
        "create_env_on_driver": True,
        "evaluation_duration": (
            episodes * original_config["env_config"].get("horizon", 1)
            if evaluation_duration_unit == "timesteps"
            else episodes
        ),
        "evaluation_duration_unit": evaluation_duration_unit,
        "evaluation_config": {},
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "disable_env_checking": True,
        "evaluation_sample_timeout_s": 300,
        "output": out_dir,
        "train_batch_size": train_batch_size,
        "num_rollout_workers": 0,
        "num_envs_per_worker": num_envs_per_worker,
        "rollout_fragment_length": rollout_fragment_length,
    }

    # for PPO and ORPO policies
    if "num_sgd_iter" in original_config:
        updates["num_sgd_iter"] = original_config["num_sgd_iter"]
    if "sgd_minibatch_size" in original_config:
        updates["sgd_minibatch_size"] = original_config["sgd_minibatch_size"]

    config_updates.update(updates)

    for policy_id in policy_ids:
        per_policy_config_updates[policy_id].update(
            {
                **config_updates,
                "output": f"{out_dir}_{policy_id}",
            }
        )

    extra_config_updates = {}  # noqa: F841

    use_local_mode = False
    ray_init_kwargs = {"local_mode": use_local_mode, "num_cpus": num_cpus}  # noqa: F841
'''
def save_trajectories_to_json(sample_batches, base_filename, orpo_filename):
    """
    Save trajectories (obs and actions) to a JSON file.

    Args:
        sample_batches (list): List of SampleBatch objects.
        filename (str): Path to save the JSON file.
    """
    base_trajectories = []
    orpo_trajectories = []
    for sample_batch in sample_batches:
        
        #obs_list = sample_batch["obs"]
        #actions_list = sample_batch["actions"]
        #rewards_list = sample_batch["rewards"]
        #print(obs_list.shape)
        #base_trajectories.append({"obs": obs_list.tolist(), "actions": actions_list.tolist(), "rewards": rewards_list.tolist()})
        
        try:
            obs_list = sample_batch["safe_policy0"]["obs"][:-1]
            actions_list = sample_batch["safe_policy0"]["actions"][:-1]
            info_list = sample_batch["safe_policy0"]["infos"][1:]
            rewards_list = []
            commute_list = []
            accel_list = []
            headway_list = []
            political_list = []
            for info in info_list:
                try:
                    rewards_list.append(info["proxy_rew"])
                    commute_list.append(info["rew_breakdown"]["InfectionSummaryAbsoluteReward"] )
                    accel_list.append(info["rew_breakdown"]["LowerStageReward"])
                    headway_list.append(info["rew_breakdown"]["SmoothStageChangesReward"])
                    political_list.append(info["true_rew_breakdown"]["PoliticalReward"])
                except Exception as e:
                    print(e)
                    rewards_list.append(0)
            base_trajectories.append({"obs": obs_list.tolist(), "actions": actions_list.tolist(), "rewards": rewards_list, "commute":commute_list, "accel": accel_list, "headway": headway_list, "political": political_list})
        except Exception as e:
            obs_list = sample_batch["current"]["obs"][:-1]
            actions_list = sample_batch["current"]["actions"][:-1]
            info_list = sample_batch["current"]["infos"][1:]
            rewards_list = []
            commute_list = []
            accel_list = []
            headway_list = []
            political_list = []
            for info in info_list:
                try:
                    rewards_list.append(info["proxy_rew"])
                    commute_list.append(info["rew_breakdown"]["InfectionSummaryAbsoluteReward"] )
                    accel_list.append(info["rew_breakdown"]["LowerStageReward"])
                    headway_list.append(info["rew_breakdown"]["SmoothStageChangesReward"])
                    political_list.append(info["true_rew_breakdown"]["PoliticalReward"])
                except Exception as e:
                    print(e)
                    rewards_list.append(0)
            orpo_trajectories.append({"obs": obs_list.tolist(), "actions": actions_list.tolist(), "rewards": rewards_list, "commute":commute_list, "accel": accel_list, "headway": headway_list, "political": political_list})
    
    base_reward = []
    base_commute = []
    base_accel = []
    base_headway = []
    base_political = []
    orpo_reward = []
    orpo_commute = []
    orpo_accel = []
    orpo_headway = []
    orpo_political = []

    for i in range(len(base_trajectories)):
        base_reward.append(sum(base_trajectories[i]["rewards"]))
        base_commute.append(sum(base_trajectories[i]["commute"]))
        base_accel.append(sum(base_trajectories[i]["accel"]))
        base_headway.append(sum(base_trajectories[i]["headway"]))
        base_political.append(sum(base_trajectories[i]["political"]))
    for i in range(len(orpo_trajectories)):
        orpo_reward.append(sum(orpo_trajectories[i]["rewards"]))
        orpo_commute.append(sum(orpo_trajectories[i]["commute"]))
        orpo_accel.append(sum(orpo_trajectories[i]["accel"]))
        orpo_headway.append(sum(orpo_trajectories[i]["headway"]))
        orpo_political.append(sum(orpo_trajectories[i]["political"]))

    print("base reward", np.mean(base_reward))
    print("base commute", np.mean(base_commute))
    print("base accel", np.mean(base_accel))
    print("base headway", np.mean(base_headway))
    print("base political", np.mean(base_political))
    print("orpo reward", np.mean(orpo_reward))
    print("orpo commute", np.mean(orpo_commute))
    print("orpo accel", np.mean(orpo_accel))
    print("orpo headway", np.mean(orpo_headway))
    print("orpo political", np.mean(orpo_political))
    

    #print(len(base_trajectories))
    #with open(base_filename, "w") as f:
        #json.dump(base_trajectories, f, indent=4)
    #with open(orpo_filename, "w") as f:
        #json.dump(orpo_trajectories, f, indent=4)
    print(f"Trajectories saved")
    '''

def save_trajectories_to_json(sample_batches, base_filename, orpo_filename):
    """
    Save trajectories (obs and actions) to a JSON file.

    Args:
        sample_batches (list): List of SampleBatch objects.
        filename (str): Path to save the JSON file.
    """
    base_trajectories = []
    orpo_trajectories = []
    for sample_batch in sample_batches:
        '''
        obs_list = sample_batch["obs"]
        actions_list = sample_batch["actions"]
        rewards_list = sample_batch["rewards"]
        #print(obs_list.shape)
        base_trajectories.append({"obs": obs_list.tolist(), "actions": actions_list.tolist(), "rewards": rewards_list.tolist()})
        '''
        try:
            obs_list = sample_batch["safe_policy0"]["obs"]
            actions_list = sample_batch["safe_policy0"]["actions"]
            rewards_list = sample_batch["safe_policy0"]["rewards"]
            base_trajectories.append({"obs": obs_list.tolist(), "actions": actions_list.tolist(), "rewards": rewards_list.tolist()})
        except Exception as e:
            obs_list = sample_batch["current"]["obs"]
            actions_list = sample_batch["current"]["actions"]
            rewards_list = sample_batch["current"]["rewards"]
            orpo_trajectories.append({"obs": obs_list.tolist(), "actions": actions_list.tolist(), "rewards": rewards_list.tolist()})
    
    #print(len(base_trajectories))
    #with open(base_filename, "w") as f:
        #json.dump(base_trajectories, f, indent=4)
    with open(orpo_filename, "w") as f:
        json.dump(orpo_trajectories, f, indent=4)
    print(f"Trajectories saved")

def _chi2_discriminator_rewards(
        discriminator_policy_scores
    ):
    rewards = (discriminator_policy_scores[:, 0].detach().exp() - 1).cpu().numpy()
    return rewards

def _sqrt_chi2_discriminator_rewards(
        discriminator_list
    ):
        chi2_reward_list = []
        for dis in discriminator_list:
            chi2_rewards = _chi2_discriminator_rewards(dis)
            chi2_reward_list.append(chi2_rewards)

        # Step 1: Flatten all values into a single array
        all_chi2_rewards = np.concatenate(chi2_reward_list)

        # Step 2: Sort the array
        sorted_rewards = np.sort(all_chi2_rewards)

        # Step 3: Trim the lowest and highest 1%
        n = len(sorted_rewards)
        trim = int(0.01 * n)
        trimmed_rewards = sorted_rewards[trim:-trim] if trim > 0 else sorted_rewards

        # Step 4: Calculate the mean
        occupancy_measure_chi2_trimmed = np.mean(trimmed_rewards)
        print("occupancy_measure_chi2_trimmed", occupancy_measure_chi2_trimmed)
        
        rewards = []
        for i in range(len(chi2_reward_list)):
            if occupancy_measure_chi2_trimmed <= 0:
                rewards.append(chi2_reward_list[i])
            else:
                rewards.append(chi2_reward_list[i]  / np.sqrt(occupancy_measure_chi2_trimmed))

        return rewards

def _sqrt_chi2_kai_rewards(
    discriminator_list, proxy_reward_list, mean_reward, square_reward
):
    chi2_reward_list = []
    for dis in discriminator_list:
        chi2_rewards = _chi2_discriminator_rewards(dis)
        chi2_reward_list.append(chi2_rewards)

    # Step 1: Flatten all values into a single array
    all_chi2_rewards = np.concatenate(chi2_reward_list)

    # Step 2: Sort the array
    sorted_rewards = np.sort(all_chi2_rewards)

    # Step 3: Trim the lowest and highest 1%
    n = len(sorted_rewards)
    trim = int(0.01 * n)
    trimmed_rewards = sorted_rewards[trim:-trim] if trim > 0 else sorted_rewards

    # Step 4: Calculate the mean
    occupancy_measure_chi2_trimmed = np.mean(trimmed_rewards)
    print("occupancy_measure_chi2_trimmed", occupancy_measure_chi2_trimmed)
    
    rewards = []
    for i in range(len(chi2_reward_list)):
        if occupancy_measure_chi2_trimmed <= 0:
            rewards.append((chi2_reward_list[i] - mean_reward * proxy_reward_list[i]))
        elif occupancy_measure_chi2_trimmed - square_reward < 0:
            rewards.append((chi2_reward_list[i] - mean_reward * proxy_reward_list[i]) / np.sqrt(occupancy_measure_chi2_trimmed))
        else:
            rewards.append((chi2_reward_list[i] - mean_reward * proxy_reward_list[i]) / np.sqrt(occupancy_measure_chi2_trimmed-square_reward))

    return rewards

def caculate_worse_reward(
        reward_list, discriminator_list
    ):
    gamma = 0.99

    #safe_batch = train_batch.policy_batches[policy_id]

    #episode_batches_safe = safe_batch.split_by_episode()
    
    discounted_sum = lambda rewards, gamma: np.sum(np.array(rewards) * np.power(gamma, np.arange(len(rewards))))
    
    # This is pi/pi_ref
    chi2_reward_reverse_list = []
    for dis in discriminator_list:
        chi2_rewards = _chi2_discriminator_rewards(dis)
        chi2_reward_reverse_list.append(1/(chi2_rewards+1e-6))
    
    
    # Caculate the mean and variance under pi_ref using importance sampling
    episode_rewards = []
    for i in range(len(reward_list)):
        episode_rewards.append(discounted_sum(reward_list[i].cpu()*chi2_reward_reverse_list[i], gamma))
    mean_reward = np.mean(episode_rewards)
    sqrt_var_mean_reward = np.sqrt(np.var(episode_rewards))
    print(mean_reward)

    #episode_rewards_safe = []
    #for i in range(len(episode_batches_safe)):
        #episode_rewards_safe.append(discounted_sum(episode_batches_safe[i][SampleBatch.REWARDS], gamma))
    #mean_reward_safe = np.mean(episode_rewards_safe)
    #sqrt_var_mean_reward_safe = np.sqrt(np.var(episode_rewards_safe))

    normalized_reward = [ (reward.cpu().numpy()-mean_reward)/sqrt_var_mean_reward for reward in reward_list]
    
    #assert np.all(np.isfinite(normalized_reward))

    # Caculate mean square reward
    episode_rewards = []
    for i in range(len(normalized_reward)):
        episode_rewards.append(discounted_sum(normalized_reward[i], gamma))
    mean_reward = np.mean(episode_rewards)*(1-gamma)
    print("mean_reward", mean_reward)
    square_reward = mean_reward**2
    print("mean_square_reward", square_reward)
    
    distance = _sqrt_chi2_kai_rewards(discriminator_list, normalized_reward,  mean_reward, square_reward)

    correlated_r = 0.3
    #correlated_r = 0.999999  # 1-1e-6, for traffic
    #correlated_r = 0.9999  # for glucose
    coeff = np.sqrt(1-correlated_r**2)/correlated_r
    print(coeff)

    episode_rewards = []
    worse_rewards_list = []
    for i in range(len(normalized_reward)):
        episode_rewards.append(discounted_sum(normalized_reward[i]-coeff*distance[i], gamma))
        worse_rewards_list.append(normalized_reward[i]-coeff*distance[i])
    
    #print(episode_rewards)
    sorted_rewards = np.sort(episode_rewards)
    print(sorted_rewards)
    n = len(episode_rewards)
    trim = int(0.01 * n)
    trimmed_rewards = sorted_rewards[trim:-trim] if trim > 0 else sorted_rewards
    print("trimmed_rewards", np.mean(trimmed_rewards))

    # Remove all -inf values
    filtered_list = [x for x in sorted_rewards if not (math.isinf(x) and x < 0)]

    print("remove -inf", np.mean(filtered_list))
    return np.mean(episode_rewards), worse_rewards_list

def weighted_orthonormalize(B_array, weights):
        
    assert B_array.shape[0] == weights.shape[0]
    
    #B_array = (B_array - np.mean(B_array, axis=0)) / (np.std(B_array, axis=0) + 1e-8)

    # Step 3: Weighted orthonormalization
    W_diag = np.diag(weights)
    M = B_array.T @ W_diag @ B_array
    #M = B_array.T @ B_array
    M += 1e-8 * np.eye(M.shape[0])           # Stability

    R = np.linalg.cholesky(M)
    R_inv = np.linalg.inv(R)
    B_tilde = B_array @ R_inv

    # Step 4: Check
    #check = B_tilde.T @ W_diag @ B_tilde
    #print("Weighted Orthonormal Check Matrix:\n", check)
    #assert np.allclose(check, np.eye(B_array.shape[1]), atol=1e-5), "Failed to orthonormalize"

    return B_tilde, R_inv

def caculate_linear_worse_reward(
        reward_list, discriminator_list, info_list
):
    discounted_sum = lambda rewards, gamma: np.sum(np.array(rewards) * np.power(gamma, np.arange(len(rewards))))
    gamma = 0.99

    correlated_r = 0.1

    # Normalize the proxy reward
    # This is pi/pi_ref
    chi2_reward_reverse_list = []
    for dis in discriminator_list:
        chi2_rewards = _chi2_discriminator_rewards(dis)
        chi2_reward_reverse_list.append(1/(chi2_rewards+1e-6))

    # Normalize the proxy reward
    
    # Caculate the mean and variance under pi_ref using importance sampling
    episode_rewards = []
    for i in range(len(reward_list)):
        episode_rewards.append(discounted_sum(reward_list[i].cpu()*chi2_reward_reverse_list[i], gamma))
    mean_reward = np.mean(episode_rewards)
    sqrt_var_mean_reward = np.sqrt(np.var(episode_rewards))


    #safe_batch = train_batch.policy_batches[policy_id]
    
    #episode_batches_safe = safe_batch.split_by_episode()
    #episode_rewards_safe = []
    #for i in range(len(episode_batches_safe)):
        #episode_rewards_safe.append(discounted_sum(episode_batches_safe[i][SampleBatch.REWARDS], gamma))
    #mean_reward_safe = np.mean(episode_rewards_safe)
    #sqrt_var_mean_reward_safe = np.sqrt(np.var(episode_rewards_safe))

    normalized_reward = []
    vel_list = []
    accel_list = []
    headway_list = []
    weight_list = []
    political_list = []
    normalized_reward_list_ = []
    for i in range(len(reward_list)):
        #vel = np.array([entry["vel"] for entry in info_list[i][1:] if "commute" in entry])
        #accel = np.array([entry["accel"] for entry in info_list[i][1:] if "accel" in entry])
        #headway = np.array([entry["headway"] for entry in info_list[i][1:] if "headway" in entry])
        vel = np.array([entry["rew_breakdown"]["InfectionSummaryAbsoluteReward"] for entry in info_list[i][1:] if "rew_breakdown" in entry])
        accel = np.array([entry["rew_breakdown"]["LowerStageReward"] for entry in info_list[i][1:] if "rew_breakdown" in entry])
        headway = np.array([entry["rew_breakdown"]["SmoothStageChangesReward"] for entry in info_list[i][1:] if "rew_breakdown" in entry])
        #political = np.array([entry["true_rew_breakdown"]["PoliticalReward"] for entry in info_list[i][1:] if "rew_breakdown" in entry])
        vel_list.extend(vel)
        accel_list.extend(accel)
        headway_list.extend(headway)
        #political_list.extend(political)
        normalized_reward_list = (reward_list[i].cpu()-mean_reward)/sqrt_var_mean_reward
        normalized_reward_list_.append(normalized_reward_list)
        if len(vel) != len(normalized_reward_list[:-1]):
            continue
        normalized_reward.extend(normalized_reward_list[:-1])
        weight_list.extend(chi2_reward_reverse_list[i][:-1])

    weight = np.array(weight_list)
    # clip negatives to 0, the original output from the network may have negative values
    weight = np.maximum(weight, 1e-2)
    #weight = weight / np.max(weight)   #Normalize weights

    # Caculate Q 
    # Step 1: Stack into feature matrix B (each row is a B(x))
    #B_array = np.column_stack((safe_vel_list, safe_accel_list, safe_headway_list))  # Shape: (N, 3)
    B_array = np.column_stack((vel_list, accel_list, headway_list))  # Shape: (N, 3)
    #B_array = np.column_stack((vel_list, accel_list, headway_list, political_list))
    B_orthogonal, R_inv = weighted_orthonormalize(B_array, weight)
    
    weight = weight[:, np.newaxis]  # (N, 1)
    B_weighted = weight * B_orthogonal             # (N, 3), each row weighted
    
    # Caculate v,d,c shape (3,)
    v = np.mean(B_orthogonal, axis=0)  # Shape: (3,)
    print(v)
    
    #d = np.mean(np.array(normalized_reward)[:, np.newaxis]*B_weighted, axis=0)  # Shape: (3, )
    d = np.mean(np.array(normalized_reward)[:, np.newaxis]*B_orthogonal, axis=0)
    print(d)

    c = np.mean(B_weighted, axis=0)    # Shape: (3,)
    print(c)

    # solve lambda
    lambdas = solve_lambda(v, d, c, correlated_r)
    if lambdas is not None:
        lambda1, lambda2, lambda3 = lambdas
        print("λ₁, λ₂, λ₃ =", lambda1, lambda2, lambda3)

        # caculate optimal theta
        b = v - lambda1 * d - lambda2 * c

        theta = (1 / (2 * lambda3))  * b
        theta = np.maximum(theta, 0)  # Ensure all elements are >= 0
    
    else:
        theta = np.array([1,1,1])
    
    
    print(theta)        
    
    linear_max_min_reward  = (B_orthogonal @ theta)  # Shape: (N,)
    
    count = 0
    final_reward = []
    
    for i in range(len(reward_list)):
        vel = np.array([entry["rew_breakdown"]["InfectionSummaryAbsoluteReward"] for entry in info_list[i][1:] if "rew_breakdown" in entry])
        #vel = np.array([entry["commute"] for entry in info_list[i][1:] if "commute" in entry])
        normalized_reward_list = (reward_list[i].cpu()-mean_reward)/sqrt_var_mean_reward
        if len(vel) != len(normalized_reward_list[:-1]):
            #final_reward.append(normalized_reward_list)
            continue
            #print(info_list)
            #print(episode_batches_safe[i][SampleBatch.REWARDS])
        length = len(vel)
        append_reward_list = []
        append_reward_list.extend(linear_max_min_reward[count:count+length])
        count += length
        append_reward_list.append(normalized_reward_list[-1])
        final_reward.append(append_reward_list)

    episode_rewards = []
    for i in range(len(final_reward)):
        episode_rewards.append(discounted_sum(final_reward[i], gamma))

    return np.mean(episode_rewards)
    
    
    
def solve_lambda(v, d, c, r, init_lambda=None, lambda3_epsilon=1e-2):
    """
    Solve for lambda_1, lambda_2, lambda_3 given v, d, c,  and scalar r.

    Returns:
        np.array([lambda_1, lambda_2, lambda_3])
    """
    # Set initial guess if not provided
    if init_lambda is None:
        # Default initial guess
        init_lambda = np.array([0.0, 0.0, -1])
        # Ensure the default initial guess satisfies the bound
        if init_lambda[2] >= -lambda3_epsilon:
            # Adjust if the default is invalid
            init_lambda[2] = -lambda3_epsilon # Or a value comfortably within the bound like -0.1


    def grad_fn(lam):
        lam1, lam2, lam3 = lam

        #if lam3 >= -lambda3_epsilon/2:
            # Returning NaN is a common way to indicate the function is not defined
            # or differentiable at this point for root finding.
            #return np.inf
        u = v - lam1 * d - lam2 * c
        theta_star = np.maximum(0, u / (2.0 * lam3)) 

        # grad_lambda1 = r - d^T * theta_star
        grad_lambda1 = r - np.dot(d, theta_star)
        # grad_lambda2 = - c^T * theta_star
        grad_lambda2 = - np.dot(c, theta_star)
        # grad_lambda3 = 1 - ||theta_star||^2 (due to sum_x C(x) B(x)B(x)^T = I)
        grad_lambda3 = 1 - np.sum(theta_star**2)

        #return np.array([grad_lambda1, grad_lambda2, grad_lambda3])
        #sum_term = np.dot(theta_star, u) - lam3 * np.sum(theta_star**2)

        # The outer objective function g = sum_term + lambda1*r + lambda3
        #g_val = sum_term + lam1 * r + lam3

        # We want to minimize -g
        #return -g_val
        return np.array([grad_lambda1, grad_lambda2, grad_lambda3])

    #result = root(grad_fn, init_lambda, method='lm')
    # Define bounds for lambda [l1, l2, l3]
    # lambda1 and lambda2 are unbounded (None, None)
    # lambda3 must be < 0. We enforce lambda3 <= -lambda3_epsilon.
    #bounds = [(None, None), # Bounds for lambda1
                #(None, None), # Bounds for lambda2
                #(None, -lambda3_epsilon)] # Bounds for lambda3 (upper bound is negative epsilon)

    # Use scipy's minimizer
    # Pass the fixed parameters (r, v, d, c) using the 'args' argument.
    # These will be passed to the neg_g function after lambda_vec.
    # method='L-BFGS-B' is a good choice for bounded optimization.
    try:
        #result = minimize(grad_fn, init_lambda, args=(v, d, c, r), method='trust-constr', bounds=bounds)
        result = root(grad_fn, init_lambda, method='lm')

    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        return None

    # Check the optimization result
    if result.success:
        # Optionally, perform an additional check that the found lambda3
        # is clearly negative, allowing for some floating-point tolerance.
        if result.x[2] < -lambda3_epsilon / 2.0: # Check if it's significantly less than 0
            return result.x
        else:
            # Optimization succeeded but the lambda3 result is on or near the boundary of the allowed region.
            # This might indicate issues or that the optimal solution is at lambda3=0 (which is outside our derived formula's domain).
            print(f"Optimization succeeded but the resulting lambda3 ({result.x[2]}) is not strictly negative.")
            # Depending on your needs, you might return None, return the result.x, or handle this case specifically.
            return result.x # Returning the result as found by the solver

    else:
        print("Optimization failed:", result.message)
        return None # Return None if the optimization did not succeed

@ex.automain
def main(
    run: str,
    episodes: int,
    per_policy_config_updates: dict,
    extra_config_updates: dict,
    policy_ids: list,
    checkpoint: str,
    out_dir: str,
    generate_histogram: bool,
    hist_x_label: str,
    ray_init_kwargs: dict,
    _log,
):
    
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(),
        **ray_init_kwargs,
    )

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_results: Dict[str, Any] = {}
    model = None
    current_policy_action_dist_class = None
    safe_policy_action_dist_class = None
    safe_policy = None

    # to make sure safe policies come first when setting the model
    if CURRENT_POLICY_ID in policy_ids:
        policy_ids.remove(CURRENT_POLICY_ID)
        policy_ids.sort()
        policy_ids.append(CURRENT_POLICY_ID)
    
    for policy_id in policy_ids:
        config_updates = per_policy_config_updates[policy_id]
        
        config_updates = Algorithm.merge_algorithm_configs(
            config_updates, extra_config_updates, _allow_unknown_configs=True
        )
        
        algorithm = load_algorithm(checkpoint, run, config_updates)
        policy = algorithm.get_policy(policy_id)
        #assert isinstance(policy, TorchPolicyV2)
        assert algorithm.evaluation_workers is not None
        selected_eval_worker_ids = [
            worker_id
            for i, worker_id in enumerate(
                algorithm.evaluation_workers.healthy_worker_ids()
            )
            if i * 1 < episodes
        ]
        
        assert isinstance(algorithm.config, AlgorithmConfig)

        all_batches = []
        total_collected_episodes = 0

        while total_collected_episodes < episodes:
            batches = algorithm.evaluation_workers.foreach_worker(
                func=lambda w: w.sample(),
                local_worker=False,
                remote_worker_ids=selected_eval_worker_ids,
                timeout_seconds=algorithm.config.evaluation_sample_timeout_s,
            )
            all_batches.extend(batches)
            total_collected_episodes += len(batches)

        
        #save_trajectories_to_json(all_batches, "base_samples.json", "maxmin_samples.json")
        #with open("samplebatches.pkl", "wb") as f:
            #pickle.dump(all_batches, f)
        
        eval_results[policy_id] = algorithm.evaluate()["evaluation"]
        with open("eval_result.json", "w") as f:
            json.dump(eval_results, f, indent=4)
        
        #with open("samplebatches.pkl", "rb") as f:
            #all_batches = pickle.load(f)

        
        #Caculate worse linear reward
        '''
        if (len(policy_ids) > 1 and CURRENT_POLICY_ID not in policy_id) or len(
            policy_ids
        ) == 1:
            model = policy.model
            assert isinstance(model, TorchModelV2)
            model.to(device)
            #safe_policy_action_dist_class = policy.dist_class
            safe_policy = policy
            if CURRENT_POLICY_ID in policy_id:
                if algorithm.get_policy(SAFE_POLICY_ID) is not None:
                    safe_policy = algorithm.get_policy(SAFE_POLICY_ID)
                    #safe_policy_action_dist_class = safe_policy.dist_class
                    model = safe_policy.model
                    logger.info("Loaded safe policy from algorithm successfully!")
                else:
                    logger.warn(
                        "Using untrained current policy discriminator for generating discriminator scores!"
                    )
        
        
        reward_list = []
        discriminator_list = []
        obs_list = []
        action_list = []
        info_list = []
        if isinstance(algorithm, ORPO):
            for file_data in all_batches:
                file_data = file_data.as_multi_agent()
                #file_data_cpu = file_data.policy_batches[policy_id].copy()
                try:
                    file_data_cuda = file_data.policy_batches[policy_id].to_device(device)
                except Exception as e:
                    continue
                #data.append(file_data_cuda)
                discriminator_policy_scores = model.discriminator(file_data_cuda)
                rewards = file_data.policy_batches[policy_id][SampleBatch.REWARDS]
                obs = file_data.policy_batches[policy_id][SampleBatch.OBS]
                action = file_data.policy_batches[policy_id][SampleBatch.ACTIONS]
                info = file_data.policy_batches[policy_id][SampleBatch.INFOS]
                reward_list.append(rewards)
                discriminator_list.append(discriminator_policy_scores)
                obs_list.append(obs)
                action_list.append(action)
                info_list.append(info)

        #with open("data.pkl", "wb") as f:
            #pickle.dump({
                #"reward_list": reward_list,
                #"discriminator_list": discriminator_list,
                #"info_list": info_list
            #}, f)
        
        
        linear_worse_reward = caculate_linear_worse_reward(reward_list, discriminator_list, info_list)
        print("linear worse reward", linear_worse_reward)
        
        '''
        
        # Caculate worse reward
        if (len(policy_ids) > 1 and CURRENT_POLICY_ID not in policy_id) or len(
            policy_ids
        ) == 1:
            model = policy.model
            assert isinstance(model, TorchModelV2)
            model.to(device)
            #safe_policy_action_dist_class = policy.dist_class
            safe_policy = policy
            if CURRENT_POLICY_ID in policy_id:
                if algorithm.get_policy(SAFE_POLICY_ID) is not None:
                    safe_policy = algorithm.get_policy(SAFE_POLICY_ID)
                    #safe_policy_action_dist_class = safe_policy.dist_class
                    model = safe_policy.model
                    logger.info("Loaded safe policy from algorithm successfully!")
                else:
                    logger.warn(
                        "Using untrained current policy discriminator for generating discriminator scores!"
                    )
        
        
        reward_list = []
        discriminator_list = []
        obs_list = []
        action_list = []
        if isinstance(algorithm, ORPO):
            for file_data in all_batches:
                file_data = file_data.as_multi_agent()
                #file_data_cpu = file_data.policy_batches[policy_id].copy()
                try:
                    file_data_cuda = file_data.policy_batches[policy_id].to_device(device)
                except Exception as e:
                    continue
                #data.append(file_data_cuda)
                discriminator_policy_scores = model.discriminator(file_data_cuda)
                rewards = file_data.policy_batches[policy_id][SampleBatch.REWARDS]
                obs = file_data.policy_batches[policy_id][SampleBatch.OBS]
                action = file_data.policy_batches[policy_id][SampleBatch.ACTIONS]
                reward_list.append(rewards)
                discriminator_list.append(discriminator_policy_scores)
                obs_list.append(obs)
                action_list.append(action)
        
        
        worse_reward, worse_reward_list = caculate_worse_reward(reward_list, discriminator_list)
        print("worse reward", worse_reward) 
        
        total_number = 0
        inf_number = 0
        for w in worse_reward_list:
            total_number += len(w)
            inf_number += len(w[~np.isfinite(w)])

        print(inf_number/total_number)