""" Entry point for evaluating/rendering a trained policy. """

import argparse
import os
import numpy as np
import time
import datetime

from rltime.general.config import load_config
from rltime.general.type_registry import get_registered_type
from rltime.env_wrappers.common import make_env_creator, EpisodeRecorder
from rltime.env_wrappers.vec_env.sub_proc import make_sub_proc_vec_env
from rltime.general.loggers import DirectoryLogger


def create_policy_from_config(config, action_space, observation_space):
    """Creates a policy from the given config and spaces

    This does not load the weights just creates the policy
    """
    if not isinstance(config, dict):
        config = load_config(config)

    train_cls = get_registered_type(
        "trainers", config['training'].get("type", None))
    assert(hasattr(train_cls, "create_policy")), \
        f"Config training class {type(train_cls)} does not have a " \
        "'create_policy' method"

    model_config = config.get("model")

    return train_cls.create_policy(
        model_config=model_config, action_space=action_space,
        observation_space=observation_space, **config.get("policy_args", {}))


def eval_policy(path, num_envs, episode_count, record=False, record_fps=60,
                render=False, render_fps=None, eps=0.001):
    """Evaluates training result at 'path', loading the last checkpoint

    The result is logged to a new line in file 'eval.json' in <path>

    Args:
        num_envs: Amount of vectorized (sub-process) ENVs to evaluate in
            parallel
        episode_count: The amount of episodes to evaluate total
        record: Whether to record episodes to MP4 (under 'recordings'
            sub-directory in <path>)
        record_fps: If <record>, the FPS to record at (These are raw ENV frames
            before any frame-skipping, so atari would usually be 60)
        render: Whether to render the ENVs in a window in real-time (Tiled if
            num_envs>1)
        render_fps: Frames-Per-Second to sync the rendering to (Valid only for
            render=True), the default (None) renders at max policy speed. These
            are acting steps, so after frame-skipping if active
        eps: Epsilon to use for random action selection

    Note: We count the first 'episode_count' episodes that started and not
        ended, as 'ended' is unfair to longer episodes in case of vectorized
        evaluation. For Example: Take a policy that achieves 100 reward in 100
        seconds 50% of the time and 0 reward in <1 second 50% of the time.
        So we'd expect if we evaluate 20 episodes to get around ~50 average
        reward (which we would if running 20 episodes serially on a single ENV)
        But if we run 16 ENVs in parallel we will likely get near-0 mean reward
        if we count the first 20 episodes that finished (Since half of the 16
        ENVs immediately end with reward 0 then restart, then half of those
        immediately end with 0 and so on, so we quickly get ~(8+4+2+1) 0-reward
        runs and don't count the ones which are running long and going to reach
        100 reward), while if we take the first 20 episodes that started (and
        ignore any that started after) we will get the expected result
    """
    print("Evaluating:", path)
    assert(num_envs <= episode_count), \
        "num_envs can't be higher than the requested episode_count"

    logger = DirectoryLogger(path, use_logging=False, tensorboard=False)

    # Load the config from the result path
    config = logger.get_config()

    # Make the env-creaton function based on the config settings
    env_args = config.get("env_args", {})
    if record:
        # If requested, add also an episode-recorder to the ENV stack
        recorder = {
            "type": EpisodeRecorder,
            "args": {
                "path": os.path.join(path, "recordings"),
                "fps": record_fps
            }
        }
        env_args['wrappers'] = [recorder] + env_args.get('wrappers', [])

    env_creator = make_env_creator(config.get("env"), **env_args)

    # Create a vectorized ENV
    env = make_sub_proc_vec_env(env_creator, num_envs)

    # Create the policy based on the config
    policy = create_policy_from_config(
        config, env.action_space, env.observation_space)

    # Load the last checkpoint
    training_step, cp_data = logger.get_checkpoint()
    # Load the weights from the checkpoint to the policy
    policy.load_state(cp_data['policy_state'])
    print("Loaded checkpoint from step:", training_step)

    # The initial policy input state
    state = policy.make_input_state(env.reset(), np.array([True] * num_envs))

    episodes_started = num_envs
    rewards = []
    # This signifies the ENV started the episode in time and should be counted
    masks = [True] * num_envs

    print(f"Running '{config.get('env')}' for {episode_count} episodes"
          f" on {num_envs} ENVs")
    while len(rewards) < episode_count:
        step_start = time.time()
        # Select the next action for each env
        preds = policy.actor_predict(state, timesteps=1)
        actions = preds['actions']
        if eps:
            # Remap to random actions with eps probability
            for i in range(num_envs):
                if np.random.rand() < eps:
                    actions[i] = env.action_space.sample()
        # Send the action and get the transition data
        obs, _, dones, info = env.step(actions)

        # Check any env if finished
        for i, env_info in enumerate(info):
            # We use the 'real' done/reward from the EpisodeTracker wrapper
            if env_info['episode_info']['done']:
                if masks[i]:
                    # Only count the first 'episode_count' that started
                    reward = env_info['episode_info']['reward']
                    rewards.append(reward)
                    print(f"Episode {len(rewards)}/{episode_count} "
                          f"finished with reward: {reward}")

                episodes_started += 1
                if episodes_started > episode_count:
                    masks[i] = False

        # Render to screen if requested
        if render:
            if render_fps:
                diff = 1./render_fps - (time.time() - step_start)
                if diff > 0:
                    time.sleep(diff)
            env.render()
        # Generate the next polict input state
        state = policy.make_input_state(obs, dones)

    env.close()

    # Log the result
    result = {
        "step": training_step,
        "date": datetime.datetime.now(),
        "episodes": episode_count,
        "envs": num_envs,
        "reward": {
            "mean": np.mean(rewards),
            "min": np.min(rewards),
            "max": np.max(rewards),
            "median": np.median(rewards),
            "std": np.std(rewards),
        }
    }
    print("Result:")
    logger.log_result("eval", result, None)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', type=str,
        help="The path to the training directory result to evaluate")
    parser.add_argument(
        '--num-envs', type=int, default=1,
        help="Amount of ENVs to run in parallel")
    parser.add_argument(
        '--episodes', type=int, default=5,
        help="Amount of episodes to run")
    parser.add_argument(
        '--record', action='store_true',
        help="Whether to record episode to MP4 (To a sub-directory in the "
             "result path). Warning: If used with --num-envs>1 the last "
             "videos will be truncated")
    parser.add_argument(
        '--record-fps', type=int, default=60,
        help="FPS to record at if --record (Typically 60FPS for atari)")
    parser.add_argument(
        '--render', action='store_true',
        help="Whether to render the episodes in real-time")
    parser.add_argument(
        '--render-fps', type=int, default=0,
        help="FPS to sync to if using --render (Set to 0 for full speed), "
        "note this is after ENV frame-skipping so if you want 60FPS with "
        "frame-skip of 4 use 15 here")
    parser.add_argument(
        '--eps', type=float, default=0.001,
        help="Epsilon value to use for random action selection during "
             "evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    eval_policy(
        args.path, num_envs=args.num_envs, episode_count=args.episodes,
        record=args.record, record_fps=args.record_fps,
        render=args.render, render_fps=args.render_fps, eps=args.eps)

if __name__ == '__main__':
    main()
