import pathlib
import collections
import ruamel.yaml as yaml
import agent
import common
import numpy as np
import sys
import traceback

def main():

    calib_episodes = 10
    configs = yaml.YAML(typ='safe', pure=True).load(
        pathlib.Path('log/logdir/atari_pong/dreamerv2/1/config.yaml').read_text())
    config = common.Config(configs)

    logdir = pathlib.Path('log/logdir/atari_pong/dreamerv2/1').expanduser()

    calib_replay = common.Replay(logdir / 'calib_episodes',
                                 capacity=1000,
                                 minlen=config.dataset.length,
                                 maxlen=config.dataset.length)

    def make_env():
        suite, task = config.task.split('_', 1)
        if suite == 'atari':
            env = common.Atari(task, config.action_repeat, config.render_size, config.atari_grayscale)
            env = common.OneHotAction(env)
        elif suite == 'dmc':
            env = common.DMC(task, config.action_repeat, config.render_size, config.dmc_camera)
            env = common.NormalizeAction(env)
        elif suite == 'crafter':
            assert config.action_repeat == 1
            env = common.Crafter(None, reward=True)
            env = common.OneHotAction(env)
        else:
            raise NotImplementedError(suite)
        return common.TimeLimit(env, config.time_limit)

    # eval_envs = [make_env() for _ in range(config.envs)]
    step = common.Counter(0)
    train_envs = [make_env() for _ in range(config.envs)]
    driver = common.Driver(train_envs)
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    agnt = agent.Agent(config, obs_space, act_space, step)
    # print(logdir / 'variables.pkl')
    try:
        print('[DEBUG] Start loading model...')
        agnt.load(logdir / 'variables.pkl')
        print('[DEBUG] Model loaded successfully!')
    except FileNotFoundError:
        print('[ERROR] Model file not found!')
        sys.exit(1)
    except Exception as e:
        print('[ERROR] Failed to load model:', e)
        traceback.print_exc()
        sys.exit(1)

    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    def save_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'[Calibration] Episode: length={length}, return={score:.1f}')
        calib_replay.add_episode(ep)

    driver.on_episode(save_episode)

    print(f'[Calibration] Start collecting {calib_episodes} episodes...')
    driver(eval_policy, episodes=calib_episodes)
    print('[Calibration] Done.')

    for env in train_envs:
        try:
            env.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()