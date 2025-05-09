import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
from pathlib import Path

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common
import argparse
import random
import shutil

def prefill_calib_eval_split(train_path, calib_path, eval_path, calib_ratio=0.2, eval_ratio=0.1, seed=42):
    train_path = pathlib.Path(train_path)
    calib_path = pathlib.Path(calib_path)
    eval_path = pathlib.Path(eval_path)
    calib_path.mkdir(parents=True, exist_ok=True)
    eval_path.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(train_path.glob("*.npz"))
    total = len(episode_files)
    random.seed(seed)
    random.shuffle(episode_files)

    num_calib = int(total * calib_ratio)
    num_eval = int(total * eval_ratio)
    calib_files = episode_files[:num_calib]
    eval_files = episode_files[num_calib:num_calib+num_eval]

    def symlink_files(files, dest):
        for file in files:
            dest_file = dest / file.name
            if dest_file.exists():
                dest_file.unlink()
            os.symlink(file.resolve(), dest_file)

    print(f"Creating {len(calib_files)} calibration symlinks...")
    symlink_files(calib_files, calib_path)
    print(f"Creating {len(eval_files)} evaluation symlinks...")
    symlink_files(eval_files, eval_path)
    print("Done splitting calibration and evaluation sets.")

def main():
  # Load config
  configs = yaml.YAML(typ='safe', pure=True).load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parser = argparse.ArgumentParser()
  parser.add_argument('--rssm_calibrate_mode', type=str, default=None)
  args, remaining = parser.parse_known_args()

  parsed, leftover = common.Flags(configs=['defaults']).parse(argv=remaining, known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
      config = config.update(configs[name])
  if args.rssm_calibrate_mode is not None:
      config = config.update({'rssm.calibrate_mode': args.rssm_calibrate_mode})
  config = common.Flags(config).parse(leftover)

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')

  # TensorFlow
  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  # Prefill calib episodes
  prefill_calib_eval_split(
    train_path='log/raw_logdir/atari_pong/dreamerv2/1/train_episodes',
    calib_path=logdir / 'calib_episodes',
    eval_path=logdir / 'eval_episodes',
    calib_ratio=0.2,
    eval_ratio=0.1
    )

  train_replay = common.Replay(logdir / 'calib_episodes', capacity=1000,
      minlen=config.dataset.length, maxlen=config.dataset.length)

  eval_replay = common.Replay(logdir / 'eval_episodes', capacity=1000,
    minlen=config.dataset.length, maxlen=config.dataset.length)
  
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  
  step = common.Counter(train_replay.stats['total_steps'])
  logger = common.Logger(step, [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ], multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  print(config.rssm.calibrate_mode)

  def train_step(tran, worker):
    mets = train_agent(next(train_dataset))
    for k, v in mets.items():
      metrics[k].append(v)
    if int(step) % config.log_every == 0:
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.write()
      if config.rssm.calibrate_mode == 'global':
        temp_var = [v for v in agnt.wm.rssm.trainable_variables if 'temperature' in v.name]
        if temp_var:
          print(f"Step {int(step)} | Temperature: {temp_var[0].numpy():.4f}")
      elif config.rssm.calibrate_mode == 'platt':
        platt_a = [v for v in agnt.wm.rssm.trainable_variables if 'platt_a' in v.name]
        platt_b = [v for v in agnt.wm.rssm.trainable_variables if 'platt_b' in v.name]
        if platt_a and platt_b:
          print(f"Step {int(step)} | Platt a: {platt_a[0].numpy():.4f} | b: {platt_b[0].numpy():.4f}")

  def make_env():
    suite, task = config.task.split('_', 1)
    if suite == 'atari':
      env = common.Atari(task, config.action_repeat, config.render_size, config.atari_grayscale)
      env = common.OneHotAction(env)
    else:
      raise NotImplementedError(suite)
    return common.TimeLimit(env, config.time_limit)

  # Environment setup
  train_envs = [make_env() for _ in range(config.envs)]
  eval_envs = [make_env() for _ in range(config.eval_eps)]
  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space

  # Driver setup
  train_driver = common.Driver(train_envs)
  eval_driver = common.Driver(eval_envs)
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  train_driver.on_step(train_step)

  # Eval logic
  def eval_episode(ep, mode='eval'):
    score = float(ep['reward'].astype(np.float64).sum())
    print(f"[Eval] Return: {score:.1f} | Length: {len(ep['reward']) - 1}")
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', len(ep['reward']) - 1)
    logger.write()

  eval_driver.on_episode(lambda ep: eval_episode(ep, mode='eval'))

  # Agent setup
  train_dataset = iter(train_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, obs_space, act_space, step)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  should_expl = common.Until(config.expl_until // config.action_repeat)
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')


  model_path = Path('log/raw_logdir/atari_pong/dreamerv2/1/variables.pkl')
  if model_path.exists():
    agnt.load(model_path)
  else:
    print('Pretrained model not found, exiting.')
    sys.exit(1)

  print("Freezing all weights except calibration parameters...")
  for var in agnt.trainable_variables:
    var._trainable = False
  for var in agnt.wm.rssm.trainable_variables:
    if any(x in var.name for x in ['temperature', 'platt_a', 'platt_b']):
      var._trainable = True
      print("Calibration trainable:", var.name)

  print("Start online calibration training.")

  while int(step) < config.steps:
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    print('Start training.')
    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')
    save_name = f'{config.rssm.calibrate_mode}-calib-model.pkl'
    agnt.save(logdir / save_name)
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass

if __name__ == '__main__':
  main()