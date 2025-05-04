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

def prefill_calib_from_existing(train_path, calib_path, ratio=0.2):
    train_path = pathlib.Path(train_path)
    calib_path = pathlib.Path(calib_path)
    calib_path.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(train_path.glob("*.npz"))
    num_to_link = int(len(episode_files) * ratio)
    selected = random.sample(episode_files, num_to_link)

    print(f"Creating {num_to_link} symlinks from {train_path} to {calib_path}...")

    for file in selected:
        dest = calib_path / file.name
        if dest.exists():
            dest.unlink()
        os.symlink(file.resolve(), dest)

    print("Done pre-filling calib_episodes with symlinks.")

def main():
  configs = yaml.YAML(typ='safe', pure=True).load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text()
  )

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
  print(logdir)
  logdir.mkdir(parents=True, exist_ok=True)

  prefill_calib_from_existing(
    train_path='log/raw_logdir/atari_pong/dreamerv2/1/train_episodes',
    calib_path=logdir / 'calib_episodes',
    ratio=0.2
  )

  files = sorted((logdir / "calib_episodes").glob("*.npz"))
  print(f"Found {len(files)} files")
  data = np.load(files[0])
  print(data.files)


if __name__ == '__main__':
  main()