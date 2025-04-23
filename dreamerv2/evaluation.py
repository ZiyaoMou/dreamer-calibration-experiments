import pickle
import pathlib
import tensorflow as tf
from agent import Agent
from common import Config
from common import EnsembleRSSM
import pathlib
import collections
import ruamel.yaml as yaml
import agent
import common
import numpy as np
import sys
import traceback
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_logits_and_labels(rssm, encoder, dataset, max_batches=100, discrete=True):
    logits_list, labels_list = [], []

    for i, batch in enumerate(dataset):
        if i >= max_batches:
            break

        obs = batch['obs'] if 'obs' in batch else batch 
        embed = encoder(obs)  # (B, T, D)
        post, prior = rssm.observe(embed, batch['action'], batch['is_first'])
        dist = rssm.get_dist(prior)
        if discrete:
            logits = dist.distribution.logits_parameter() 
            probs = tf.nn.softmax(logits, axis=-1)
            logits_reshaped = tf.reshape(probs, [-1, probs.shape[-1]])  # (B*T*stoch, discrete)
            labels = tf.argmax(probs, axis=-1)  # (B, T, stoch)
            labels_reshaped = tf.reshape(labels, [-1])  # (B*T*stoch,)

            logits_list.append(logits_reshaped)
            labels_list.append(labels_reshaped)

        else:
            mean = dist.mean()  # (B, T, stoch)
            std = dist.stddev()
            binary_label = tf.cast(mean > 0, tf.int32)
            probs = tf.math.sigmoid(mean)  # (B, T, stoch)

            logits_list.append(tf.reshape(probs, [-1]))
            labels_list.append(tf.reshape(binary_label, [-1]))

    logits_all = tf.concat(logits_list, axis=0).numpy()
    labels_all = tf.concat(labels_list, axis=0).numpy()
    return logits_all, labels_all

def compute_ece(probs, labels, num_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(num_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        count = np.sum(mask)
        if count == 0:
            continue
        acc = np.mean(accuracies[mask])
        conf = np.mean(confidences[mask])
        ece += (count / len(labels)) * abs(acc - conf)

        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(count)

    return ece, bins, bin_accs, bin_confs

def compute_brier_score(probs, labels, num_classes):
    # One-hot encoding
    onehot = np.eye(num_classes)[labels]
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))

def plot_calibration_curve(bins, accs, confs, title):
    os.makedirs('evaluation', exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.plot(confs, accs, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.show()
    safe_title = title.replace('/', '_').replace(' ', '_')
    save_path = os.path.join('evaluation', f'{safe_title}.png')
    plt.savefig(save_path)
    plt.close()

def load_rssm_from_pkl(pkl_path, config, obs_space, act_space, step):
    agent = Agent(config, obs_space, act_space, step)
    agent.load(pkl_path)
    return agent.wm.rssm, agent.wm.encoder

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
    rssm_raw, encoder_raw = load_rssm_from_pkl(
        'log/cal_logdir/variables.pkl', config, obs_space, act_space, step)
    rssm_cal, encoder_cal = load_rssm_from_pkl(
    'log/cal_logdir/atari_pong/dreamerv2/1/variables.pkl', config, obs_space, act_space, step)
    rssm_platt_cal, encoder_platt_cal = load_rssm_from_pkl(
    'log/cal_logdir_platt/atari_pong/dreamerv2/1/platt-variables.pkl', config, obs_space, act_space, step)
    
    replay = common.Replay(Path('log/cal_logdir/calib_episodes'), **dict(
        capacity=1000,
        minlen=config.dataset.length,
        maxlen=config.dataset.length
    ))
    dataset = iter(replay.dataset(**config.dataset))

    probs_raw, labels = get_logits_and_labels(rssm_raw, encoder_raw, dataset)
    probs_cal, _ = get_logits_and_labels(rssm_cal, encoder_cal, dataset)
    probs_platt_cal, _ = get_logits_and_labels(rssm_platt_cal, encoder_platt_cal, dataset)

    ece_raw, bins, accs_raw, confs_raw = compute_ece(probs_raw, labels)
    ece_cal, _, accs_cal, confs_cal = compute_ece(probs_cal, labels)
    ece_platt_cal, _platt, accs_platt_cal, confs_platt_cal = compute_ece(probs_platt_cal, labels)

    num_classes = probs_raw.shape[1]

    print('[raw_prob]', probs_raw)
    print('[cal_prob]', probs_cal)
    print('[labels]', labels)

    brier_raw = compute_brier_score(probs_raw, labels, num_classes)
    brier_cal = compute_brier_score(probs_cal, labels, num_classes)
    brier_platt_cal = compute_brier_score(probs_platt_cal, labels, num_classes)

    print(f"[Raw] ECE: {ece_raw:.4f}, Brier Score: {brier_raw:.4f}")
    print(f"[Calibrated] ECE: {ece_cal:.4f}, Brier Score: {brier_cal:.4f}")
    print(f"[Platt Calibrated ECE: {ece_platt_cal: .4f}, Brier Score: {brier_platt_cal:.4f}]")

    raw_conf = np.max(probs_raw, axis=1)
    cal_conf = np.max(probs_cal, axis=1)
    platt_conf = np.max(probs_platt_cal, axis=1)

    print(f"[Raw] Confidence range: {raw_conf.min():.3f} ~ {raw_conf.max():.3f}")
    print(f"[Cal] Confidence range: {cal_conf.min():.3f} ~ {cal_conf.max():.3f}")
    print(f"[Platt Cal] Confidence range: {platt_conf.min():.3f} ~ {platt_conf.max():.3f}")

    plot_calibration_curve(bins, accs_raw, confs_raw, 'Raw Model')
    plot_calibration_curve(bins, accs_cal, confs_cal, 'Temperature Calibrated Model')
    plot_calibration_curve(bins, accs_platt_cal, confs_platt_cal, 'Platt Calibrated Model')

if __name__ == '__main__':
  main()
