import pathlib
import collections
import ruamel.yaml as yaml
import agent
import common
import numpy as np
import sys
import traceback
import tensorflow as tf

class GaussianTemperatureScaler(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.temperature = tf.Variable(1.0, trainable=True, dtype=tf.float32)

    def call(self, mean, std):
        return mean, std * self.temperature

    def nll_loss(self, mean, std, target):
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        return -dist.log_prob(target)

    def fit(self, mean, std, target, epochs=100, lr=1e-2):
        opt = tf.keras.optimizers.Adam(lr)
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.nll_loss(mean, std * self.temperature, target)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, [self.temperature])
            opt.apply_gradients(zip(grads, [self.temperature]))

def main():

    calib_episodes = 10
    configs = yaml.YAML(typ='safe', pure=True).load(
        pathlib.Path('log1/logdir1/atari_pong/dreamerv2/1/config.yaml').read_text())
    config = common.Config(configs)

    logdir = pathlib.Path('log1/logdir1/atari_pong/dreamerv2/1').expanduser()

    calib_replay = common.Replay(logdir / 'calib_episodes',
                                 capacity=1000,
                                 minlen=config.dataset.length,
                                 maxlen=config.dataset.length)

    dataset = iter(calib_replay.dataset(batch=16, length=50))                             

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

    def extract_rssm_calibration_data(dataset, agent, max_batches=100):
    rssm = agent.rssm
    encoder = agent.encoder

    preds = []
    targets = []

    for i, batch in enumerate(dataset):
        if i >= max_batches:
            break

        embed = encoder(batch)
        post, prior = rssm.observe(embed, batch['action'], batch['is_first'])

        if rssm._discrete:
            logits = prior['logit']
            logits = tf.reshape(logits, [logits.shape[0], logits.shape[1], -1])  
            preds.append(tf.reshape(logits, [-1, logits.shape[-1]]))
        else:
            means = prior['mean']
            preds.append(tf.reshape(means, [-1, means.shape[-1]]))

        flat_embed = tf.reshape(embed, [-1, embed.shape[-1]])
        targets.append(flat_embed)

    return tf.concat(preds, axis=0), tf.concat(targets, axis=0)

    preds, targets = extract_rssm_calibration_data(dataset, agnt, max_batches=100)
    stds = tf.ones_like(preds) * config.rssm_min_std
    scaler = GaussianTemperatureScaler()
    scaler.fit(preds, stds, targets)
    
    driver(eval_policy, episodes=calib_episodes)
    print("Optimal temperature:", scaler.temperature.numpy())

if __name__ == '__main__':
    main()