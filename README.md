# dreamer-calibration-experiments
## Manual Instructions

Get dependencies:

- environment.yml
Contains all conda installed packages and their versions and channels, use 
`conda env create -f environment.yml` to rebuild the environment with one click.

- requirements.txt
Lists all packages installed by pip and their versions, use `pip install -r requirements.txt` to restore those packages.

Note: 
If you see
````
import gym.envs.atari
ModuleNotFoundError: No module named 'gym.envs.atari'
````

Please try building Gym from source yourself.

````
wget https://github.com/openai/gym/archive/refs/tags/0.19.0.zip -O gym-0.19.0.zip
unzip gym-0.19.0.zip
cd gym-0.19.0
sed -i 's/opencv-python>=3\./opencv-python>=3.0/' setup.py
pip install --no-build-isolation .
pip install autorom
AutoROM --accept-license
````

Use the following code to verify,
````
import cv2, gym, gym.envs.atari
print("cv2:", cv2.__version__)
env = gym.make("Pong-v0")
print("Loaded Atari env:", env)
````