# TD3 algorithm in deep reinforcement learning
This work has been done as CS591 Deep Learning course project based on "[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)" paper.

### Requirements
To run this algorithm you can make an virtual conda environment

`conda create -n your_env_name python=3.6`

and activate it by

`conda activate your_env_name`

You need to install some modules includes:

`conda install pytorch torchvision -c pytorch`

`conda install -c conda-forge matplotlib`

`pip install pybullet`

`conda install -c akode gym`

### How to run?

To run this algorithm there are two ways:

####First:

1- clone all files 

2- open "TD3_training.py" and change env_name on line 39 to desired environment which is defined in PyBullet

3- run "TD3_training.py" 

4- optimal policy will store in "pytorch_models"

5- to visualize the interaction of agent with environment you can open "main.py" file

6- in "main.py" line 33 change env_name to the same environment you train the algorithm on

7- you can see the video of the agent on "exp/brs/monitor" folder
#### Second:
1- clone all files 

2- open "TD3.py" and change env_name on line 39 to desired environment which is defined in PyBullet

3-
