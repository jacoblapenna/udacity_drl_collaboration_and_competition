# Collaboration and Control (Tennis)

This project solves the Udacity tennis environment. It uses a neural network defined in [model.py](/model.py). The Agent class implements all the code needed to train the model given specific instantiation hyperparameters. The report details the specific architecture and how the model was trained in various ways.

Follow the instructions in the [Tennis](/Tennis.ipynb) notebook to install the dependencies. Due to the inability to find the correct environment online, the workspace was used to complete this project. For this reason, the environment path within the Agent class defaults to the recommended environment path.

To train an agent, simply instantiate it with the desired hyperparameters and call `agent.train_agent()` with the desired episode max count and solution threshold. The session stores the scores for each actor for each episode, along with the hyperperameters used for that training session. This information is store in the SQLite database for later comparison in the [Report][\Report.ipynb].

NOTE: if running outside the workpace, agent instantiation must use the correct environment path. This is the path where the Unity environment is located. As mentioned, the project is so outdated I could not find the correct Unity dependencies online. This means I could only run the project within the workspace with the older Unity dependencies and builds. On my system, the provided environment build was too old for the available Unity downloads. For this reason, only a basic README is needed, as it is expected that this project is ran on the provided Udacity workspace virtual machine with all dependencies pre-installed and the environment path already setup.

## BS Requirement From Udacity:

In order to pass the project I am required to say something about the state space in this README. The states space is comprised of 8 data points for the balls position and velocity. Three of these 8D data frames are stacked together for each state observation to make a 24-point state for each actor. Each observation is a (1, 2, 24) tesnor where each actor only takes the 24-point observation that it would see/experience in game play. This is the input for each actor model. The output is a 2-value action for forwar/back and "jump" (i.e., paddle up to hit). Solution success is when either actor reaches an average score of 0.5 (i.e., 5 hits in a round) over the last 100 games.

## Future Work

I'd like to investigate shared resources between agents/players. Specifically, I wonder if a mixture of experts architecture would work here to reduce computational complexity in the multi-model architecture. Such an architecture would use typical MoE structure of one model with a gating network that would learn which expert (i.e., player) to route outputs from. MoE is typically for supervised clustering applications, however, a critic may be employed to facilitate unsupervised clustering. This could be applied to this scenario because what is multiplayer game play other than just unsupervised action clustering. To my knowledge, this has not been done before in the literature and such an architecture would represent a truly novel application of this learning method.
