Group Members:
Logan Caffey 10898282
Zach Flagg 10940872

Programming Language:
Python

Code Structure:
main.py: the file that runs the training script.
run.py: the file that runs the model 1 time without training.
DQAgent.py: the file that has the class with the agent model.
map.py: the file that handles the environment
PER.py: has a class for a prioritized experience replay
DataReader.py: reads house layout files and preps them to be used in the map class


Running The Code:
To run this code first build the docker container with "docker build -t housecleanbot:latest ."
Then to run the scrypt that shows the model in action run "python3 run.py"
To train the bot run "python3 main.py"
