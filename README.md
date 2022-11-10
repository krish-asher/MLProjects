# Machine Learning Projects

Here I included a few of the machine learning projects that I have created.  My primary motivation behind these projects was learning what exactly you can do with machine learning.  I find it fascinating how ML can be utilized to analyze complex problems that would be difficult to solve using typical programming methods.

## Reinforcement Learning Agent
This reinforcment learning agent balances a pole that is placed on a moving cart (environment is provided by openAI).  The agent learns from each successive attempt how to balance the pole until ultimately it is able to balance the pole.  The way this works is every second the pole hasn't toppled off the cart the agent gets a reward.  The goal of the agent is to get the highest reward.  After many successive attempts, the agent learns how to balance the pole.  This specific type of reinforcement learning is called Q-Learning. Libraries used for this project:
* NumPy - analyzes the data that is outputted by the agent
* TensorFlow - creates deep neural network for the agent
* Keras-RL - creates Q-Learning model which allows the agent to learn after each attempt

## Digit Recognizer
This deep neural network takes in 28x28 image of a handwritten digit and computes what the numerical value of the digit wth 95% accuracy. The model uses data from the mnist library to train the DNN.  Libraries that were used in this project: 
* CV2 (openCV) - processes the images that the user inputs
* NumPy - resizes the image that the user inputs
* MatPlotLib - graphs the number that was examined for the user to view
* TensorFlow - creates the deep neural network that analyzes the handwritten digits

## Regression Model
This deep neural network regression model predicts the miles per gallon (MPG) for a car based on certain attributes of the car like horsepower or weight.  This model uses data given by the UCI Machine Learnig Repository to create the DNN regression model. Libraries that were used in this project:
* Pandas - reads the CSV file with the input data
* NumPy - manipulates the input data so it can be outputted for the user to analyze
* MatPlotLib - graphs the regression model for user to analyze
* TensorFlow - creates the deep neural network that gives the regression model for the attribute and the MPG


