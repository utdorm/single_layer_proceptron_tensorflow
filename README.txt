///////////////////////////////////////////////
This project is implemented using python programming language. Tested with python 3.6.4, but it may be 
compatabile with other versions.

Installing python from links below
https://www.python.org/downloads/

///////////////////////////////////////////////
The project is current running based on keras.io library, which is a high-level neural network API, written in Python and capable of 
running on top of tensorflow machine learning backends.

Installing Keras.io from links below
https://keras.io/#installation


///////////////////////////////////////////////
Step by Step
	1. Run create_training_set.py || to convert your dataset into arrays of numbers. These arrays represents the
	the data samples. After running the file, you will see how many classes that will be generate based on your 
	folder orginization[*].
	
	2. Run main.py || to generate a log folder that contains lots of neural network architectures combination. 
	Also, make sure to add more combination as well as the hidden layers.

	3. (At this point, assume that you found the best model combination, eg, 32x28x18 architecture) 
	Run training_manual.py || to create a model of that architecture. Be sure to change the nodes number. After the training
	precedure is done, there will be a file named XXXXXX.model[**] exstention. You will need this model for prediction.

	4. Test out your beloved model. Run prediction.py || this file will open a unique sample from ./example[***] folder, then uses
	the last saved model to classify the unique sample. 

[*]: Create a new folder named ./dataset if you don't have one
[**]: Please change the name of the model inside the code to your preferences
[***]: Please copy your unique sample here, if you want to predict the sample

///////////////////////////////////////////////
Installations

To run the project
	1. open python IDE
	2. open the main.py
	3. open "Run" tab on the top menu bar then "Run Module" (Shortcut: F5)

IMPORTANT NOTE: The project contains third-party open-source library usage.
If the project failed to compile due to missing library, please navigate to project directory using a 
terminal/command prompt then run the command below.

$ pip install -r requirements.txt

///////////////////////////////////////////////
Running the Saved Model

The main.py file will not generate the model, in other to save the specific model combination to your liking,
you will have to compile another file called, training_manual.py. You will have the ability to save the model and
also customize the model combination.

The model name will be saved after the traninig procedure is done.

///////////////////////////////////////////////
Data Collection

The data does not needed for current stage of the project, but it is available for download for 
personal usage as well in the link below.
https://drive.google.com/open?id=1bpk3pQKvp2VclWziWmuXoWsuealmviLY