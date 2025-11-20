The Config.py file contains only different hyperparameters that you can set and in that the model that is currently selected is resnet and the use_domain_adaptation variable that is used is currently set to False so it will not use the domain adaptation code it has written.

The Evaluator.py file contains the functions which will load the trained models and compute different metrics like AUC etc and plot the training curves

The Dataloader.py just loads the data set and splits the data into training, validation etc

The trainer.py is just the code to train the selected model for multiple epochs

To train on a data set you will have to copy all the codes from these different python files placing it in the order of Config.py, Dataloader.py, models.py, trainer.py, evaluator.py and main_trainer.py and run the code

If you run the code as is then it will not execute any adaptation techniques
