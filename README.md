# Identification of heavy particles through machine learning for ATLAS trigger system

The file analysis_fastjet.py is a python file with the code of the analysis of the dataset using only fastjet and diverse plots to evaluate the performances.

The file ML_example_analysis_loss=log.py is a python file with the code of the construction of the neural network predicting log(mass), with the loss function defined as the mean squarred error on the output (the logarithme of the mass). It's accompanied by diverse plots to evaluate the performances.

The file ML_example_analysis_flip.py is a python file with the code of the construction of the neural network trained with 4 examplars of the dataset, with eta or phi or both of them flipped. It's accompanied by diverse plots to evaluate the performances.

The file ML_example_analysis_fastjet.py is a python file with the code of the construction of the neural network trained on fastjet results and diverse plots to evaluate the performances.
