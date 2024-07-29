# Identification of heavy particles through machine learning for ATLAS trigger system

All the codes use a dataset saved in the directory, so be aware to change the name of the data file at the beggining of each file. To run, it also needs to satisfy some requirements. You can download them using the file *requirements.txt*.

The file *analysis_fastjet.py* is a python file with the code of the analysis of the dataset using only fastjet and diverse plots to evaluate the performances.

The file *ML_example_analysis_loss=log.py* is a python file with the code of the construction of the neural network predicting log(mass), with the loss function defined as the mean squarred error on the output (the logarithme of the mass). It's accompanied by diverse plots to evaluate the performances.

The file *ML_example_analysis_loss=sym.py* is a python file with the code of the construction of the neural network predicting the mass, with the loss function defined as (1 - mass predicted / true mass)^2. It's accompanied by diverse plots to evaluate the performances. In comments, there is the code for the 3D plot of pT as a function of eta and phi and the 2D histogram of pT. It takes a lot of time to run. This version of the neural network is the one used in the below alternatives.

The file *ML_example_analysis_flip.py* is a python file with the code of the construction of the neural network trained with 4 examplars of the dataset, with eta or phi or both of them flipped. It's accompanied by diverse plots to evaluate the performances.

The file *ML_example_analysis_fastjet.py* is a python file with the code of the construction of the neural network trained on fastjet results and diverse plots to evaluate the performances.

In every file we have the code to do a pT cut on the data. It located at the beginning of the file, when the data are read, an it's in comments.

