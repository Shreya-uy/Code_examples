The repository comprises code examples for some tasks in machine learning, reinforcement learning and natural language processing.

**MLP_assign_** : This file contains the code to perform a classification task on the MNIST dataset using stochastic gradient descent. I have used the scikit learn library to preprocess the data and fit SGD classifiers by tuning different hyperparameters. The loss curves generated at various stages of training are plotted and compared for performance.

**NLP_assign_classifer**: In this file, I have implemented 2 models - a generative and a discriminative model to attribute authorship to lines of text. The goal is to use both models to generate text/identify the author of the text. The generative models are implemented using the nltk library. I have used unigram, bigram and trigram models. Perplexity scores are used to compare performance of the models and make predictions. The discriminative classifier has been implemented using the hugging face library. Finally, a comparison is made of the results of both the models.

**RL_assign_jackscar**: This code implements a resource allocation optimisation problem using reinforcement learning, wherein I have arrived at the optimal inventory level of cars at 2 locations given the variables of rental earned, storage cost, transfer cost and return cost, given the distribution of these costs. Policy iteration is used to arrive at the ideal policy to minimise cost.

**project_appDev** : This is an application development project to create a web based Kanban board. The application has been built 
using the python-flask framework and the SQLAlchemy database architecture. Further details are contained in the README file in the project folder.
