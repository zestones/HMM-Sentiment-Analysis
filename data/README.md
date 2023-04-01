# Hidden Markov Model project
This project is a demonstration of how to use Hidden Markov Models (HMMs) to predict the probability of a given word transitioning to the next word in a sentence. The code is written in Python and uses the `hmmlearn` library. 

## Getting Started 

To get started with this project, you will need to install the `hmmlearn` library. 
You can do this by running the following command: ``` pip install --force-reinstall hmmlearn==0.2.6 ``` 
Once you have installed the `hmmlearn` library, you can run the code in a Jupyter notebook or in a Python environment of your choice. 

## POS Tagging

For this first part of the project, we will use a dataset of sentences and their corresponding POS tags. The dataset is available on Kaggle at the following link: https://www.kaggle.com/code/trunganhdinh/hidden-markov-model-for-pos-tagging/data?select=NER+dataset.csv


The code consists of several parts: 
1. Preprocessing the data
2. Creating a transition probability matrix 
3. Defining a Hidden Markov Model 
4. Calculating the log likelihood of a given sentence 


To create a transition probability matrix, the code takes a set of sentences and splits them into individual words. 
It then creates a count matrix that stores the frequency of transitions between words. The count matrix is normalized to get the transition probability matrix. To define a Hidden Markov Model, the code uses the `hmmlearn` library. 

It defines the number of hidden states and observable states, and trains the model using the Baum-Welch algorithm. To calculate the log likelihood of a given sentence, the code takes a list of words and converts it to a list of word ids. It then iterates through the list of word ids, calculating the log likelihood of each transition based on the word transition matrix. 

## Toy Example 

The code also includes a toy example that demonstrates how to use Hidden Markov Models. In this example, we have **7 hidden states** *('happy', 'sad', 'angry', 'surprised', 'calm', 'fearful' and 'disgusted')* that relate to the emotions.
The observable states are the facial expressions *('smiling', 'frowning', 'neutral', 'grimacing', 'wide-eyed' and 'closed-eyes')*. 

In this toy example we use the ``baum_welch`` algorithm to train the model. The ``baum_welch`` algorithm is an iterative algorithm that uses the forward-backward algorithm to calculate the probability of a given sequence of observations. 
Then we use the ``viterbi`` algorithm to calculate the most likely sequence of hidden states that generated a sequence of observations from the transition and emission matrix that we calculated using the ``baum_welch`` algorithm. The start probability is defined as follow :
````python
n_hidden_states = len(hidden_emotions)
startprob = np.ones(n_hidden_states) / n_hidden_states
````

The start_prob is set to be uniform across all emotions.