import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    """
    A Naïve Bayes classifier that assumes conditional independence among features.
    This implementation uses categorical (count-based) probabilities and log probabilities to prevent underflow.
    """
    
    def __init__(self):
        """
        Initialize necessary data structures for the classifier.
        - self.classes: Stores the unique class labels.
        - self.class_counts: A dictionary to count occurrences of each class.
        - self.feature_counts: A dictionary to count occurrences of feature values for each class.
        - self.feature_totals: A dictionary to store total feature occurrences per class.
        """
        self.classes = None  # Stores unique class labels
        self.class_counts = {}  # Dictionary to store class counts
        self.feature_counts = {}  # Dictionary to store feature counts per class
        self.feature_totals = {}  # Dictionary to store total feature occurrences per class

    def fit(self, X, y):
        """
        Train the Naïve Bayes classifier using count-based probabilities.
        Steps:
        1. Identify unique class labels from `y`.
        2. Count occurrences of each class label to compute prior probabilities.
        3. Count occurrences of feature values for each class to compute likelihood probabilities.
        
        :param X: 2D list or numpy array, where each row represents a sample and each column represents a feature.
        :param y: 1D list or numpy array, where each element corresponds to the class label of a sample.
        """
        self.classes = np.unique(y)  # Step 1: Identify unique class labels
        
        for c_label in self.classes:
            self.class_counts[c_label] = 0
            self.feature_counts[c_label] = {}
            for i in range(X.shape[1]):
                self.feature_counts[c_label][i] = defaultdict(int)
            self.feature_totals[c_label] = np.zeros(X.shape[1], dtype=int)
        
        for xi, label in zip(X, y):  # Iterate over each sample
            # Step 2: Count class occurrences
            # ...
            self.class_counts[label] = self.class_counts[label] + 1
            
            # Initialize feature count storage for each class if not present
            # ...
            for i,feature in enumerate(xi):
                self.feature_counts[label][i][feature] = self.feature_counts[label][i][feature] + 1
                # Step 3: Count occurrences of each feature value for each class
                # ...
                self.feature_totals[label][i] = self.feature_totals[label][i] + 1
            

    def _class_probability(self, x, c):
        """
        Compute the log probability of class `c` given input sample `x` using Naïve Bayes formula.
        Steps:
        1. Compute the log prior probability of class `c`.
        2. Compute the log likelihood of the input sample `x` given class `c`.
        3. Sum log prior and log likelihood to get the final log probability.
        
        :param x: 1D list or numpy array representing a single sample.
        :param c: Class label for which probability is being computed.
        :return: Computed log probability of `c` given `x`.
        """
        # Step 1: Compute log prior probability log(P(C))
        # log_prior = ...
        log_prior = np.log(self.class_counts[c] / np.sum(list(self.class_counts.values())))
        
        # Step 2: Compute log likelihood log(P(X|C)) by apply laplace smoothing
        log_likelihood = 0
        # ...
        for i, xi in enumerate(x):
            log_likelihood= log_likelihood + np.log((self.feature_counts[c][i][xi]+1)/(self.feature_totals[c][i]+len(self.feature_counts[c][i])))
        
        # Step 3: Compute final log probability log(P(C|X)) (ignoring P(X))
        return log_prior + log_likelihood

    def predict(self, X):
        """
        Predict class labels for given input `X`.
        Steps:
        1. Compute the log probability for each class using `_class_probability`.
        2. Choose the class with the highest log probability as the prediction.
        
        :param X: 2D list or numpy array where each row represents a sample.
        :return: 1D numpy array of predicted class labels.
        """
        y_pred = []
        for x in X:
            # Step 1: Compute log probability for each class
            # ...
            class_log_probs = {}
            for c in self.classes:
                class_log_probs[c]=self._class_probability(x,c)

            # Step 2: Choose the class with the highest probability
            # ...
            predicted_class = max(class_log_probs,key=class_log_probs.get)
            y_pred.append(predicted_class)
           

        return np.array(y_pred)