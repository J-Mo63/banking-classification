from scipy.stats import mode
from sklearn.utils import shuffle
import numpy as np


class BagClassifier:
    predictions = []

    def add_prediction(self, pred):
        self.predictions.append(pred)

    def get_vote(self):
        return mode(self.predictions, axis=0)[0][0]
