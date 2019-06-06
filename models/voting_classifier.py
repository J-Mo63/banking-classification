from scipy.stats import mode


class BagClassifier:
    predictions = []

    def add_prediction(self, pred):
        self.predictions.append(pred)

    def get_vote(self):
        return mode(self.predictions, axis=0)[0][0]
