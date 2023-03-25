import torch

import NAME


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.loss = Loss()

    def __call__(self):
        return self.loss()

    def update(self, logits, target):
        # Detach from graph
        logits = logits.detach()

        # Update loss
        self.loss.update(logits, target)

    def reset(self):
        self.loss.reset()


###############################################################################
# Individual metric
###############################################################################


class Loss():

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'loss': (self.total / self.count).item()}

    def update(self, logits, target):
        self.total += NAME.train.loss(logits, target)
        self.count += target.shape[0]

    def reset(self):
        self.count = 0
        self.total = 0.
