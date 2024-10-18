"""
This is a class that takes in a training set (Stimulus object),
and then produces the cross-validation and test sets.
Essentially, it produces 2 randomly shuffled versions of
the training set (Stimulus object).
The result is an object that contains 3 Stimulus objects.
TXT stands for Traning, X validation, and Test.
"""
from stimulus import Stimulus
import numpy as np
import copy


class TXTSets(object):

    def __init__(self, base, n=1):
        '''
        Takes in a Stimulus object as training set.
        randomly shuffles the order of its patterns,
        and comes up with the Validation and Test sets.
        NOTE: I call it xval_set even though it's not really cross-validation.
        '''
        self.repetitions = n
        self.base = copy.deepcopy(base)
        self.training_set = self.shuffle( self.expand(copy.deepcopy(base), n) )
        self.xval_set = self.shuffle( self.expand(copy.deepcopy(base), n) )
        self.test_set = self.shuffle( self.expand(copy.deepcopy(base), n) )

    @classmethod
    def shuffle(cls, stimulus):
        # takes in the building blocks of a stimulus, and shuffles the patterns.
        # first by generating an order to re-arrange by.
        # stimulus building blocks to re-order: patterns, gaps, labels.
        order = np.random.permutation(len(stimulus.patterns[0]))
        pattern_params, patterns, gaps, labels = ([],[],[],[])
        for i, params in enumerate(stimulus.pattern_params):
            pattern_params.append(np.array(params)[order].tolist())
            patterns.append(np.array(stimulus.patterns[i])[order].tolist())
            gaps.append(np.array(stimulus.gaps[i])[order].tolist())
        labels = np.array(stimulus.labels)[order].tolist()
        return cls.remake_stimulus(stimulus, pattern_params, patterns, gaps, labels)

    @staticmethod
    def expand(stimulus, n=1):
        # expands every attribute (all lists) of a stimulus by a factor of n
        stimulus.labels *= n
        stimulus.duration *= n
        for i in range(len(stimulus.patterns)):
            stimulus.pattern_params[i] *= n
            stimulus.patterns[i] *= n
            stimulus.gaps[i] *= n
            stimulus.spike_trains[i] *= n
        return stimulus

    @staticmethod
    def remake_stimulus(stimulus, pattern_params, patterns, gaps, labels):
        # resets the stimulus object's properties based on the new patterns/gaps/labels
        # does not generate a new object, but merely modifies the same object
        s = []  # the new stimulus spike trains
        for i, ith_fiber in enumerate(patterns):
            fiber, time = ([], 0.0)
            for j, pattern in enumerate(ith_fiber):
                time += gaps[i][j]['duration']  # update time, gaps start first
                spike_times = [spike + time for spike in pattern] # the actual pattern times
                fiber += spike_times
                time  += pattern_params[i][j]['duration'] # update time, pattern duration
            s.append(fiber)
        stimulus.labels = labels
        stimulus.spike_trains = s
        stimulus.pattern_params = pattern_params
        stimulus.patterns = patterns
        stimulus.gaps = gaps
        stimulus.duration = time
        return stimulus


if __name__ == "__main__":
    pass
