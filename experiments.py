from mozaik.experiments import *
from mozaik.experiments.vision import *
from parameters import ParameterSet


def create_experiments(model):

    spont = NoStimulation(model, ParameterSet({"duration": 5000}))

    orientation_tuning = MeasureOrientationTuningFullfield(model, ParameterSet(
    {
        'num_orientations': 2,
        'spatial_frequency': 0.8,
        'temporal_frequency': 2,
        'grating_duration': 2000,
        'contrasts': [10, 100],
        'num_trials': 1,
        'shuffle_stimuli': True
    }))

    return [spont, orientation_tuning]
