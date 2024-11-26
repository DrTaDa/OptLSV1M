from mpi4py import MPI
from pyNN import nest
import nest

from mozaik.experiments import *
from mozaik.experiments.vision import *
from parameters import ParameterSet
from mozaik.controller import run_workflow

from model import SelfSustainedPushPull


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


mpi_comm = MPI.COMM_WORLD

nest.Install("stepcurrentmodule")
nest.Install("nestmlmodule")

data_store, model = run_workflow('SelfSustainedPushPull', SelfSustainedPushPull, create_experiments)
