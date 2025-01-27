from mpi4py import MPI
from model import SelfSustainedPushPull

from mozaik.controller import run_workflow
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.microstimulation import IntraCorticalMicroStimulationStimulus
from parameters import ParameterSet

from pyNN import nest
import nest


def create_experiments_stc(model):

    return [

        # Spontaneous Activity
        NoStimulation(model, ParameterSet({'duration': 20000})),

        # Size Tuning
        MeasureSizeTuning(model, ParameterSet(
            {'num_sizes': 16, 'max_size': 5.0, 'log_spacing': False, 'orientations': [0],
             'positions': [(0,0)], 'spatial_frequency': 0.8, 'temporal_frequency': 2,
             'grating_duration': 2*143*7, 'contrasts': [10, 100], 'num_trials': 2, 'shuffle_stimuli': True
            })),
    ]


mpi_comm = MPI.COMM_WORLD

nest.Install("stepcurrentmodule")
nest.Install("nestmlmodule")

data_store, model = run_workflow('SelfSustainedPushPull', SelfSustainedPushPull, create_experiments_stc)
