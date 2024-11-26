from mpi4py import MPI
from model import SelfSustainedPushPull
from experiments import create_experiments
from mozaik.controller import run_workflow
from pyNN import nest
import nest

mpi_comm = MPI.COMM_WORLD

nest.Install("stepcurrentmodule")
nest.Install("nestmlmodule")

data_store, model = run_workflow('SelfSustainedPushPull', SelfSustainedPushPull, create_experiments)
