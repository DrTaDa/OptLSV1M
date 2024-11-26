import multiprocessing as mp
from bluepyopt.deapext.optimisationsCMA import DEAPOptimisationCMA
from evaluator import define_evaluator


def get_map_function(offspring_size):
    pool = mp.Pool(processes=offspring_size)
    return pool.map


offspring_size = 12                 # Size of the population used by the optimizer
timeout = 4000                      # Hard cut-off for the evaluation of an individual (in seconds)
optimiser_seed = 2                  # random seed for the optimiser
optimiser_sigma = 0.4               # width of the search at the first generation of the optimisation
optimiser_centroid = None           # List (optional): starting solution
max_ngen = 500                      # Maximum number of generation of the optimiser
cp_filename = "opt_check.pkl"       # Path to the checkpoint file of the optimisation
continue_cp = False                 # Should the optimisation resume from the informed checkpoint file
run_script = "run_experiment.py"    # Path to the Mozaik run script to use for evaluation
parameters_url = "param/defaults"   # Path the Mozaik parameters

evaluator = define_evaluator(
    run_script=run_script,
    parameters_url=parameters_url,
    timeout = timeout
)

map_function = get_map_function(offspring_size)

optimizer = DEAPOptimisationCMA(
    evaluator=evaluator,
    use_scoop=False,
    seed=optimiser_seed,
    offspring_size=offspring_size,
    map_function=map_function,
    selector_name="single_objective",
    use_stagnation_criterion=False,
    sigma=optimiser_sigma,
    centroids=[optimiser_centroid]
)

optimizer.run(
    max_ngen=max_ngen,
    cp_filename=cp_filename,
    cp_frequency=1,
    continue_cp=continue_cp,
)
