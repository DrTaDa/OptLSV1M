import multiprocessing as mp
from bluepyopt.deapext.optimisationsCMA import DEAPOptimisationCMA
from evaluator import define_evaluator

offspring_size = 12                 # Size of the population used by the optimizer
timeout = 4000                      # Hard cut-off for the evaluation of an individual (in seconds)
optimiser_seed = 3                  # random seed for the optimiser
optimiser_sigma = 0.3               # width of the search at the first generation of the optimisation, default: 0.4
optimiser_centroid = [0.00013271, 0.00028332, 0.00114099, 0.00123946, 0.00132614, 0.00145843, 6.31e-05, 0.000158, 0.00177875, 0.00030436, 0.00021774, 0.00304404, 0.00075802, 0.00090406, 106.93095163, 4.02107783, 47.7803816, 97.30448993, 4.77814342, 47.97863771, 0.92729631, 24.45879866, 33.30082499, 65.52261412]           # List (optional): starting solution
max_ngen = 200                      # Maximum number of generation of the optimiser
continue_cp = False                 # Should the optimisation resume from the informed checkpoint file
run_script = "run_experiment.py"    # Path to the Mozaik run script to use for evaluation
parameters_url = "param/defaults"   # Path the Mozaik parameters
config_optimisation = "./param/config_optimisation" # path to the optimisation parameters and targets

evaluator = define_evaluator(
    run_script=run_script,
    parameters_url=parameters_url,
    timeout = timeout,
    config_optimisation=config_optimisation
)

if not continue_cp:
    cp_filename = f"./{evaluator.optimization_id}/opt_check.pkl"       # Path to the checkpoint file of the optimisation
else:
    raise Exception("Please inform the path to the last chckpointfile here !")
    cp_filename = None # <------

map_function = mp.Pool(processes=offspring_size).map

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
