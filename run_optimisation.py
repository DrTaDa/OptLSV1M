import multiprocessing as mp
from bluepyopt.deapext.optimisationsCMA import DEAPOptimisationCMA
from evaluator import define_evaluator


offspring_size = 12                 # Size of the population used by the optimizer
timeout = 4000                      # Hard cut-off for the evaluation of an individual (in seconds)
optimiser_seed = 2                  # random seed for the optimiser
optimiser_sigma = 0.2               # width of the search at the first generation of the optimisation, default: 0.4
optimiser_centroid = [0.000134817438063558, 0.00020827084594156843, 0.002053775324509431, 0.0013179966487546454, 0.001963397357677063,
                      0.0025121112247027517, 4.5477873270338303e-05, 0.00032000000004000004, 0.0017032297441187262, 0.00022910037375134277,
                      0.0002193777752123396, 0.002329464255291176, 0.0005971521377206546, 0.0013495971064910305, 184.64678354679643,
                      2.5499984091679835, 42.784347763541824, 71.35421043803875, 2.9149154194508498, 99.31477756298135]           # List (optional): starting solution
max_ngen = 500                      # Maximum number of generation of the optimiser
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
