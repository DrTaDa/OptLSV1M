import multiprocessing as mp
from bluepyopt.deapext.optimisationsCMA import DEAPOptimisationCMA
from evaluator import define_evaluator

offspring_size = 12                 # Size of the population used by the optimizer
timeout = 4000                      # Hard cut-off for the evaluation of an individual (in seconds)
optimiser_seed = 666                  # random seed for the optimiser
optimiser_sigma = 0.2               # width of the search at the first generation of the optimisation, default: 0.4
optimiser_centroid = [8.307350226797982e-05, 0.0002852393907629799, 0.0019074007889368942, 0.002559973038428656, 0.003361197544247738, 0.003264451653058492, 6.1919434681979445e-06, 0.00012010447880314855, 0.0002662616437552525, 0.0032355307510103194, 0.0002410492201142536, 0.00023018498983994344, 0.002811734606022205, 0.00031761177483506317, 0.00299999999925, 177.13203394831413, 2.3522606763870484, 90.27087629563451, 204.88982867820556, 5.999999998, 50.06895499004019, 0.9666246212745299, 65.46773298352808, 66.50211711546561, 124.56722392199246] # List (optional): starting solution
max_ngen = 100                      # Maximum number of generation of the optimiser
continue_cp = False                 # Should the optimisation resume from the informed checkpoint file
run_script = "run_experiment.py"    # Path to the Mozaik run script to use for evaluation
parameters_url = "param_split/defaults"   # Path the Mozaik parameters
config_optimisation = "./param_split/config_optimisation" # path to the optimisation parameters and targets

evaluator = define_evaluator(
    run_script=run_script,
    parameters_url=parameters_url,
    timeout = timeout,
    config_optimisation=config_optimisation
)

if not continue_cp:
    cp_filename = f"./{evaluator.optimization_id}/opt_check.pkl"       # Path to the checkpoint file of the optimisation
else:
    print("RESUMING FROM CHECKPOINT")
    cp_filename = ""

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
