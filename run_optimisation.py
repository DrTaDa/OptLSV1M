import multiprocessing as mp
from bluepyopt.deapext.optimisationsCMA import DEAPOptimisationCMA
from evaluator import define_evaluator

offspring_size = 12                 # Size of the population used by the optimizer
timeout = 4000                      # Hard cut-off for the evaluation of an individual (in seconds)
optimiser_seed = 36                  # random seed for the optimiser
optimiser_sigma = 0.4               # width of the search at the first generation of the optimisation, default: 0.4
optimiser_centroid = [0.00017231206360188508, 0.00026298300409550535, 0.0016793226341963673, 0.0015751607007732005, 0.0016999999993999998, 0.0024041080676130725, 7.753183982743447e-06, 0.00019637189182125067, 0.0020462807161389395, 0.0002866670645249675, 0.0001819611986992853, 0.0021409484254594794, 0.0005065733746189931, 0.0009413578992548964, 177.08156935468776, 2.510994895071973, 47.342716815999395, 138.28441625017956, 4.614509433730632, 41.36709644943765, 0.8982124828697061, 14.652718635458825, 39.146715382323286, 119.48306530300536] # List (optional): starting solution
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
    print("RESTARTING FROM CHECKPOINT")
    cp_filename = "./20250224-224804_Optimization/opt_check.pkl"

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
    #centroids=[optimiser_centroid]
)

optimizer.run(
    max_ngen=max_ngen,
    cp_filename=cp_filename,
    cp_frequency=1,
    continue_cp=continue_cp,
)
