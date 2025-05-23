import multiprocessing as mp
import json
from bluepyopt.deapext.optimisationsCMA import DEAPOptimisationCMA
from evaluator import define_evaluator

offspring_size = 16                 # Size of the population used by the optimizer
timeout = 4000                      # Hard cut-off for the evaluation of an individual (in seconds)
optimiser_seed = 1                  # random seed for the optimiser
optimiser_sigma = 0.3               # width of the search at the first generation of the optimisation, default: 0.4
optimiser_centroid = [0.00010029373386100313, 0.0003380726474680518, 0.002528845813842111, 0.0026194147314384363, 0.002821211428064238, 0.0029799187297856176, 0.00020051186258592256, 0.00020092145843992628, 0.00036501712861841223, 0.004321634769040701, 0.00024146793167836338, 0.00020389830942880126, 0.0032665329747964917, 0.0002623739074357583, 0.004491625662455231, 0.5396194306958887, 137.25104949924173, 70.35565834291869, 134.6202010272549, 1.5791028899801998] # List (optional): starting solution

# Start from middle of bounds
if False:
    with open("./param_split_1split/config_optimisation") as f:
        opt_confi = json.load(f)
    optimiser_centroid = [((v[0] + v[1]) / 2) for v in opt_confi["parameters"].values()]
    print(optimiser_centroid)

max_ngen = 100                      # Maximum number of generation of the optimiser
continue_cp = False                 # Should the optimisation resume from the informed checkpoint file
run_script = "run_experiment.py"    # Path to the Mozaik run script to use for evaluation
parameters_url = "param_split_1split/defaults"   # Path the Mozaik parameters
config_optimisation = "./param_split_1split/config_optimisation" # path to the optimisation parameters and targets

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
    cp_filename = "./20250408-083912_Optimization/opt_check.pkl"

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
