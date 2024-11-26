import multiprocessing as mp
from bluepyopt.deapext.optimisationsCMA import DEAPOptimisationCMA

def get_map_function(offspring_size):
    pool = mp.Pool(processes=offspring_size)
    return pool.map


def define_objectives():

    objectives = [
        SpontActivityTarget(name="SpontActivity_V1_Exc_L4", target_value=1.4, sheet_name="V1_Exc_L4", norm=0.2, max_score=10),
        SpontActivityTarget(name="SpontActivity_V1_Inh_L4", target_value=7.6, sheet_name="V1_Inh_L4", norm=0.4, max_score=10),
        SpontActivityTarget(name="SpontActivity_V1_Exc_L2/3", target_value=2, sheet_name="V1_Exc_L2/3", norm=0.2, max_score=10),
        SpontActivityTarget(name="SpontActivity_V1_Inh_L2/3", target_value=4.7, sheet_name="V1_Inh_L2/3", norm=0.4, max_score=10)
    ]

    sheet_names = ["V1_Exc_L4", "V1_Inh_L4", "V1_Exc_L2/3", "V1_Inh_L2/3"]
    for sheet_name in sheet_names:
        objectives += [
            IrregularityTarget(name=f"Irregularity_{sheet_name}", target_value=1., sheet_name=sheet_name, norm=0.1, max_score=10),
            SynchronyTarget(name=f"Synchrony_{sheet_name}", target_value=0., sheet_name=sheet_name, norm=0.01, max_score=10),
            OrientationTuningPreferenceTarget(
                name=f"OrientationTuningPreference_{sheet_name}", target_value=5., sheet_name=sheet_name, norm=0.5, max_score=10
            ),
            OrientationTuningOrthoHighTarget(
                name=f"OrientationTuningOrthoHigh_{sheet_name}", target_value=0., sheet_name=sheet_name, norm=0.3, max_score=10
            ),
            OrientationTuningOrthoLowTarget(
                name=f"OrientationTuningOrthoLow_{sheet_name}", target_value=0., sheet_name=sheet_name, norm=0.3, max_score=10
            )
        ]

    """with open(f"ICMS_exp_data.pickle", 'rb') as fp:
        exp_data = pickle.load(fp)

    base_norms = [0.75, 150, 0.75]
    for idx, _class in enumerate([ICMSExcAmplitudeTarget, ICMSInhDurationTarget, ICMSInhDepthTarget]):
        objectives.append(
            _class(
                _class.__name__,
                target_values=list(exp_data[41][idx][1]),
                value_distances=list(exp_data[41][idx][0]), 
                sheet_names=sheet_names,
                active_electrode=45,
                stimulation_frequency=0.5,
                amplitude=41000.0,
                norms=[base_norms[idx] for i in range(len(list(exp_data[41][idx][0])))],
                max_score=10
            )
        )"""

    return objectives


def define_parameters():

    parameters = [
        Parameter(name='sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight', lower_bound=0.00014, upper_bound=0.00024),
        Parameter(name='sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight', lower_bound=0.00018, upper_bound=0.00027),
        Parameter(name='sheets.l4_cortex_exc.AfferentConnection.base_weight', lower_bound=0.0016, upper_bound=0.0025),

        Parameter(name='sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight', lower_bound=0.0008, upper_bound=0.0018),
        Parameter(name='sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight', lower_bound=0.0011, upper_bound=0.0022),
        Parameter(name='sheets.l4_cortex_inh.AfferentConnection.base_weight', lower_bound=0.0022, upper_bound=0.0033),

        Parameter(name='sheets.l23_cortex_exc.L23ExcL23ExcConnection.base_weight', lower_bound=0.00007, upper_bound=0.00019),
        Parameter(name='sheets.l23_cortex_exc.L23ExcL23InhConnection.base_weight', lower_bound=0.00031, upper_bound=0.00039),
        Parameter(name='sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight', lower_bound=0.0008, upper_bound=0.0014),
        Parameter(name='sheets.l23_cortex_exc.L23ExcL4ExcConnection.base_weight', lower_bound=0.00018, upper_bound=0.00025),
        Parameter(name='sheets.l23_cortex_exc.L23ExcL4InhConnection.base_weight', lower_bound=0.00020, upper_bound=0.00029),

        Parameter(name='sheets.l23_cortex_inh.L23InhL23ExcConnection.base_weight', lower_bound=0.0014, upper_bound=0.0024),
        Parameter(name='sheets.l23_cortex_inh.L23InhL23InhConnection.base_weight', lower_bound=0.0003, upper_bound=0.0014),
        Parameter(name='sheets.l23_cortex_inh.L4ExcL23InhConnection.base_weight', lower_bound=0.0005, upper_bound=0.0015),

        Parameter(name='sheets.l4_cortex_exc.params.cell.params.tau_syn_exc', lower_bound=1.2, upper_bound=1.8),
        Parameter(name='sheets.l4_cortex_exc.params.cell.params.tau_syn_inh', lower_bound=3.9, upper_bound=4.5),
    ]

    """parameters = [
        Parameter(name='sheets.l4_cortex_exc.params.cell.params.tau_w', lower_bound=50, upper_bound=200),
        Parameter(name='sheets.l4_cortex_exc.params.cell.params.a', lower_bound=1, upper_bound=10),
        Parameter(name='sheets.l4_cortex_exc.params.cell.params.b', lower_bound=40, upper_bound=120),

        Parameter(name='experiment_scaler_a', lower_bound=90, upper_bound=110),
        Parameter(name='experiment_scaler_b', lower_bound=0.75, upper_bound=0.92),
        Parameter(name='experiment_scaler_c', lower_bound=-1, upper_bound=1.0),
    ]"""

    return parameters


def define_evaluator(run_script, parameters_url, timeout):

    simulator_name = "nest"

    backend = SlurmSequentialBackend(
        num_threads=1,
        num_mpi=16,
        slurm_options=['--hint=nomultithread', '-N 1-1', '--exclude=w1,w2,w3,w4,w5,w6,w7,w8,w10'],
        path_to_mozaik_env='[PATH_TO_ENV]'
    )

    objectives = define_objectives()
    params = define_parameters()
    fitness_calculator = FitnessCalculator(objectives)

    return Evaluator(fitness_calculator, params, run_script, simulator_name, parameters_url, backend, pynn_seed=1, timeout=timeout)


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
