from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackend

CombinationParameterSearch(
    SlurmSequentialBackend(
        num_threads=1,
        num_mpi=16,
        slurm_options=['--hint=nomultithread', '-N 1-1', '--exclude=w1,w2,w3,w4,w5,w6,w7,w8,w10,w12,w16'],
        path_to_mozaik_env='[PATH_TO_ENV]'
    ),
    {
        "pynn_seed": [1],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight": [6.65e-05],
        "sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight": [0.00022712],
        "sheets.l4_cortex_exc.AfferentConnection.base_weight": [0.0018434],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight": [0.00125752],
        "sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight": [0.00288719],
        "sheets.l4_cortex_inh.AfferentConnection.base_weight": [0.00268427],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection.base_weight": [3.476e-05],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection_biais.base_weight": [0.00021501],
        "sheets.l23_cortex_exc.L23ExcL23InhConnection.base_weight": [0.00024486],
        "sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight": [0.0049995],
        "sheets.l23_cortex_exc.L23ExcL4ExcConnection.base_weight": [0.00021577],
        "sheets.l23_cortex_exc.L23ExcL4InhConnection.base_weight": [0.00027882],
        "sheets.l23_cortex_inh.L23InhL23ExcConnection.base_weight": [0.00312415],
        "sheets.l23_cortex_inh.L23InhL23InhConnection.base_weight": [0.00061016],
        "sheets.l23_cortex_inh.L4ExcL23InhConnection.base_weight": [0.00434925],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.U": [0.69506409],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.tau_rec": [124.05860042],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.short_term_plasticity.tau_rec": [39.0711949],
        "sheets.l4_cortex_exc.AfferentConnection.short_term_plasticity.tau_rec": [126.58986454],
        "sheets.l23_cortex_exc.percentage_bias": [0.20524251],
    }
).run_parameter_search()
