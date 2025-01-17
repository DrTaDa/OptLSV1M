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
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight": [0.00013271],
        "sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight": [0.00028332],
        "sheets.l4_cortex_exc.AfferentConnection.base_weight": [0.00114099],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight": [0.00123946],
        "sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight": [0.00132614],
        "sheets.l4_cortex_inh.AfferentConnection.base_weight": [0.00145843],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection.base_weight": [6.31e-05],
        "sheets.l23_cortex_exc.L23ExcL23InhConnection.base_weight": [0.000158],
        "sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight": [0.00177875],
        "sheets.l23_cortex_exc.L23ExcL4ExcConnection.base_weight": [0.00030436],
        "sheets.l23_cortex_exc.L23ExcL4InhConnection.base_weight": [0.00021774],
        "sheets.l23_cortex_inh.L23InhL23ExcConnection.base_weight": [0.00304404],
        "sheets.l23_cortex_inh.L23InhL23InhConnection.base_weight": [0.00075802],
        "sheets.l23_cortex_inh.L4ExcL23InhConnection.base_weight": [0.00090406],
        "sheets.l4_cortex_inh.params.cell.params.tau_w": [106.93095163],
        "sheets.l4_cortex_inh.params.cell.params.a": [4.02107783],
        "sheets.l4_cortex_inh.params.cell.params.b": [47.7803816],
        "sheets.l23_cortex_inh.params.cell.params.tau_w": [97.30448993],
        "sheets.l23_cortex_inh.params.cell.params.a": [4.77814342],
        "sheets.l23_cortex_inh.params.cell.params.b": [47.97863771],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.U": [0.92729631],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.tau_rec": [24.45879866],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.short_term_plasticity.tau_rec": [33.30082499],
        "sheets.l4_cortex_exc.AfferentConnection.short_term_plasticity.tau_rec": [65.52261412],
    }
).run_parameter_search()
