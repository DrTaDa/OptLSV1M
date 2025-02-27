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
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight": [0.00012371],
        "sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight": [0.0002971],
        "sheets.l4_cortex_exc.AfferentConnection.base_weight": [0.00213231],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight": [0.00201003],
        "sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight": [0.00277724],
        "sheets.l4_cortex_inh.AfferentConnection.base_weight": [0.00249477],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection.base_weight": [6.41e-06],
        "sheets.l23_cortex_exc.L23ExcL23InhConnection.base_weight": [0.00016059],
        "sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight": [0.00161867],
        "sheets.l23_cortex_exc.L23ExcL4ExcConnection.base_weight": [0.00036719],
        "sheets.l23_cortex_exc.L23ExcL4InhConnection.base_weight": [0.00020311],
        "sheets.l23_cortex_inh.L23InhL23ExcConnection.base_weight": [0.00249985],
        "sheets.l23_cortex_inh.L23InhL23InhConnection.base_weight": [0.0007334],
        "sheets.l23_cortex_inh.L4ExcL23InhConnection.base_weight": [0.00110945],
        "sheets.l4_cortex_inh.params.cell.params.tau_w": [76.17256024],
        "sheets.l4_cortex_inh.params.cell.params.a": [3.56250822],
        "sheets.l4_cortex_inh.params.cell.params.b": [48.06020686],
        "sheets.l23_cortex_inh.params.cell.params.tau_w": [99.31587759],
        "sheets.l23_cortex_inh.params.cell.params.a": [5.57301634],
        "sheets.l23_cortex_inh.params.cell.params.b": [50.22330442],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.U": [0.91720166],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.tau_rec": [136.4720665],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.short_term_plasticity.tau_rec": [45.24974672],
        "sheets.l4_cortex_exc.AfferentConnection.short_term_plasticity.tau_rec": [138.0648628],
    }
).run_parameter_search()
