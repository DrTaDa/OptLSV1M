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
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight": [0.00014628],
        "sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight": [0.00028055],
        "sheets.l4_cortex_exc.AfferentConnection.base_weight": [0.00130547],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight": [0.00118972],
        "sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight": [0.00132694],
        "sheets.l4_cortex_inh.AfferentConnection.base_weight": [0.00162447],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection.base_weight": [3.811e-05],
        "sheets.l23_cortex_exc.L23ExcL23InhConnection.base_weight": [0.00018623],
        "sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight": [0.00172112],
        "sheets.l23_cortex_exc.L23ExcL4ExcConnection.base_weight": [0.0003491],
        "sheets.l23_cortex_exc.L23ExcL4InhConnection.base_weight": [0.00022506],
        "sheets.l23_cortex_inh.L23InhL23ExcConnection.base_weight": [0.00347213],
        "sheets.l23_cortex_inh.L23InhL23InhConnection.base_weight": [0.00058415],
        "sheets.l23_cortex_inh.L4ExcL23InhConnection.base_weight": [0.00085439],
        "sheets.l4_cortex_inh.params.cell.params.tau_w": [79.28592],
        "sheets.l4_cortex_inh.params.cell.params.a": [3.27125966],
        "sheets.l4_cortex_inh.params.cell.params.b": [51.56410977],
        "sheets.l23_cortex_inh.params.cell.params.tau_w": [136.91716338],
        "sheets.l23_cortex_inh.params.cell.params.a": [4.92307142],
        "sheets.l23_cortex_inh.params.cell.params.b": [67.67296789],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.U": [0.95544135],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.tau_rec": [27.33935244],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.short_term_plasticity.tau_rec": [22.31884732],
        "sheets.l4_cortex_exc.AfferentConnection.short_term_plasticity.tau_rec": [88.03835315],
    }
).run_parameter_search()
