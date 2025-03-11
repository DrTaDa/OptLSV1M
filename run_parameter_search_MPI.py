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
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight": [7.986e-05],
        "sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight": [0.0002754],
        "sheets.l4_cortex_exc.AfferentConnection.base_weight": [0.00201591],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight": [0.0022841],
        "sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight": [0.00378726],
        "sheets.l4_cortex_inh.AfferentConnection.base_weight": [0.00355374],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection.base_weight": [1e-06],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection_biais.base_weight": [0.00010534],
        "sheets.l23_cortex_exc.L23ExcL23InhConnection.base_weight": [0.0002799],
        "sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight": [0.00294981],
        "sheets.l23_cortex_exc.L23ExcL4ExcConnection.base_weight": [0.00027166],
        "sheets.l23_cortex_exc.L23ExcL4InhConnection.base_weight": [0.00026338],
        "sheets.l23_cortex_inh.L23InhL23ExcConnection.base_weight": [0.00312134],
        "sheets.l23_cortex_inh.L23InhL23InhConnection.base_weight": [0.00037342],
        "sheets.l23_cortex_inh.L4ExcL23InhConnection.base_weight": [0.00315083],
        "sheets.l4_cortex_inh.params.cell.params.tau_w": [164.30020387],
        "sheets.l4_cortex_inh.params.cell.params.a": [2.16834981],
        "sheets.l4_cortex_inh.params.cell.params.b": [87.63036467],
        "sheets.l23_cortex_inh.params.cell.params.tau_w": [193.51668541],
        "sheets.l23_cortex_inh.params.cell.params.a": [5.57286494],
        "sheets.l23_cortex_inh.params.cell.params.b": [61.77597969],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.U": [0.93407325],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.tau_rec": [63.44855425],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.short_term_plasticity.tau_rec": [53.42305657],
        "sheets.l4_cortex_exc.AfferentConnection.short_term_plasticity.tau_rec": [133.87259811],
    }
).run_parameter_search()
