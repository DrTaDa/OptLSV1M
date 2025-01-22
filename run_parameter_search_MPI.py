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
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight": [0.00011216],
        "sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight": [0.00031048],
        "sheets.l4_cortex_exc.AfferentConnection.base_weight": [0.0013222],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight": [0.00110694],
        "sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight": [0.00109669],
        "sheets.l4_cortex_inh.AfferentConnection.base_weight": [0.00181861],
        "sheets.l23_cortex_exc.L23ExcL23ExcConnection.base_weight": [2.543e-05],
        "sheets.l23_cortex_exc.L23ExcL23InhConnection.base_weight": [0.00014197],
        "sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight": [0.00237137],
        "sheets.l23_cortex_exc.L23ExcL4ExcConnection.base_weight": [0.00029511],
        "sheets.l23_cortex_exc.L23ExcL4InhConnection.base_weight": [0.000295],
        "sheets.l23_cortex_inh.L23InhL23ExcConnection.base_weight": [0.00387448],
        "sheets.l23_cortex_inh.L23InhL23InhConnection.base_weight": [0.00100635],
        "sheets.l23_cortex_inh.L4ExcL23InhConnection.base_weight": [0.00101676],
        "sheets.l4_cortex_inh.params.cell.params.tau_w": [127.3079734],
        "sheets.l4_cortex_inh.params.cell.params.a": [3.03794152],
        "sheets.l4_cortex_inh.params.cell.params.b": [51.75440485],
        "sheets.l23_cortex_inh.params.cell.params.tau_w": [86.07210084],
        "sheets.l23_cortex_inh.params.cell.params.a": [4.01996791],
        "sheets.l23_cortex_inh.params.cell.params.b": [52.81390677],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.U": [1.0],
        "sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.tau_rec": [28.74049428],
        "sheets.l4_cortex_inh.L4InhL4ExcConnection.short_term_plasticity.tau_rec": [45.8877127],
        "sheets.l4_cortex_exc.AfferentConnection.short_term_plasticity.tau_rec": [134.89393372],
    }
).run_parameter_search()
