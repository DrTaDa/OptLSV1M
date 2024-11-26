import os
import time
from datetime import datetime
import subprocess
import shutil
import psutil

from parameters import ParameterSet
from mozaik.storage.datastore import PickledDataStore
from mozaik.meta_workflow.parameter_search import ParameterSearch

from targets import *


def get_data_store(path):

    parameters = ParameterSet(
            {'root_directory': path,
             'store_stimuli': False}
    )

    try:
        data_store = PickledDataStore(
                load=True,
                parameters=parameters,
                replace=True
            )
    except:
        return None

    return data_store


class FitnessCalculator():

    def __init__(self, objectives):
        self.objectives = objectives

    def calculate_scores(self, data_store):
        return {t.name: t.calculate_score(data_store) for t in self.objectives}

    def calculate_values(self, data_store):
        return {t.name: t.calculate_value(data_store) for t in self.objectives}


class SlurmSequentialBackend():

    def __init__(self, num_threads, num_mpi, path_to_mozaik_env, slurm_options=None):
        self.num_threads = num_threads
        self.num_mpi = num_mpi
        self.path_to_mozaik_env = path_to_mozaik_env
        if slurm_options == None:
           self.slurm_options = []
        else:
           self.slurm_options = slurm_options 

    def execute_job(self, run_script, simulator_name, parameters_url, parameters, simulation_run_name):

        modified_parameters = []
        for k in parameters.keys():
            modified_parameters.append(k)
            modified_parameters.append(str(parameters[k]))

        from subprocess import Popen, PIPE
        sbatch_cmd = ['sbatch'] + self.slurm_options + ['-o', parameters['results_dir'][2:-2] + "/slurm-%j.out"]
        p = Popen(
            sbatch_cmd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            text=True
        )

        data = '\n'.join([
            '#!/bin/bash',
            '#SBATCH -n ' + str(self.num_mpi),
            '#SBATCH -c ' + str(self.num_threads),
            'source ' + str(self.path_to_mozaik_env),
            'cd ' + os.getcwd(),
            ' '.join(
                ["srun --mpi=pmix_v3 python", run_script, simulator_name, str(self.num_threads), parameters_url] + \
                modified_parameters + [simulation_run_name] + ['>'] + [parameters['results_dir'][1:-1] + '/OUTFILE' + str(time.time())]
            ),
        ])

        slurm_com = p.communicate(input=data)[0]
        p.stdin.close()

        return slurm_com


class Parameter():

    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bounds = [lower_bound, upper_bound]


class Evaluator(ParameterSearch):

    def __init__(self, fitness_calculator, params, run_script, simulator_name, parameters_url, backend, pynn_seed=0, timeout=3600):
        self.fitness_calculator = fitness_calculator
        self.objectives = self.fitness_calculator.objectives
        self.params = params
        self.param_names = [p.name for p in self.params]
        self.timeout = timeout
        self.run_script = run_script
        self.simulator_name = simulator_name
        self.parameters_url = parameters_url
        self.backend = backend
        self.pynn_seed = pynn_seed
        self.optimization_id = str(datetime.now().strftime('%Y%m%d-%H%M%S') + "_Optimization")
        os.mkdir(self.optimization_id)

    def check_evaluation_finished(self, slurm_id):
        slurm_status = str(subprocess.run(['squeue', '-j', slurm_id], stdout=subprocess.PIPE).stdout.decode('utf-8'))
        if slurm_id in slurm_status:
            return False
        return True

    def evaluate_data_store(self, data_store):
        return self.fitness_calculator.calculate_scores(data_store)

    def evaluate_parameters(self, parameters):

        process = psutil.Process()
        
        pid = os.getpid()

        # Create parameter set
        _params = {'pynn_seed': self.pynn_seed}
        _params.update(parameters)
        print(f"On pid {pid}. Launching job for parameters: ", _params)

        # Run the model
        _params['results_dir'] = '\"\'' + os.getcwd() + '/' + self.optimization_id + '/\'\"'
        slurm_com = self.backend.execute_job(self.run_script, self.simulator_name, self.parameters_url, _params, f'Opt_{pid}')
        slurm_id = slurm_com.replace("Submitted batch job ", "").rstrip()

        # Wait for run to finish
        print(f"On pid {pid}. Waiting for slurm job {slurm_id} to finish ...")
        t1 = time.time()
        while not self.check_evaluation_finished(slurm_id):
            time.sleep(20)
            if (t1 - time.time()) > self.timeout:
                return sum([obj.max_score for obj in self.objectives])
        print(f"On pid {pid}. Slurm job finished.")

        # Load data store
        for subf in glob.glob(f"./{self.optimization_id}/SelfSustainedPushPull_Opt_{pid}*"):
            data_store = get_data_store(subf)
            break

        print(f"On pid {pid}. Computing scores ...")
        score = self.evaluate_data_store(data_store)
        print(f"On pid {pid}. Done computing scores.")

        shutil.rmtree(subf) 

        print(f"On pid {pid}. Scores = {score}")
        return score

    def init_simulator_and_evaluate_with_lists(self, param_list=None, target='scores'):
        parameters_dict = {k.name: v for k, v in zip(self.params, param_list)}
        score_dict = self.evaluate_parameters(parameters_dict)
        print("Sum scores: ", numpy.sum([score_dict[t.name] for t in self.objectives]))
        return [score_dict[t.name] for t in self.objectives]

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

    return parameters


def define_fitness_calculator():
    objectives = define_objectives()
    return FitnessCalculator(objectives)


def define_evaluator(run_script, parameters_url, timeout):

    simulator_name = "nest"

    backend = SlurmSequentialBackend(
        num_threads=1,
        num_mpi=16,
        slurm_options=['--hint=nomultithread', '-N 1-1', '--exclude=w1,w2,w3,w4,w5,w6,w7,w8,w10'],
        path_to_mozaik_env='[PATH_TO_ENV]'
    )

    params = define_parameters()
    fitness_calculator = define_fitness_calculator()

    return Evaluator(fitness_calculator, params, run_script, simulator_name, parameters_url, backend, pynn_seed=1, timeout=timeout)
