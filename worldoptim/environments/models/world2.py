# This model is an implementation of:
# World 2 (complete with article reference and pyworld2 reference

from worldoptim.environments.models.base_model import BaseModel
from worldoptim.environments.models.utils.pyworld import *
from worldoptim.utils import *
from scipy.interpolate import interp1d

class World2(BaseModel):
    def __init__(self,
                 stochastic=False,
                 noise_params=0.1,
                 range_delay=(0, 21),
                 dt=1,
                 json_tables=None,
                 json_switches=None,
                 year_min=1900,
                 year_max=2100
                 ):
        """
        Implementation of the SEIRAH model from Prague et al., 2020:
        Population modeling of early COVID-19 epidemic dynamics in French regions and estimation of the lockdown impact on infection rate.

        Parameters
        ----------
        region: str
                Region identifier.
        stochastic: bool
                    Whether to use stochastic models or not.
        noise_params: float
                      Normally distributed parameters have an stdev of 'noise_params' x their mean.

        Attributes
        ---------
        region
        stochastic
        noise_params
        """
        self.json_tables = json_tables
        self.json_switches = json_switches
        self.year_min = year_min
        self.year_max = year_max
        self.dt = dt
        self.stochastic = stochastic
        self.noise = noise_params

        # Initialize distributions of parameters and initial conditions for all regions
        self.define_params_and_initial_state_distributions()

        # stuff to track
        self.labels_for_agent = ['']
        self.stocks = ['P', 'POL', 'CI', 'NR', 'CIAF']
        self.rates = ['QL', 'DR', 'BR', 'NRUR', 'CIG', 'CID', 'POLG', 'POLA']
        self.control_variables = ['NRUN', 'POLN', 'CIGN', 'BRN', 'FC']
        self.control_variables_ranges = ((0.2, 1), (0.2, 1), (0.2, 1), (0.2, 1), (0.2, 2))  # to think about
        self.state_labels_for_agent = self.stocks + self.rates + self.control_variables

        # Sample initial conditions and initial model parameters
        # just to get their labels
        self._sample_model_params()
        self._sample_initial_state()
        super().__init__(internal_states_labels=sorted(self.initial_state.keys()),
                         internal_params_labels=sorted(self.initial_internal_params.keys()),
                         stochastic=stochastic,
                         range_delay=range_delay)

    def update_json_tables(self, json_tables):
        self.json_tables = json_tables

    # def update_json_switches(self, json_switches):
    #     self.json_switches = json_switches

    def define_params_and_initial_state_distributions(self):
        """
        Extract and define distributions of parameters for all French regions
        """


        self._all_internal_params_distribs = dict(LA=DiracDist(params=135e6, stochastic=self.stochastic),
                                                  PDN=DiracDist(params=26.5, stochastic=self.stochastic),
                                                  CIAFN=DiracDist(params=0.3, stochastic=self.stochastic),
                                                  ECIRN=DiracDist(params=1, stochastic=self.stochastic),
                                                  CIAFT=DiracDist(params=15, stochastic=self.stochastic),
                                                  POLS=DiracDist(params=3.6e9, stochastic=self.stochastic),
                                                  FN=DiracDist(params=1, stochastic=self.stochastic),
                                                  QLS=DiracDist(params=1, stochastic=self.stochastic),
                                                  BRN=DiracDist(params=0.04, stochastic=self.stochastic),
                                                  NRUN=DiracDist(params=1, stochastic=self.stochastic),
                                                  DRN=DiracDist(params=0.028, stochastic=self.stochastic),
                                                  FC=DiracDist(params=1, stochastic=self.stochastic),
                                                  CIGN=DiracDist(params=0.05, stochastic=self.stochastic),
                                                  CIDN=DiracDist(params=0.025, stochastic=self.stochastic),
                                                  POLN=DiracDist(params=1, stochastic=self.stochastic))

        self._all_initial_state_distribs = dict(P=DiracDist(params=1.65e9, stochastic=self.stochastic),
                                                NR=DiracDist(params=900e9, stochastic=self.stochastic),
                                                CI=DiracDist(params=0.4e9, stochastic=self.stochastic),
                                                POL=DiracDist(params=0.2e9, stochastic=self.stochastic),
                                                CIAF=DiracDist(params=0.2, stochastic=self.stochastic),
                                                BR=None,
                                                DR=None,
                                                CR=None,
                                                NRUR=DiracDist(params=0., stochastic=self.stochastic),
                                                NRFR=None,
                                                CIR=None,
                                                CIG=None,
                                                CID=None,
                                                CIRA=None,
                                                MSL=None,
                                                ECIR=None,
                                                FR=None,
                                                POLR=None,
                                                POLG=None,
                                                POLA=None,
                                                QL=None,
                                                )





    def _sample_initial_state(self):
        """
        Samples an initial model state from its distribution (Dirac distributions if self.stochastic is False).


        """
        # sample initial states
        self.initial_state = dict()
        for k in self._all_initial_state_distribs.keys():
            if self._all_initial_state_distribs[k] != None:
                self.initial_state[k] = self._all_initial_state_distribs[k].sample()
            else:
                self.initial_state[k] = None  # to complete with formulas below

        # complete other initial states with formulas from pyworld 2 'set_state_variables'

        # initialize population
        self.initial_state['BR'] = np.nan
        self.initial_state['DR'] = np.nan

        # initialize natural resources
        self.initial_state['NRFR'] = self.initial_state['NR'] / self.initial_state['NR']

        # initialize capital investment
        self.initial_state['CR'] = self.initial_state['P'] / (self.initial_internal_params['LA'] * self.initial_internal_params['PDN'])
        self.initial_state['CIR'] = self.initial_state['CI'] / self.initial_state['P']

        # initialize pollution
        self.initial_state['POLG'] = self.initial_state['P'] * self.initial_internal_params['POLN'] * self.f['POLCM'](self.initial_state['CIR'])
        self.initial_state['POLR'] = self.initial_state['POL'] / self.initial_internal_params['POLS']
        self.initial_state['POLA'] = self.initial_state['POL'] / self.f['POLAT'](self.initial_state['POLR'])

        # initialize capital investment in agriculture fraction
        self.initial_state['CID'] = np.nan
        self.initial_state['CIG'] = np.nan

        # initialize other intermediary variables
        self.initial_state['CIRA'] = self.initial_state['CIR'] * self.initial_state['CIAF'] / self.initial_internal_params['CIAFN']
        self.initial_state['FR'] = (self.f['FPCI'](self.initial_state['CIRA']) * self.f['FCM'](self.initial_state['CR'])) * \
                                   (self.f['FPM'](self.initial_state['POLR']) * self.initial_internal_params['FC']) / self.initial_internal_params['FN']

        self.initial_state['ECIR'] = self.initial_state['CIR'] * (1 - self.initial_state['CIAF']) * self.f['NREM'](self.initial_state['NRFR']) / \
                                     (1 - self.initial_internal_params['CIAFN'])
        self.initial_state['MSL'] = self.initial_state['ECIR'] / self.initial_internal_params['ECIRN']
        self.initial_state['QL'] = np.nan

        # add control variables in the state as they can now change (due to external control)
        for k in self.control_variables:
            self.initial_state[k] = self.initial_internal_params[k]

    def _sample_model_params(self):
        """
        Samples parameters of the model from their distribution (Dirac distributions if self.stochastic is False).

        """
        self.initial_internal_params = dict()
        for k in self._all_internal_params_distribs.keys():
            self.initial_internal_params[k] = self._all_internal_params_distribs[k].sample()

        self.complete_param_distrib_with_table_params(self.json_tables)
        # self.complete_param_distrib_with_switch_params(self.json_switches)
        self._reset_model_params()
        self.update_funcs()

    # def update_model_params(self, new_params):
    #     assert sorted(self.current_internal_params.keys()) == sorted(new_params.keys())
    #     self.current_internal_params = new_params
    #     self.update_funcs()


    def run_one_step(self, state):

        new_state = dict()
        # update population state variable
        new_state['BR'] = state['P'] * self.current_internal_params['BRN'] * self.f['BRMM'](state['MSL']) * self.f['BRCM'](state['CR']) * \
                          self.f['BRFM'](state['FR']) * self.f['BRPM'](state['POLR'])
        new_state['DR'] = state['P'] * self.current_internal_params['DRN'] * self.f['DRMM'](state['MSL']) * self.f['DRPM'](state['POLR']) * \
                          self.f['DRFM'](state['FR']) * self.f['DRCM'](state['CR'])
        new_state['P'] = state['P'] + (new_state['BR'] - new_state['DR']) * self.dt

        # update natural resources state variable
        new_state['NRUR'] = (state['P'] * self.current_internal_params['NRUN']) * self.f['NRMM'](state['MSL'])
        new_state['NR'] = state['NR'] - new_state['NRUR'] * self.dt
        new_state['NRFR'] = new_state['NR'] / self.initial_state['NR']

        # update capital investment state variable
        new_state['CID'] = state['CI'] * self.current_internal_params['CIDN']
        new_state['CIG'] = (state['P'] * self.f['CIM'](state['MSL'])) * self.current_internal_params['CIGN']
        new_state['CI'] = state['CI'] + self.dt * (new_state['CIG'] - new_state['CID'])
        new_state['CR'] = new_state['P'] / (self.current_internal_params['LA'] * self.current_internal_params['PDN'])
        new_state['CIR'] = new_state['CI'] / new_state['P']

        # update pollution state variable
        new_state['POLG'] = state['P'] * self.current_internal_params['POLN'] * self.f['POLCM'](state['CIR'])
        new_state['POLA'] = state['POL'] / self.f['POLAT'](state['POLR'])
        new_state['POL'] = state['POL'] + (new_state['POLG'] - new_state['POLA']) * self.dt
        new_state['POLR'] = new_state['POL'] / self.current_internal_params['POLS']

        # update capital investment in agriculutre fraction state variable
        new_state['CIAF'] = state['CIAF'] + (self.f['CFIFR'](state['FR']) * self.f['CIQR'](self.f['QLM'](state['MSL']) / self.f['QLF'](state['FR'])) - state['CIAF']) * \
                            (self.dt / self.current_internal_params['CIAFT'])


        # update other intermediary variables
        new_state['CIRA'] = new_state['CIR'] * new_state['CIAF'] / self.current_internal_params['CIAFN']
        new_state['FR'] = (self.f['FCM'](new_state['CR']) * self.f['FPCI'](new_state['CIRA']) * self.f['FPM'](new_state['POLR']) * self.current_internal_params['FC']) / \
                          self.current_internal_params['FN']
        new_state['ECIR'] = (new_state['CIR'] * (1 - new_state['CIAF']) * self.f['NREM'](new_state['NRFR'])) / (1 - self.current_internal_params['CIAFN'])
        new_state['MSL'] = new_state['ECIR'] / self.current_internal_params['ECIRN']
        new_state['QL'] = (self.current_internal_params['QLS'] * self.f['QLM'](new_state['MSL']) * self.f['QLC'](new_state['CR']) * self.f['QLF'](new_state['FR'])) * \
                          self.f['QLP'](new_state['POLR'])

        # add control variables in the state as they can now change (due to external control)
        for k in self.control_variables:
            new_state[k] = self.current_internal_params[k]

        self.current_state = new_state.copy()
        return new_state


    def run_n_steps(self, current_state=None, n=1, labelled_states=False, time=None):
        """
        Runs the model for n steps

        Parameters
        ----------
        current_state: 1D nd.array
                       Current model state.
        n: int
           Number of steps the model should be run for.

        labelled_states: bool
                         Whether the result should be a dict with state labels or a nd array.

        Returns
        -------
        dict or 2D nd.array
            Returns a dict if labelled_states is True, where keys are state labels.
            Returns an array of size (n, n_states) of the last n model states.

        """

        states = [self.current_state.copy()]
        for i in range(n):
            new_state = self.run_one_step(states[-1])
            states.append(new_state)

        vector_states = np.array([self.get_vector_state(dict_state) for dict_state in states])


        # format results
        if labelled_states:
            return self._convert_to_labelled_states(vector_states)
        else:
            return vector_states

    def get_vector_state(self, dict_state):
        return np.array([dict_state[k] for k in self.internal_states_labels])

    def complete_param_distrib_with_table_params(self, json_tables=None):
        if json_tables is None:
            json_tables = "functions_table_default.json"
            json_tables = get_repo_path() + f'data/model_data/world2/{json_tables}'
        with open(json_tables) as fjson:
            tables = json.load(fjson)

        # first, extract all parameters and add them to the initial_internal_params attribute
        self.table_func_names = ["BRCM", "BRFM", "BRMM", "BRPM",
                                 "DRCM", "DRFM", "DRMM", "DRPM",
                                 "CFIFR", "CIM", "CIQR", "FCM", "FPCI", "FPM",
                                 "NREM", "NRMM", "POLAT", "POLCM",
                                 "QLC", "QLF", "QLM", "QLP"]

        table_names = []
        for table in tables:
            table_name = table['y.name']
            for i in range(len(table['x.values'])):
                self.initial_internal_params[table_name + f'_{i}'] = (table["x.values"][i], table["y.values"][i])
            table_names.append(table_name)
        assert sorted(self.table_func_names) == sorted(table_names)  # check that we have all params


    def update_table_funcs(self):
        self.tables = dict()
        for table_func_name in self.table_func_names:
            func_param_keys = sorted([key for key in self.initial_internal_params.keys() if table_func_name in key])
            params = np.array([self.current_internal_params[k] for k in func_param_keys])
            func = interp1d(params[:, 0], params[:, 1],
                            bounds_error=False,
                            fill_value=(params[0, 1],
                                        params[-1, 1]))
            self.tables[table_func_name] = func

    # def complete_param_distrib_with_switch_params(self, json_switches=None):
    #     if json_switches is None:
    #         json_switches = "functions_switch_default.json"
    #         json_switches = get_repo_path() + f'data/model_data/world2/{json_switches}'
    #     with open(json_switches) as fjson:
    #         switches = json.load(fjson)
    #
    #     # first, extract all parameters and add them to the initial_internal_params attribute
    #     self.switch_func_names = ["BRN", "DRN", "CIDN", "CIGN", "FC", "NRUN", "POLN"]
    #
    #     switch_names = []
    #     for func_name in self.switch_func_names:
    #         for switch in switches:
    #             if func_name in switch.keys():
    #                 self.initial_internal_params[func_name] = (switch[func_name], switch[func_name + "1"], switch["trigger.value"])
    #                 switch_names.append(func_name)
    #     assert sorted(self.switch_func_names) == sorted(switch_names)  # check that we have all params

    # def update_switch_funcs(self):
    #     self.switches = dict()
    #     for switch_func_name in self.switch_func_names:
    #         params = self.current_internal_params[switch_func_name]
    #         self.switches[switch_func_name] = Clipper(params[0], params[1], params[2])


    def update_funcs(self):
        self.update_table_funcs()
        # self.update_switch_funcs()
        self.f = self.tables
        # self.f.update(self.switches)


if __name__ == '__main__':

    ## TEST normal conditions

    # # Get model
    # model = World2(stochastic=False)
    #
    # # Run simulation
    # simulation_horizon = 200
    # model_states = model.run_n_steps(n=simulation_horizon)
    #
    # # Plot
    # time = np.arange(model.year_min, model.year_min + simulation_horizon + 1)
    # labels = model.internal_states_labels
    # to_plot = ['P', 'POLR', 'CI', 'QL', 'NR']
    # states_of_interest = model_states[:, np.array([labels.index(k) for k in to_plot])]
    # plot_world_state(time=time, states=states_of_interest, title="World2 scenario - standard run")

    ## TEST increased natural resources
    # Get model
    model = World2(stochastic=False)

    # Run simulation
    simulation_horizon = 200
    # run 1900-1970 without change
    model_states = model.run_n_steps(n=71)

    # apply parameters changes
    model.current_internal_params['NRUN'] = 0.25
    new_model_states = model.run_n_steps(n=simulation_horizon - 71)
    model_states = np.concatenate([model_states, new_model_states[1:]], axis=0)

    # Plot
    time = np.arange(model.year_min, model.year_min + simulation_horizon + 1)
    labels = model.internal_states_labels
    to_plot = ['P', 'POLR', 'CI', 'QL', 'NR']
    states_of_interest = model_states[:, np.array([labels.index(k) for k in to_plot])]
    plot_world_state(time=time, states=states_of_interest, title="World2 scenario - increase natural resources run")
