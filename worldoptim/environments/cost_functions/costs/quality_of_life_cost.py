from worldoptim.environments.cost_functions.costs.base_cost_function import BaseCostFunction


class QoLCost(BaseCostFunction):
    def __init__(self,
                 id_cost,
                 scale_factor=0.65 * 1e3,
                 range_constraints=()):
        """
         Cost related to the quality of life.

         Parameters
         ----------
         id_cost: int
             Identifier of the cost in the list of costs
         ratio_death_to_R: float
             Ratio of dead people computed from the number of recovered people, (in [0, 1]).
         scale_factor: float
             Scaling factor of the cost (in [0, 1])
         range_constraints: tuple
             Min and max values for the maximum constraint on the cost (size 2).

         Attributes
         ----------
         id_cost
         """
        super().__init__(scale_factor=scale_factor,
                         range_constraints=range_constraints)
        self.id_cost = id_cost

    def compute_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Computes GDP loss since the last state.

        Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action:

        Returns
        -------
        new_deaths: 1D nd.array
            number of deaths for each state.

        """
        # compute new deaths and pib loss
        qol = state[:, label_to_id['QL']]
        cost = 12 - qol  # Try to get the QoL between 0 and 12
        return cost

    def compute_cumulative_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Compute cumulative costs since start of episode.

        Parameters
        ----------
               Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action:

        Returns
        -------
        cumulative_cost: 1D nd.array
            Cumulative costs for each state.
        """
        cumulative_cost = state[:, label_to_id['cumulative_cost_{}'.format(self.id_cost)]]

        return cumulative_cost
