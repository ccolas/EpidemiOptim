from worldoptim.environments.cost_functions.costs.base_cost_function import BaseCostFunction
import numpy as np


class DeathRate(BaseCostFunction):
    def __init__(self,
                 id_cost,
                 drn,  # base DR computed in 1970
                 scale_factor=0.65 * 1e3,
                 range_constraints=()):
        """
         Cost related to the death rate.

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
         ratio_death_to_R
         id_cost
         """
        super().__init__(scale_factor=scale_factor,
                         range_constraints=range_constraints)
        self.id_cost = id_cost
        self.drn = drn

    def cost_from_normalized_dr(self, x, shift=1, beta=1):
        # function that increases exponentially above 1, has a 0 cost below 1.
        y = np.exp(x.copy() * beta + shift) - np.exp(beta + shift)
        y[np.where(x < 1)] = 0
        return y

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
        # compute death rate
        # DR state is actually the number of deaths. True death rate is DR / P. We normalize by the DR of 1970, i.e. DRN
        death_rate = state[:, label_to_id['DR']] / state[:, label_to_id['P']] / self.drn
        # now we want the cost to be 0 when this normalized death rate is below 1 (below 1970's level), but then increases exponentially
        cost = self.cost_from_normalized_dr(death_rate)
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
