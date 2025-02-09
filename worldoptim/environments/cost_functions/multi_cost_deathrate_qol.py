from worldoptim.environments.cost_functions.base_multi_cost_function import BaseMultiCostFunction
from worldoptim.environments.cost_functions.costs.death_rate_cost import DeathRate
from worldoptim.environments.cost_functions.costs.quality_of_life_cost import QoLCost
import numpy as np


class MultiCostDeathrateQOL(BaseMultiCostFunction):
    # This function computes two independent costs:
    # The death rate cost: exponential function of death rate, with minimum threshold
    # The quality of life cost: - QoL
    def __init__(self,
                 drn,
                 use_constraints=False,
                 beta_default=0.5,
                 ):
        """
        Multi-objective cost functions with two costs: death rate and QoL. It is controllable by
        the mixing parameter beta.

        Parameters
        ----------
        use_constraints: bool
            Whether to use constraints on the maximum values of cumulative rewards.
        beta_default: float
            Default mixing parameter.
        """
        super().__init__(use_constraints=use_constraints)

        self.beta_default = beta_default
        self.beta = self.beta_default

        # Initialize cost functions
        self.death_rate_cost = DeathRate(id_cost=0, drn=drn)
        self.qol_cost = QoLCost(id_cost=1)

        self.costs = [self.death_rate_cost, self.qol_cost]
        self.nb_costs = len(self.costs)

        if self.use_constraints:
            self.goal_dim = 3
            self.constraints_ids = [[1], [2]]  # ids of the constraints in the goal vector (0 is mixing param)
        else:
            self.goal_dim = 1
            self.constraints_ids = []

    def sample_goal_params(self):
        """
        Sample goal parameters.

        Returns
        -------
        goal: 1D nd.array
            Made of three params in [0, 1]: beta is the mixing parameter,
            the following are normalized constraints on the maximal values of cumulative costs.

        """
        beta = np.random.rand()
        if self.use_constraints:
            raise NotImplementedError

        #     r = np.random.rand()
        #     if r < 0.25:
        #         constraints = [1, np.random.rand()]
        #     elif r < 0.5:
        #         constraints = [np.random.rand(), 1]
        #     else:
        #         constraints = [1, 1]
        #     goal = [beta] + constraints
        else:
            goal = [beta]
        return np.array(goal)

    def get_eval_goals(self, n):
        if self.use_constraints:
            raise NotImplementedError
        else:
            eval_goals = np.atleast_2d(np.array([0] * n + [0.25] * n + [0.5] * n + [0.75] * n + [1] * n)).transpose()
        return eval_goals

    def get_main_goal(self):
        if self.use_constraints:
            raise NotImplementedError
        else:
            eval_goals = np.array([0.5])
        return eval_goals

    def set_goal_params(self, goal):
        """
        Set a goal.

        Parameters
        ----------
        goal: 1D nd.array
            Should be of size 3: mixing parameter beta and normalized constraints, all in [0, 1].

        """
        self.beta = goal[0]
        if self.use_constraints:
            raise NotImplementedError

            # if len(goal[1:]) == self.nb_costs:
            #     for v, c in zip(goal[1:], self.costs):
            #         c.set_constraint(v)
            # else:
            #     for c in self.costs:
            #         c.set_constraint(1.)


    def compute_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Compute each cost and an aggregated measure of costs as well as constraints.

        Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action: int or nd.array
            Int for discrete envs and nd.array in continuous envs.

        Returns
        -------
        cost_aggregated: float
            Aggregated cost using the beta mixing parameter.
        costs: list of floats
            All costs.
        over_constraints: list of bools
            Whether the constraints are violated.
        """
        previous_state = np.atleast_2d(previous_state)
        state = np.atleast_2d(state)

        # compute costs
        costs = np.array([c.compute_cost(previous_state, state, label_to_id, action, others) for c in self.costs]).transpose()
        cumulative_costs = np.array([c.compute_cumulative_cost(previous_state, state, label_to_id, action, others) for c in self.costs]).transpose()

        # Apply constraints
        if self.use_constraints:
            raise NotImplementedError

            # constraints = np.array([c.normalized_constraint for c in self.costs])
            # over_constraints = np.array([[c.check_constraint(value=cumul_cost, normalized_constraint=const)
            #                               for c, cumul_cost, const in zip(self.costs, cumulative_costs[i], constraints)]
            #                              for i in range(cumulative_costs.shape[0])])
            # cost_aggregated = self.compute_aggregated_cost(costs.copy(), constraints=over_constraints)
        else:
            cost_aggregated = self.compute_aggregated_cost(costs.copy())
            over_constraints = np.atleast_2d([False] * state.shape[0]).transpose()
        return cost_aggregated, costs, over_constraints

    def compute_aggregated_cost(self, costs, beta=None, constraints=None):
        """
        Compute aggregated measure of cost with mixing beta and optional constraints.
        Parameters
        ----------
        costs: 2D nd.array
            Array of costs (n_points, n_costs).
        beta: float
            Mixing parameters for the two costs.
        constraints: nd.array of bools, optional
            Whether constraints are violated (n_point, n_costs).

        Returns
        -------
        float
            Aggregated cost.
        """
        if beta is None:
            beta = self.beta
        factors = np.array([1 - beta, beta])
        normalized_costs = np.array([cf.scale(c) for (cf, c) in zip(self.costs, costs.transpose())])
        cost_aggregated = np.matmul(factors, normalized_costs)

        if self.use_constraints:
            if constraints is not None:
                cost_aggregated[np.argwhere(np.sum(constraints, axis=1) > 0)] = 100
        return cost_aggregated
