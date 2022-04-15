from worldoptim.environments.cost_functions.multi_cost_death_gdp_controllable import MultiCostDeathGdpControllable
from worldoptim.environments.cost_functions.multi_cost_deathrate_qol import MultiCostDeathrateQOL

admissible_cost_functions = ['multi_cost_deathrate_qol', 'multi_cost_death_gdp_controllable']

def get_cost_function(cost_function_id, params={}):
    assert cost_function_id in admissible_cost_functions
    if cost_function_id == 'multi_cost_death_gdp_controllable':
        return MultiCostDeathGdpControllable(**params)
    elif cost_function_id == 'multi_cost_deathrate_qol':
        return MultiCostDeathrateQOL(**params)
    else:
        raise NotImplementedError
