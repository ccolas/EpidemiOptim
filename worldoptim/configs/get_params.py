
def get_params(config_id, expe_name=None):
    """
    Get experiment parameters.

    Parameters
    ----------
    config_id: str
        Name of the config.
    expe_name: str
        Name of the experiment, optional.

    Returns
    -------
    params: dict
        Dictionary of experiment parameters.

    """
    if config_id == 'dqn':
        from worldoptim.configs.dqn import params
    elif config_id == 'goal_dqn':
        from worldoptim.configs.goal_dqn import params
    elif config_id == 'goal_dqn_constraints':
        from worldoptim.configs.goal_dqn_constraints import params
    elif config_id == 'nsga_ii':
        from worldoptim.configs.nsga_ii import params
    elif config_id == 'nsga_ii_world2':
        from worldoptim.configs.nsga_ii_world2 import params
    else:
        raise NotImplementedError
    if expe_name:
        params.update(expe_name=expe_name)
    return params