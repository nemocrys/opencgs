import copy


def replace_params(base_dict, sim_dict):
    """Replaces entries of base_dict that are listed in sim_dict with
    the values given in sim_dict. Recursive function to cover the whole
    depth of the dict.

    Args:
        base_dict (dict): Dictionary containing all simulation parameters
        sim_dict (dict): Dictionary containing the parameters differing from base

    Raises:
        ValueError: If a key of sim_dict is not contained in base_dict

    Returns:
        dict: Modified version of base_dict, now containing the values of sim_dict
        at respective keys.
    """
    for key in sim_dict:
        if not key in base_dict.keys():
            raise ValueError("Wrong parameter name in simulation.yml: " + key)
        if type(sim_dict[key]) is dict:
            base_dict[key] = replace_params(base_dict[key], sim_dict[key])
        else:
            base_dict[key] = sim_dict[key]
    return base_dict


def search_parameter_lists(sim_dict):
    """Searches for lists in values of sim dict. Only key-value pairs
    with lists are kept. Recursive function to cover the whole depth of
    the dict.

    Args:
        sim_dict (dict): Dictionary containing the simulation parameters

    Returns:
        dict: dict (of dicts) with only list as values.
    """
    param_lists_dict = {}
    for key in sim_dict:
        if type(sim_dict[key]) is dict:
            params = search_parameter_lists(sim_dict[key])
            if params != {}:
                param_lists_dict.update({key: params})
        if type(sim_dict[key]) is list:
            param_lists_dict.update({key: sim_dict[key]})
    return param_lists_dict


def get_first_parameter(sim_dict):
    """For the setup of the base simulation for the permutations,
    a set of parameters containing the first value of each list
    is required. For the whole depth of the dict, the following
    is done: If value = dict - recursion, if value = list - take
    first element, else - keep entry as is.

    Args:
        sim_dict (dict): Dictionary containing simulation parameters

    Returns:
        dict: Simulation parameter dict without lists
    """
    base_params = {}
    for key in sim_dict:
        if type(sim_dict[key]) is dict:
            params = get_first_parameter(sim_dict[key])
            if params != {}:
                base_params.update({key: params})
        elif not type(sim_dict[key]) is list:
            base_params.update({key: sim_dict[key]})
    return base_params


def create_permutations(param_lists_dict, base_dict, sim_dict):
    """Creates simulations for all listed parameters.

    Args:
        param_lists_dict (dict): Dict containing parameter lists for respective keys.
        base_dict (dict): Dict containing base parameters.
        sim_dict (dict): Dict containing user defined parameters for this simulation
        (both lists and values only).

    Returns:
        list: Dict with all parameter permutations.
    """
    single_params = get_first_parameter(sim_dict)
    base = replace_params(base_dict, single_params)

    variations = create_permutation_lists(param_lists_dict)

    simulations = [base]
    for variation in variations:
        simulations_new = []
        for sim in simulations:
            for variant in variation:
                sim_new = replace_params(copy.deepcopy(sim), variant)
                print(variant)
                name = (
                    str(variant)
                    .replace("{", "")
                    .replace("}", "")
                    .replace(" ", "")
                    .replace("'", "")
                    .split(":")
                )
                name = name[-2] + "_" + name[-1]
                print(name)
                sim_new["setup"]["name"] += "_" + name
                simulations_new.append(sim_new)
        simulations = simulations_new

    return simulations


def create_permutation_lists(param_lists_dict):
    # TODO dirty solution! -> recursive?
    variations = []
    for key_0, value_0 in param_lists_dict.items():
        if type(value_0) is list:
            variation = []
            for element in value_0:
                variation.append({key_0: element})
            variations.append(variation)
        else:
            for key_1, value_1 in value_0.items():
                if type(value_1) is list:
                    variation = []
                    for element in value_1:
                        variation.append({key_0: {key_1: element}})
                    variations.append(variation)
                else:
                    for key_2, value_2 in value_1.items():
                        if type(value_2) is list:
                            variation = []
                            for element in value_2:
                                variation.append({key_0: {key_1: {key_2: element}}})
                            variations.append(variation)
                        else:
                            raise ValueError("To high depth in data dict.")
    return variations


def create_variations(param_lists_dict, base_dict, sim_dict):
    # TODO dirty solution here - recursion?
    single_params = get_first_parameter(sim_dict)
    base = replace_params(base_dict, single_params)

    simulations = []
    for key0, val0 in param_lists_dict.items():
        if type(val0) is list:
            for val in val0:
                sim_new = replace_params(copy.deepcopy(base), {key0: val})
                sim_new["setup"]["name"] = key0 + "_" + str(val)
                simulations.append(sim_new)
        else:
            for key1, val1 in val0.items():
                if type(val1) is list:
                    for val in val1:
                        sim_new = replace_params(
                            copy.deepcopy(base), {key0: {key1: val}}
                        )
                        sim_new["setup"]["name"] = key0 + "_" + key1 + "_" + str(val)
                        simulations.append(sim_new)
                else:
                    for key2, val2 in val1.items():
                        if type(val2) is list:
                            for val in val2:
                                sim_new = replace_params(
                                    copy.deepcopy(base), {key0: {key1: {key2: val}}}
                                )
                                sim_new["setup"]["name"] = (
                                    key0 + "_" + key1 + "_" + key2 + "_" + str(val)
                                )
                                simulations.append(sim_new)
                        else:
                            raise ValueError("To high depth in data dict.")

    return simulations
