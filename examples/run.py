import opencg.control as ctrl


config = './examples/cz_test/config.yml'


if __name__ == "__main__":
    simulations = ctrl.create_simulations(config)
    ctrl.execute(simulations)
