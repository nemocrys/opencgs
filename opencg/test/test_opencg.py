from pyelmer.gmsh_objects import Model
import opencg.geo.czochralski as cz

def test_crucible():
    model = Model()
    config = {
        'h': 0.1,
        'r_in': 0.05,
        'r_out': 0.06,
        't_bt': 0.01,
        'T_init': 0.01
    }
    crucible = cz.crucible(model, 2, **config)
    # model.show()
    assert crucible.params.h == 0.1


if __name__ == "__main__":
    test_crucible()