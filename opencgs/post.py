from dataclasses import dataclass
import meshio
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml

from opencgs.elements import Triangle1st, Triangle2nd, Node, Line1st, Line2nd
pd.DataFrame.iteritems = pd.DataFrame.items

@dataclass
class HeatfluxSurf:
    """Dataclass for definition of surfaces for heat flux evaluation."""
    ID: int
    BodyIDs: list
    lmbd: float


@dataclass
class Simulation:
    """Dataclass for evaluation of parameter studies."""
    dir: str
    name: str
    val: float

    @property
    def probes(self):
        with open(self.dir + "/03_results/probes.yml") as f:
            return yaml.safe_load(f)

    @property
    def heat_fluxes(self):
        with open(self.dir + "/03_results/heat-fluxes.yml") as f:
            return yaml.safe_load(f)


def heat_flux(sim_dir, res_dir, plot=False, save=True, normal_proj=True):
    """Evaluate heat flux over boundaries from temperature field.

    Args:
        sim_dir (str): simulation directory
        res_dir (str): results directory (for output)
        plot (bool, optional): Show plots. Defaults to False.
        save (bool, optional): Save plots to file. Defaults to True.
        normal_proj (bool, optional): Project heat fluxes on the normal
            of the respective boundary. Defaults to True.

    Returns:
        tuple: figure, axis, dictionary with heat fluxes.
    """
    # import data
    files = os.listdir(sim_dir)
    vtu = []
    for f in files:
        ending = f.split(".")[-1]
        if ending == "msh" and f != "result.msh":
            msh = f
        if ending == "vtu":
            vtu.append(f)
    msh = meshio.read(sim_dir + "/" + msh)
    vtu = meshio.read(sim_dir + "/" + vtu[-1])
    with open(sim_dir + "/post_processing.yml", "r") as f:
        data = yaml.safe_load(f)

    # detect interpolation order
    if "triangle" in msh.cells_dict:
        order = 1
    elif "triangle6" in msh.cells_dict:
        order = 2

    # extract data from msh and vtu
    if order == 1:
        boundary_ids = msh.cell_data_dict["gmsh:physical"]["line"]
        nodes = msh.cells_dict["line"]
        df_lines = pd.DataFrame(
            {"BoundaryID": boundary_ids, "Node1": nodes[:, 0], "Node2": nodes[:, 1]}
        )

        surface_ids = msh.cell_data_dict["gmsh:physical"]["triangle"]
        nodes = msh.cells_dict["triangle"]
        df_elements = pd.DataFrame(
            {
                "SurfaceID": surface_ids,
                "Node1": nodes[:, 0],
                "Node2": nodes[:, 1],
                "Node3": nodes[:, 2],
            }
        )

    elif order == 2:
        boundary_ids = msh.cell_data_dict["gmsh:physical"]["line3"]
        nodes = msh.cells_dict["line3"]
        df_lines = pd.DataFrame(
            {
                "BoundaryID": boundary_ids,
                "Node1": nodes[:, 0],
                "Node2": nodes[:, 1],
                "Node3": nodes[:, 2],
            }
        )

        surface_ids = msh.cell_data_dict["gmsh:physical"]["triangle6"]
        nodes = msh.cells_dict["triangle6"]
        df_elements = pd.DataFrame(
            {
                "SurfaceID": surface_ids,
                "Node1": nodes[:, 0],
                "Node2": nodes[:, 1],
                "Node3": nodes[:, 2],
                "Node4": nodes[:, 3],
                "Node5": nodes[:, 4],
                "Node6": nodes[:, 5],
            }
        )

    points = (
        vtu.points
    )  # take coordinates from vtu because mesh may have been deformed by PhaseChangeSolver
    temperature = vtu.point_data["temperature"]
    df_nodes = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "T": temperature[:, 0],
        }
    )

    # evaluate boundaries
    x_quiver = []  # coordinates for quiver plot
    q_quiver = []  # heat fluxes for quiver plot
    l_quiver = []  # boundaries of elements for plot of geometry
    fluxes = {}
    for boundary in data:
        df_l = df_lines.loc[df_lines["BoundaryID"] == data[boundary]["ID"]]
        df_e = df_elements.loc[df_elements["SurfaceID"].isin(data[boundary]["BodyIDs"])]
        lmbd = data[boundary]["lmbd"]
        # find elements on boundary & in body
        element_ids = []
        for _, line in df_l.iterrows():
            if order == 1:
                nodes = [line["Node1"], line["Node2"]]
                (idx,) = df_e.loc[
                    (df_e["Node1"].isin(nodes)) & (df_e["Node2"].isin(nodes))
                    | (df_e["Node1"].isin(nodes)) & (df_e["Node3"].isin(nodes))
                    | (df_e["Node2"].isin(nodes)) & (df_e["Node3"].isin(nodes))
                ].index
            elif order == 2:
                nodes = [line["Node1"], line["Node2"], line["Node3"]]
                (idx,) = df_e.loc[
                    (df_e["Node1"].isin(nodes))
                    & (df_e["Node4"].isin(nodes))
                    & (df_e["Node2"].isin(nodes))
                    | (df_e["Node2"].isin(nodes))
                    & (df_e["Node5"].isin(nodes))
                    & (df_e["Node3"].isin(nodes))
                    | (df_e["Node3"].isin(nodes))
                    & (df_e["Node6"].isin(nodes))
                    & (df_e["Node1"].isin(nodes))
                ].index
            element_ids.append(idx)
        df_l.insert(3, "ElementID", element_ids)

        Q = 0
        for _, line in df_l.iterrows():
            element = df_e.loc[line["ElementID"] == df_e.index].iloc[0]
            node = df_nodes.iloc[element["Node1"]]
            n1 = Node(node["x"], node["y"], node["z"], node["T"])
            node = df_nodes.iloc[element["Node2"]]
            n2 = Node(node["x"], node["y"], node["z"], node["T"])
            node = df_nodes.iloc[element["Node3"]]
            n3 = Node(node["x"], node["y"], node["z"], node["T"])
            line_node = df_nodes.iloc[line["Node1"]]
            line_n1 = Node(line_node["x"], line_node["y"], line_node["z"])
            line_node = df_nodes.iloc[line["Node2"]]
            line_n2 = Node(line_node["x"], line_node["y"], line_node["z"])
            if order == 1:
                t = Triangle1st([n1, n2, n3])
                l = Line1st([line_n1, line_n2])
            if order == 2:
                node = df_nodes.iloc[element["Node4"]]
                n4 = Node(node["x"], node["y"], node["z"], node["T"])
                node = df_nodes.iloc[element["Node5"]]
                n5 = Node(node["x"], node["y"], node["z"], node["T"])
                node = df_nodes.iloc[element["Node6"]]
                n6 = Node(node["x"], node["y"], node["z"], node["T"])
                line_node = df_nodes.iloc[line["Node3"]]
                line_n3 = Node(line_node["x"], line_node["y"], line_node["z"])
                t = Triangle2nd([n1, n2, n3, n4, n5, n6])
                l = Line2nd([line_n1, line_n2, line_n3])

            # check orientation of normal, correct if necessary
            if not ((n1.X == l.n1.X).all() or (n1.X == l.n2.X).all()):
                n = n1
            elif not ((n2.X == l.n1.X).all() or (n2.X == l.n2.X).all()):
                n = n2
            elif not ((n3.X == l.n1.X).all() or (n3.X == l.n2.X).all()):
                n = n3
            if np.linalg.norm(l.n1.X - l.normal - n.X) > np.linalg.norm(
                l.n1.X + l.normal - n.X
            ):
                l.invert_normal()

            # TODO non-axi-symmetric cases
            # area along x-axis: circle ring
            r_in = min([l.n1.x, l.n2.x])
            r_out = max([l.n1.x, l.n2.x])
            A_x = np.pi * (r_out ** 2 - r_in ** 2)
            # area along y-axis: cylinder
            dy = abs(l.n2.y - l.n1.y)
            r_mean = np.mean([l.n1.x, l.n2.x])
            A_y = 2 * np.pi * r_mean * dy

            center = l.n1.X + 0.5 * (l.n2.X - l.n1.X)
            T_grad = t.B_e(center[0], center[1]) @ t.T

            Q += -lmbd * (
                A_y * T_grad[0] * np.sign(l.normal[0])
                + A_x * T_grad[1] * np.sign(l.normal[1])
            )
            x_quiver.append(center)
            l_quiver.append(np.array([l.n1.X, l.n2.X]))
            if not normal_proj:
                q_quiver.append(-lmbd * T_grad)
            else:
                q_quiver.append(-lmbd * T_grad @ l.normal * l.normal)
        fluxes.update({boundary: float(Q)})

    fig1, ax1 = plt.subplots(1, 1, figsize=(24, 16))
    fig2, ax2 = plt.subplots(1, 1, figsize=(24, 16))
    fig3, ax3 = plt.subplots(1, 1, figsize=(24, 16))
    x_quiver = np.array(x_quiver)
    q_quiver = np.array(q_quiver)
    for l in l_quiver:
        ax1.plot(l[:, 0], l[:, 1], "b", linewidth=0.5)
        ax2.plot(l[:, 0], l[:, 1], "b", linewidth=0.5)
        ax3.plot(l[:, 0], l[:, 1], "b", linewidth=0.5)
        ax1.plot(
            [0, 0], [x_quiver[:, 1].min(), x_quiver[:, 1].max()], "b", linewidth=0.5
        )
        ax2.plot(
            [0, 0], [x_quiver[:, 1].min(), x_quiver[:, 1].max()], "b", linewidth=0.5
        )
        ax3.plot(
            [0, 0], [x_quiver[:, 1].min(), x_quiver[:, 1].max()], "b", linewidth=0.5
        )
    ax1.quiver(
        x_quiver[:, 0],
        x_quiver[:, 1],
        q_quiver[:, 0],
        q_quiver[:, 1],
        width=5e-4,
        scale=2.5e5,
    )
    ax2.quiver(
        x_quiver[:, 0],
        x_quiver[:, 1],
        q_quiver[:, 0],
        q_quiver[:, 1],
        width=5e-4,
        scale=1e6,
    )
    ax3.quiver(
        x_quiver[:, 0],
        x_quiver[:, 1],
        q_quiver[:, 0],
        q_quiver[:, 1],
        width=5e-4,
        scale=5e5,
    )
    # legend arrow
    x_max = x_quiver[:, 0].max()
    y_min = x_quiver[:, 1].min()
    ax1.quiver(x_max / 4, y_min - 0.015, 2.5e5, 0, width=1e-3)
    ax1.text(x_max / 4, y_min - 0.01, "$2.5\cdot10^5~\mathrm{W/m^2}$")
    ax2.quiver(x_max / 4, y_min - 0.015, 1e6, 0, width=1e-3)
    ax2.text(x_max / 4, y_min - 0.01, "$1.0\cdot10^6~\mathrm{W/m^2}$")
    ax3.quiver(x_max / 4, y_min - 0.015, 5e5, 0, width=1e-3)
    ax3.text(x_max / 4, y_min - 0.01, "$5.0\cdot10^5~\mathrm{W/m^2}$")
    ax1.axis("equal")
    ax2.axis("equal")
    ax3.axis("equal")
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    if save:
        with open(res_dir + "/heat-fluxes.yml", "w") as f:
            yaml.dump(fluxes, f)
        fig1.savefig(res_dir + "/heat-fluxes-fine.pdf", transparent=True)
        fig2.savefig(res_dir + "/heat-fluxes-medium.pdf", transparent=True)
        fig3.savefig(res_dir + "/heat-fluxes-fine2.pdf", transparent=True)
        fig1.savefig(res_dir + "/heat-fluxes-fine.png", transparent=True)
        fig2.savefig(res_dir + "/heat-fluxes-medium.png", transparent=True)
        fig3.savefig(res_dir + "/heat-fluxes-fine2.png", transparent=True)
    if plot:
        plt.show()
    return fig2, ax2, fluxes


def parameter_study(sim_dir, plot_dir):
    """Create overview plots of parameter study."""
    # scan directory
    simulation_dirs = os.listdir(sim_dir)
    param_sweeps = {}
    for sim in simulation_dirs:
        param = "_".join(sim.split("_")[1:]).split("=")[0]
        value = float(sim.split("=")[-1])
        simulation = Simulation(f"{sim_dir}/{sim}", param, value)
        if param not in param_sweeps:
            param_sweeps.update({param: {value: simulation}})
        else:
            param_sweeps[param].update({value: simulation})
    # plot
    for param, sim_dict in param_sweeps.items():
        if len(param_sweeps) == 1:
            plot_dir_ = plot_dir
        else:
            plot_dir_ = f"{plot_dir}/{param}"
            os.mkdir(plot_dir_)
        for probe in next(iter(sim_dict.values())).probes:
            fig, ax = plt.subplots(1, 1)
            x = []
            y = []
            for val in sorted(sim_dict):
                x.append(val)
                y.append(sim_dict[val].probes[probe])
            ax.plot(x, y, "x-")
            ax.grid(linestyle=":")
            ax.set_xlabel(param)
            ax.set_ylabel(probe)
            fig.tight_layout()
            fig.savefig(f"{plot_dir_}/pb_{probe}.png")
            plt.close(fig)
        for hf in next(iter(sim_dict.values())).heat_fluxes:
            fig, ax = plt.subplots(1, 1)
            x = []
            y = []
            for val in sorted(sim_dict):
                x.append(val)
                y.append(sim_dict[val].heat_fluxes[hf])
            ax.plot(x, y, "x-")
            ax.grid(linestyle=":")
            ax.set_xlabel(param)
            ax.set_ylabel(hf)
            fig.tight_layout()
            fig.savefig(f"{plot_dir_}/hf_{hf}.png")
            plt.close(fig)
