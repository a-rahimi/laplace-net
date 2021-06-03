from typing import Iterable, Tuple

import matplotlib.pylab as plt
import numpy as np
import torch


def lattice_edges(
    image_height: int, image_width: int
) -> Tuple[Iterable[int], Iterable[int]]:
    """The set of edges in a 2D 4-connected graph.

    The edges in the network connect each vertex to its four adjacent vertices.
    A vertex (i,j) is represented as its flat index into the 2D image, as the
    integer i * image_width + j.

    An edge connects two vertices, so we represent the edges sa two lists: the
    flat index of the start vertex, and the flat index of the end vertex.

    Every pair of connected vertices appears twice in this representation: once
    where the vertex is the center, and once again when it's a neighbor.
    """

    # nx2 array of coordinates (i,j)
    Is, Js = np.meshgrid(range(image_height), range(image_width))
    centers = np.vstack((Is.flatten(), Js.flatten())).T

    adjacency = np.array(((0, -1), (0, +1), (-1, 0), (+1, 0)))

    # 4n x 2 array of neighbors of (i,j)
    neighbors = (centers[:, None, :] + adjacency).reshape(-1, 2)

    # Index of valid neighbors
    i_valid = (
        (0 <= neighbors[:, 0])
        * (neighbors[:, 0] < image_height)
        * (0 <= neighbors[:, 1])
        * (neighbors[:, 1] < image_width)
    )

    # Represent the edges in the network. Each entry of edges_center is
    # (i,j), and its corresponding entry in edges_neighbor is an element of
    # N(i,j)
    edges_center = centers.repeat(len(adjacency), axis=0)[i_valid].dot((image_width, 1))
    edges_neighbor = neighbors[i_valid].dot((image_width, 1))

    return edges_center, edges_neighbor


class SolveLaplace(torch.nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()

        # Parameters
        self.log_resistances = torch.nn.Parameter(
            torch.rand(image_height, image_width, dtype=torch.float64)
        )
        self.output_offset = torch.nn.Parameter(torch.tensor(0.0))

        self.edges_center, self.edges_neighbor = lattice_edges(
            image_height, image_width
        )

    def forward(self, input_currents: torch.Tensor):
        params_height, params_width = self.log_resistances.shape
        n_batches, image_height, image_width = input_currents.shape
        assert params_height == image_height
        assert params_width == image_width

        # This slab can't store or leak currents. Ensure the total current flux is 0.
        input_currents = input_currents - input_currents.mean()

        # To enforce that resistance must always be positive, the instantaneous
        # resistance at node (i,j) is supplied as the log-resistances r[i,j].
        # The instantaneous resistance at node (i,j) is  R[i,j] = exp(r[i,j]).
        #
        # Let (u,v) = N(i,j) denote the neighbors of node (i,j).
        # The resistance between two adjacent nodes (i,j) and (u,v) can be
        # computed from the instantaneous resistance at the nodes: it's the
        # sum of their instantaneous resistance:
        #
        #        R[(i,j), (u,v)] = exp(r[i,j]) + exp(r[u,v])
        #
        # To compute the voltage at every node (i,j), we'll use the fact that
        # the current flowing out of the node must equal the current flowing in:
        #
        #    I[i,j] = sum_{(u,v) in N(i,j)} (V[i,j] - V[u,v]) / R[(i,j), (u,v)]
        #
        # In matrix form, this gives
        #
        #    Z V = I,
        #
        # where row (i,j) of matrix Z has the form
        #
        #    [... -1/R[(i,j), (u1,v1)] ... sum_{(u,v) in N(i,j)} 1/R[(i,j),(u,v)] ... -1/R[(i,j), (ui,vi)] ...]
        #
        # The non-zero entries are at columns (u,v) in N(i,j), the neighbors of (i,j).

        # Element (i,j) of this vector is R[i,j]
        center_conductances = torch.exp(self.log_resistances).flatten()

        Z_off_diagonal = torch.sparse_coo_tensor(
            ((self.edges_center, self.edges_neighbor)),
            center_conductances[self.edges_center]
            + center_conductances[self.edges_neighbor],
            (image_width * image_height, image_width * image_height),
        ).to_dense()

        Z = torch.diag(Z_off_diagonal.sum(axis=1)) - Z_off_diagonal

        # Z is symmetric and has a null-space, with Z 1 = 0, so it won't do to
        # just run solve(Z, I). Furthermore, pytorch doesn't have a built-in
        # way to find th minimum norm solution to an over-determined system of
        # equations. There's an easy work-around: We know 1'I = 0 also, because
        # the sum of currents flowing into the resistor network must equal the
        # sum of outgoing currents. This implies the following:
        #
        #   Claim: Z+11' has full rank. Furthermore, if x satisfies
        #        (Z + 11') x = I, it also satisfies Z x = I.
        #
        # An easy proof is to write the SVD of Z+11' in terms of the SVD Z=USV',
        # and to notice that Z (Z+11')^-1 y = y.
        #
        # All this to say that instead of solve(Z, y), we run solve(Z + 1, y)

        return (
            torch.linalg.solve(
                Z + 1,
                input_currents.reshape(n_batches, image_width * image_height, 1),
            ).reshape(input_currents.shape)
            + self.output_offset
        )


class LayeredLaplace(torch.nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.lap1 = SolveLaplace(image_height, image_width)
        self.voltage_to_current_converter = torch.nn.Identity()
        self.lap2 = SolveLaplace(image_height, image_width)

    def forward(self, x):
        return self.lap2(self.voltage_to_current_converter(self.lap1(x)))


def main():
    plt.style.use("dark_background")

    V_target = torch.zeros(5, 5, dtype=torch.float64)
    V_target[V_target.shape[0] // 2, 1:-1] = 1.0

    I_input = torch.zeros(*V_target.shape, dtype=torch.float64)
    I_input[0, 0] = 1
    I_input[-1, -1] = -1

    model = LayeredLaplace(*V_target.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    # Set up plots.
    fig, axs = plt.subplots(2, 3, figsize=(18, 6))

    axs[0, 0].imshow(V_target, cmap=plt.cm.hot)
    axs[0, 0].set_title("Target voltages")

    axs[0, 1].imshow(I_input, cmap=plt.cm.hot)
    axs[0, 1].set_title("Input current")

    axs[0, 2].axis("off")

    (h_losses,) = axs[1, 0].plot([])
    axs[1, 0].set_title("Losses")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Loss")

    im_vout = axs[1, 1].imshow(0 * V_target, cmap=plt.cm.hot)
    axs[1, 1].set_title("Observed voltages")

    im_log_resistances = axs[1, 2].imshow(
        torch.exp(model.lap1.log_resistances.detach()).numpy(),
        cmap=plt.cm.hot,
    )
    axs[1, 2].set_title("Resistances")

    losses = []
    for it in range(1000):
        # Take one optimization step.
        optimizer.zero_grad()
        V_out = model(I_input[None, :, :])
        loss = ((V_out - V_target) ** 2).sum()
        loss.backward()
        optimizer.step()

        losses.append(float(loss))

        # Update the plots
        h_losses.set_data(range(len(losses)), losses)
        axs[1, 0].set_xlim((0, len(losses)))
        axs[1, 0].set_ylim((min(losses), max(losses)))
        im_vout.set_data(V_out.detach().numpy()[0])
        im_vout.autoscale()
        im_log_resistances.set_data(
            torch.exp(model.lap1.log_resistances.detach()).numpy()
        )
        im_log_resistances.autoscale()
        plt.pause(1e-4)
        plt.show(block=False)
    plt.show()


if __name__ == "__main__":
    main()
