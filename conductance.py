from typing import Iterable, Tuple

import matplotlib.pylab as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_func


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


class SolvePoisson(nn.Module):
    """Solve the possoin equation on a 2D grid.

    Solve a problem of the form

       ∇²(R y) = x

    on a 2D grid for y. Here, R is a 2D array with dimension (height, width),
    and x, and y are batched 2D arrays with dimensions as batch_size x height x
    width.

    Since the solver is written in torch, you can compute the gradient of the
    solution with respect to both x and R.

    This is a generic solver for Poisson's equation, but the nomenclature is
    specific to solving a problem for a resistive sheet whose conductance
    varies over space.

    Example: Suppose a current I(i,j) is driven through each point (i,j) of a
    resistive sheet whose resistance at point (i,j) is R(i,j) =
    exp(log_resistance(i,j)). To compute the voltage at every point (i,j), we
    would solve

          ∇²(exp(log_resistances) voltages) = currents

    Run:
          solver = SolvePoissoin(log_resistances)
          voltages = solver(currents)

    But we can also solve for the resistances if we're given the currents and
    the observed voltages.

    To enforce that resistance must always be positive, the instantaneous
    resistance at node (i,j) is supplied as the log-resistances r[i,j].  The
    instantaneous resistance at node (i,j) is  R[i,j] = exp(r[i,j]).

    Even though in the continuous representation, c and x appear
    interchangeable, they play different roles in the discrete problem.  Let
    (u,v) = N(i,j) denote the neighbors of node (i,j).  The resistance between
    two adjacent nodes (i,j) and (u,v) can be computed from the instantaneous
    resistance at the nodes: it's the sum of their instantaneous resistance:

           R[(i,j), (u,v)] = exp(r[i,j]) + exp(r[u,v])

    To compute the voltage at every node (i,j), we'll use the fact that the
    current flowing out of the node must equal the current flowing in:

           I[i,j] = sum_{(u,v) in N(i,j)} (V[i,j] - V[u,v]) / R[(i,j), (u,v)]

    In matrix form, this gives

           Z V = I,

    where row (i,j) of matrix Z has the form

          [... 1/R[(i,j), (u1,v1)] ... -Z_ii ...  1/R[(i,j), (ui,vi)] ...]

    where Z_ii is the sum of the all the other entries in the row.  The
    non-zero entries are at columns (u,v) in N(i,j), the neighbors of (i,j).
    """

    def __init__(self, log_resistances: torch.Tensor):
        super().__init__()
        self.log_resistances = nn.Parameter(log_resistances)

        self.edges_center, self.edges_neighbor = lattice_edges(*log_resistances.shape)

    def forward(self, input_currents: torch.Tensor):
        params_height, params_width = self.log_resistances.shape
        n_batches, image_height, image_width = input_currents.shape
        assert params_height == image_height
        assert params_width == image_width

        # This slab can't store or leak currents. Ensure the total current flux is 0.
        input_currents = input_currents - input_currents.mean()

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
        # All this to say that instead of solve(Z, y), we run solve(Z + 2, y)

        return (
            torch.linalg.solve(
                Z + 1,
                input_currents.double().reshape(
                    n_batches, image_width * image_height, 1
                ),
            )
            .reshape(input_currents.shape)
            .float()
        )


class SolvePoissonTensor(nn.Module):
    """Solve one Poisson equation per channel and sum the results.

    Args:
        x: (num_batches, in_planes, height, width)
        R: (in_planes, height, width)

    Returns:
        y: (num_batches, height, width)

    For each plane c, solve for y_c in

        ∇²(R_c y_c) = x_c

    Then return y = bias + sum_c w_c y_c.
    """

    def __init__(self, in_planes: int, image_height: int, image_width: int):
        super().__init__()

        self.solvers = nn.ModuleList(
            [
                SolvePoisson(torch.rand(image_height, image_width, dtype=torch.float64))
                for _ in range(in_planes)
            ]
        )
        self.weights = nn.Parameter(torch.ones(in_planes))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_currents: torch.Tensor):
        in_planes = input_currents.shape[1]
        assert in_planes == len(self.solvers)

        # ys has shape (in_planes, num_batches, height, width)
        ys = torch.stack(
            [
                self.weights[i] * self.solvers[i](input_currents[:, i, :, :])
                for i in range(in_planes)
            ]
        )

        # return (num_batches, height, width)
        return self.bias + ys.sum(axis=0)


class MultiSolvePoissonTensor(nn.Module):
    """A bank of tensoriized poisson solvers.

    Multiple replicas of SolvePoissonTensor, stacked to create a multi-channel image.
    """

    def __init__(
        self, in_planes: int, image_height: int, image_width: int, out_planes: int
    ):
        super().__init__()

        self.tensor_solvers = nn.ModuleList(
            SolvePoissonTensor(in_planes, image_height, image_width)
            for _ in range(out_planes)
        )

    def forward(self, input_currents: torch.Tensor) -> torch.Tensor:
        # r is (out_planes, num_batches, height, width)
        r = torch.stack(
            [tensor_solver(input_currents) for tensor_solver in self.tensor_solvers]
        )
        # Convert to (num_batches, out_planes, height, width)
        return r.transpose(0, 1)


class BasicBlock(nn.Module):
    def __init__(
        self, in_planes: int, image_height: int, image_width: int, out_planes: int
    ):
        super().__init__()
        self.solver1 = MultiSolvePoissonTensor(
            in_planes, image_height, image_width, out_planes
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.solver2 = MultiSolvePoissonTensor(
            out_planes, image_height, image_width, out_planes
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch_func.relu(self.bn1(self.solver1(x)))
        out = self.bn2(self.solver2(out))

        # The final output may have more planes (aka channels) than the input x.
        # If so, pad x to match the output.
        in_planes = x.shape[1]
        out_planes = out.shape[1]
        if in_planes < out_planes:
            if (out_planes - in_planes) % 2:
                raise ValueError(
                    "For now, planes may vary by even numbers only: in_planes = %d, out_planes = %d"
                    % (in_planes, out_planes)
                )
            pad = (out_planes - in_planes) // 2
            shortcut = torch_func.pad(
                x,
                (0, 0, 0, 0, pad, pad),
            )
        elif in_planes == out_planes:
            shortcut = x
        else:
            raise ValueError(
                "For now, planes may only increase with layers: in_planes = %d, out_planes = %d"
                % (in_planes, out_planes)
            )

        return torch_func.relu(out + shortcut)


def make_copies(num_copies, cls, *args, **kwargs):
    return nn.Sequential(*[cls(*args, **kwargs) for _ in range(num_copies)])


class PoissonNet(nn.Module):
    """A resnet-like network where convolutions have been replaced by Poisson solver."""

    def __init__(
        self, image_height: int = 32, image_width: int = 32, num_classes: int = 10
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.group1 = make_copies(2, BasicBlock, 4, image_height, image_width, 4)

        self.group1_to_group2 = BasicBlock(4, image_height // 2, image_width // 2, 8)

        self.group2 = make_copies(
            2, BasicBlock, 8, image_height // 2, image_width // 2, 8
        )

        self.group2_to_group3 = BasicBlock(8, image_height // 4, image_width // 4, 16)

        self.group3 = make_copies(
            2, BasicBlock, 16, image_height // 4, image_width // 4, 16
        )
        self.linear = nn.Linear(16, num_classes)

    def forward(self, x):
        # x is batch x 3 x 32 x 32

        out = torch_func.relu(self.bn1(self.conv1(x)))
        # out is batch x 4 x 32 x 32

        out = self.group1(out)
        # out is batch x 4 x 32 x 32

        out = torch_func.avg_pool2d(out, 2)
        # out is batch x 4 x 16 x 16

        out = self.group1_to_group2(out)
        # out is batch x 8 x 16 x 16

        out = self.group2(out)
        # out is batch x 8 x 16 x 16

        out = torch_func.avg_pool2d(out, 2)
        # out is batch x 8 x 8 x 8

        out = self.group2_to_group3(out)
        # out is batch x 16 x 8 x 8

        out = self.group3(out)
        # out is batch x 16 x 8 x 8

        out = out.sum(axis=(2, 3))
        # out is batch x 16

        out = self.linear(out)
        # outis batch x num_classes

        return out


def main():
    plt.style.use("dark_background")

    V_target = torch.zeros(5, 5, dtype=torch.float64)
    V_target[V_target.shape[0] // 2, 1:-1] = 1.0

    I_input = torch.zeros(*V_target.shape, dtype=torch.float64)
    I_input[0, 0] = 1
    I_input[-1, -1] = -1

    model = MultiSolvePoissonTensor(1, 5, 5, 2)
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

    im_vout2 = axs[1, 2].imshow(
        0 * V_target,
        cmap=plt.cm.hot,
    )
    axs[1, 2].set_title("Observed second voltages")

    losses = []
    for it in range(1000):
        # Take one optimization step.
        optimizer.zero_grad()
        V_out = model(I_input[None, None, :, :])
        assert V_out.shape == (1, 2, 5, 5), V_out.shape
        # There is only one batch.
        V_out = V_out[0]

        loss = ((V_out[0] - V_target) ** 2).sum() + ((V_out[1] - V_target) ** 2).sum()
        loss.backward()
        optimizer.step()

        losses.append(float(loss))

        # Update the plots
        h_losses.set_data(range(len(losses)), losses)
        axs[1, 0].set_xlim((0, len(losses)))
        axs[1, 0].set_ylim((min(losses), max(losses)))

        im_vout.set_data(V_out[0].detach().numpy())
        im_vout.autoscale()

        im_vout2.set_data(V_out[1].detach().numpy())
        im_vout2.autoscale()

        plt.pause(1e-4)
        plt.show(block=False)
    plt.show()


if __name__ == "__main__":
    main()