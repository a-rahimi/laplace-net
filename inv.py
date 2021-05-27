import matplotlib.pylab as plt
import numpy as np
import torch

# Solve
#   min_X || X^-1 A - I ||
# by gradient descent when X is a 5x5 matrix with zeros everywhere
# except the diagonal and the upper right and lower left entries.

class Invert(torch.nn.Module):
    def __init__(self, X: torch.Tensor):
        super().__init__()
        self.X = torch.nn.Parameter(X)

        
    def forward(self, A: torch.Tensor):
        assert A.shape == (5, 5)

        X_full = torch.sparse_coo_tensor(
            [[0, 1, 2, 3, 4, 0, 4], [0, 1, 2, 3, 4, 4, 0]], self.X, (5,5)
        ).to_dense() + torch.eye(5, 5)

        return torch.sum((torch.linalg.solve(X_full, A) - torch.eye(*A.shape)) ** 2)


def main():
    A = torch.rand(5, 5, dtype=torch.float64)
    A += torch.eye(*A.shape)

    model = Invert(torch.ones(A.shape[0] + 2, dtype=torch.double))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    losses = np.zeros(1000)
    for it in range(losses.size):
        optimizer.zero_grad()
        loss = model(A)
        loss.backward()
        optimizer.step()

        losses[it] = float(loss)
    losses = losses[:it]

    print("A =\n", A)
    print("X =\n", model.X)
    print("loss(X) = ", float(model(A)))
    print("dloss/dX = ", model.X.grad)

    plt.plot(losses)
    plt.show()


main()
