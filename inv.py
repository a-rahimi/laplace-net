import matplotlib.pylab as plt
import numpy as np
import torch

# Solve
#   min_X || X^-1 A - I ||
# by gradient descent. The optimum is reached when X = A.

class Invert(torch.nn.Module):
    def __init__(self, X: torch.Tensor):
        super().__init__()
        self.X = torch.nn.Parameter(X)

    def forward(self, A: torch.Tensor):
        assert A.shape == self.X.shape

        return torch.sum((torch.linalg.solve(self.X, A) - torch.eye(*A.shape))**2)

def main():
    A = torch.rand(5, 5, dtype=torch.float64)
    A += torch.eye(*A.shape)

    model = Invert(torch.eye(*A.shape, dtype=torch.double))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

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

    Xhat = np.linalg.inv(A)
    print("Xhat =\n", Xhat)
    print("loss(Xhat) = 0")

    plt.plot(losses)
    plt.show()

main()
