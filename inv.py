import matplotlib.pylab as plt
import numpy as np
import torch

#   min_D f(D) = || D A - I ||^2
# The opt is reached at D such that for all diagonal dD, we have
#   0 = df = ||D A - I + dD A||^2 - ||D A - I||^2
#     = 2 tr (DA-I)' dD A
#     = 2 tr A (DA-I)' dD
# This implies
#     diag(AA'D) = diag(A)
#     diag(AA') D = diag(A)
#     D = diag(A) / diag(AA')

class Invert(torch.nn.Module):
    def __init__(self, X: torch.Tensor):
        super().__init__()
        self.X = torch.nn.Parameter(X)

    def forward(self, A: torch.Tensor):
        assert A.shape == (self.X.shape[0], self.X.shape[0])

        return torch.sum((torch.diag(self.X) @ A - torch.eye(*A.shape))**2)

def main():
    A = torch.eye(5, 5, dtype=torch.double)
    A += .5 * torch.rand(A.shape, dtype=A.dtype)

    model = Invert(torch.zeros(A.shape[0], dtype=torch.double))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    losses = np.zeros(10000)
    for it in range(losses.size):
        optimizer.zero_grad()
        loss = model(A)
        loss.backward()
        optimizer.step()

        losses[it] = float(loss)
        #if it>1 and losses[it] > losses[it-1]:
            #break
    losses = losses[:it]

    print("A =\n", A)
    print("X =\n", model.X)
    print("loss(X) = ", float(model(A)))

    Xhat = torch.diag(A) / torch.diag(A @ A.T)
    print("Xhat =\n", Xhat)
    print("loss(Xhat) = ", float(Invert(Xhat)(A)))

    plt.plot(losses)
    plt.show()

main()
