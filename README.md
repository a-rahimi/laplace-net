# Replacing convolutional layers with physics

Implements a resnet-like model where the conv layers are replaced with layers that solve Laplace's equation. Why do this? Because
most physical systems (like the diffusion of heat, or how charges distribute in a network of capacitors or resistors) implicitly solve
Laplace's equation as they reach a relaxed state. That means we could replace these layers with easy-to-build physical systems.

The model is trained on CIFAR10 and gets about 90% accuracy.
