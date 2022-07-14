import torch
import matplotlib.pyplot as plt
from bindsnet         import encoding
from bindsnet.network import Network, nodes, topology, monitors


network = Network(dt=1.0)  # Instantiates network.
X = nodes.Input(100)  # Input layer.
Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.
# Spike monitor objects.
M1 = monitors.Monitor(obj=X, state_vars=['s'])
M2 = monitors.Monitor(obj=Y, state_vars=['s'])
# Add everything to the network object.
network.add_layer(layer=X, name='X')
network.add_layer(layer=Y, name='Y')
network.add_connection(connection=C, source='X', target='Y')
network.add_monitor(monitor=M1, name='X')
network.add_monitor(monitor=M2, name='Y')
# Create Poisson-distributed spike train inputs.
data = 15 * torch.rand(100)  # Generate random Poisson rates for 100 input neurons.
train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.
# Simulate network on generated spike trains.
inputs = {'X' : train}  # Create inputs mapping.
network.run(inputs=inputs, time=5000)  # Run network simulation.
# Plot spikes of input and output layers.
