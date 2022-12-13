from typing import OrderedDict
import torch, torch.nn as nn
import snntorch as snn


batch_size = 32


data_path='home/h/Desktop/snn/mnist'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
print('downloading data')
mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
print('finished downloading data')
# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

from snntorch import surrogate

beta = 0.9  # neuron decay rate
spike_grad = surrogate.fast_sigmoid()

#  Initialize Network
layers = [nn.Conv2d(1, 8, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(8, 16, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(16*4*4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)]


dict_layers = []

for i in range(len(layers)):
    dict_layers.append((str(i),layers[i]))

net = nn.Sequential(OrderedDict(dict_layers)
                    ).to(device)



#net = Net(layers)

from snntorch import utils

def forward_pass(net, data, num_steps):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)

import snntorch.functional as SF

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

from snntorch import backprop
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt


num_epochs = 3
num_steps = 25  # run for 25 time steps

loss_hist = []
acc_hist = []

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec = forward_pass(net, data, num_steps)
        #print(spk_rec)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # print every 25 iterations
        if i % 25 == 0:
          print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
          #print(data)
          #print(targets)
          # check accuracy on a single batch
          print(spk_rec.size())
          spk_results = torch.stack(tuple(spk_rec), dim=0)[:, 0, :].to('cpu')
          print(spk_results.size())
          fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
          labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
          #splt.spike_count(spk_results, fig, ax, labels, num_steps = num_steps, time_step=1e-3)
          #plt.show()
          with torch.no_grad():
            anim = splt.spike_count(spk_results, fig, ax, labels, animate=True, interpolate=5, num_steps = num_steps, time_step=1e-3)
          anim.save("spike_bar.gif")
          acc = SF.accuracy_rate(spk_rec, targets)
          acc_hist.append(acc)
          print(f"Accuracy: {acc * 100:.2f}%\n")

        # uncomment for faster termination
        if i >= 200:
            break


'''
# training loop
for epoch in range(num_epochs):

    avg_loss = backprop.BPTT(net, train_loader, num_steps=num_steps,
                          optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)

    print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")
'''
def test_accuracy(data_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec = forward_pass(net, data, num_steps)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

print(f"Test set accuracy: {test_accuracy(test_loader, net, num_steps)*100:.3f}%")
