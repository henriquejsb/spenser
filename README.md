# SPENSER: Towards a NeuroEvolutionary Approach for Convolutional Spiking Neural Networks

Spiking Neural Networks (SNNs) have attracted recent interest due to their energy efficiency and biological plausibility. However, the performance of SNNs still lags behind traditional Artificial Neural Networks (ANNs), as there is no consensus on the best learning algorithm for SNNs. Best-performing SNNs are based on ANN to SNN conversion or learning with spike-based backpropagation through surrogate gradients. The focus of recent research has been on developing and testing different learning strategies, with hand-tailored architectures and parameter tuning. Neuroevolution (NE), has proven successful as a way to automatically design ANNs and tune parameters, but its applications to SNNs are still at an early stage. DENSER is a NE framework for the automatic design and parametrization of ANNs, based on the principles of Genetic Algorithms (GA) and Structured Grammatical Evolution (SGE). In this paper, we propose SPENSER, a NE framework for SNN generation based on DENSER, for image classification on the MNIST and Fashion-MNIST datasets. SPENSER generates competitive performing networks with a test accuracy of 99.42% and 91.65% respectively.

KEYWORDS: spiking neural networks, neuroevolution, SPENSER, computer vision

If you want to use this work, please cite us:

Henrique Branquinho, Nuno Lourenço, and Ernesto Costa. 2023. ** SPENSER:Towards a NeuroEvolutionary Approach for Convolutional Spiking Neural Networks** . In _Genetic and Evolutionary Computation Conference Companion (GECCO ’23 Companion), July 15–19, 2023, Lisbon, Portugal_.
https://arxiv.org/abs/2305.10987
```
python main.py -c config.json -g cnn.grammar -d [mnist|fashion_mnist|cifar-10]
```

README in progress.

If you have any questions feel free to contact me at branquinho@dei.uc.pt !

