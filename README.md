# 1. TF-Blocks

The scope of this repo is to provide a modularized architecture of tensorflow project so that
you can put more focus on each individual part more independently.


### 2.1 Motivation

Since tensorflow is a Symbolic Programming framework, quite different than Pytorch or Caffe2,
the computation, neural network architecture and training are totally four independent things!

Therefore I separate them into following four parts and also provide template module for each of them, 
as well as different other implementations such as CNN, UNET, ... etc. so that you can refer to to develop
your own ones. That's why I cal this repo `tf-blocks`.

I used this repo as basis for my actual daily researches to speed up prototyping and try my best to put everything
in a way of production ready quality.

**If you like it, please star it!**

**If you wanna contribute, feel free to submit Pull Request to make it better!**


# 2. Components

### 2.1 Graph Computation
 
I separate the computation part and placeholder the actual network architecture  so that when you need to investigate/develop
this part, you don't need to mix the improvements on graph computation with the network.


### 2.2 Network

Network is all about architecture and you may need to follow some constraints so that the graph module can be compatible
with this module. However, it's constructed in a way that you truly can just focus on the architecture.

### 2.3 Training

While you have network/graph computation done, all left is how you wanna train it, including:

* Which device you wanna run it? (CPU/GPU/Multi-Towers)

* How you wanna manage the data feed in/batch?

* How to config the tensorboard summary display?


### 2.4 DataGenerator

This part is all about data:

* How to load data from file path?

* How to pre-process the data?

* How to set the ratio of data?


# Run Example App

The scope of this repo is to make it way easier to build a customized training app which you can focus
on separate part.

For demo cases, I made a CNN base neural network pipeline with MNIST dataset API (cuz it's just free).

The demo app is composed of:

```
/apps/cnn_training_app.py
/config/cnn_config.py
/models/cnn_net.py
```

Basically, you can run it with:

```
# Navigate to /tf-blocks/

python apps/cnn_training_app.py

```