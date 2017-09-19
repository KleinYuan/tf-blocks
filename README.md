# CNN Training Pipeline

- [X] Highly modularized and configurable that you can configure each step

- [X] Highly flexible that you can easily invoke your modifications in any places without impacting others

- [X] Highly testable that you can test every single pieces

- [X] Well organized that you don't get confused by what's going on and why you name this like this this this


# Naming Conventions Descriptions

- [X] net: short for networks, basically, define the architecture, like how the tensor should flow

- [X] graph: you may find a better description [here](https://www.tensorflow.org/programmers_guide/graphs)
however in this repo, in summary, you can view graph is everything computation ready however without worrying abou the specific details of networks

# How to setup and run?

### Setup

Basically, just run:

```

# Strongly recommend that you use an anaconda env, with python 2.7
pip install tensorflow
pip install -r requirements.txt
echo "Ta da !"

# Run generic_app.py needs this and if you don't wanna be bothered by opencv, just delete that app.
conda install -c https://conda.anaconda.org/menpo opencv3

```

Then, go ahead and download datasets from https://www.kaggle.com/c/facial-keypoints-detection

And, organize them under data/kaggle_face

### Run

```
# Navigate to apps/
python face_recognigtion_app.py
```


### Generic Pipeline

For generic training pipeline, you can config whatever in `generic_training_app.py`.

Then simply run it:

```
# Navigate to apps/
python generic_training_app.py

```