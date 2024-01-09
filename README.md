# ROB-498-Robot-Learning

This is a project that aims to control a robot arm to perform a planar pushing task(with and without obstacles) using Neural ODEs for learning dynamics.

## Installation

1. Clone the repository.

    ```
    git clone https://github.com/Prabhakaran-Gokul/ROB-498-Robot-Learning.git
    ```
2. Create virtual environment. For example:
    ```
    conda create -n myenv
    conda activate myenv
     ```
3. Install requirements.
    ```
    ./install.sh
    ```

## Run Demo
Run Demo file.
```
python3 demo.py
```

## Results
The example below shows a result of training the NeuralODE model with the following hyperparameters:
- 6x100 input layer, 100x100 hidden layer, 100x6 output layer
- training_integration_method:  dopri5
- learning rate: 0.001

<u>Environment without obstacles</u>

![](./plots/2023-04-21%2002:23:19.080359/pushing_visualization_free.gif)

<u>Environment with an obstacle</u>

![](./plots/2023-04-21%2002:23:19.080359/pushing_visualization_obs.gif)
