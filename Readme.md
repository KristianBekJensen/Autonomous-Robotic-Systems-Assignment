# Search Robot

To get all required libaries use: `pip install -r requirements.txt`

## Manuel Control

Boot up by running `main.py`

### Keybinds

<kbd>w</kbd> <kbd>a</kbd> <kbd>s</kbd><kbd>d</kbd> : Movement Controls

<kbd>1</kbd>: Visualise the parts of the map grid that the robot is currently observing with its sensors.

<kbd>2</kbd>: Draw lines to deteced landmarks.

<kbd>3</kbd>: Visalise the kalman filters Sigma.

<kbd>4</kbd>: Draws the robots estimated path in red.

<kbd>5</kbd>: Draw robots sensors.

<kbd>6</kbd>: Visualize the mapping accuracy.

## Training

The `config.py` file allows to change parameters for Training.

To start the training run `evolution.py`

A file containg the current genration wil be saved in the poppulation folder every 10 genrations or after traning is complete.

To visualize the best indvidual in a genration use `replay_best.py`. Input your filepath on line 6 to a genration and run `replay_best.py`.
