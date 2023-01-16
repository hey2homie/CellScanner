# Installation

### Unix

Please, clone the repository and install all dependencies that are listed in the `requirements.txt` file.

    $ git clone https://github.com/hey2homie/CellScanner.git
    $ pip install -r requirements.txt

### Windows 

[TO BE UPDATED]

# Using CellScanner

### Launching 

In the directory `./CellScanner_1.2.0/` are two files: `start.py` and `interface.py`. First launches graphical interface
for the CellScanner, while interface is used to run CellScanner inside the command. To launch GUI, please type in the 
terminal:
    
    $ python start.py

The basic syntax to use CellScanner through command line is:
    
    $ python interface.py COMMAND [OPTIONS]

Available `COMMANDs` are: `predict`, `train`, `validate`, `settings`. To run prediction or validate model, type in the 
following command:

    $ python interface.py predict -p PATH_TO_FILE/FOLDER
    $ python interface.py validate -p PATH_TO_FILE/FOLDER

To train model, in additional to positional argument `-p` (path), `-n` (name of the model), and `-m` (type of the model)
should be specified. However, if not, the prompt will appear. Example of the command to train autoencoder:

    $ python interface.py train -p PATH_TO_FILE/FOLDER -n NAME_OF_MODEL -m autoencoder

To see current settings, type in:

    $ python interface.py settings -s

Use the following command to change settings:

    $ python interface.py settings -c PARAMETER

Prompt will apper with available options for the `PARAMETER` if any. An example:

    $ python interface.py settings -c vis_dims

Additionally, parameter value can be specified directly in the command-line to avoid prompts:

    $ python interface.py settings -c vis_dims -v UMAP
    $ python interface.py settings -c num_umap_cores -v 4

Lastly, you can call help using `-h` flag.

    $ python interface.py -h

### Training models

##### Classifier

To train your model, you will need to provide reference files in the `Training` window. It is recommended to provide at 
least one reference file for each class, although the more references you provide, the better the results will be. 
However, it is important to prioritize the quality of the reference files over their quantity. Additionally, add more or 
less equal amount of files for the same label. Lastly, the files used for training should be named either 
`Name_name-rest.fcs` or `Name-rest.fcs`. However, the labels can be later manually modified in 
`./config/models_info.yml`.

There are multiple settings that can be adjusted to improve the results However, they are optional. Here is the 
explanation of each option:
- `Number of epochs`. An epoch is a single pass through the entire training dataset. In general, the more epochs you 
use, the longer the training will take, but it may also result in a better model. On the other hand, if you use too 
many epochs, your model may overfit.
- `Batch size`. the batch size is the number of training examples used in one forward/backward pass. The larger the 
batch size, the more memory space you'll need. Batch size is also a factor in how long it takes to train a model. 
Smaller batch sizes are generally slower to train, because the model has to make more forward/backward passes to cover 
all the training examples. However, smaller batch sizes can also be more noise-resistant, since they average the error 
gradients over a smaller number of examples.
- `Learning rate`. The learning rate controls how much the model's weights are updated based on the loss gradient, 
with a higher learning rate leading to faster progress but also a higher risk of convergence issues, and a lower 
learning rate being slower but also less prone to such issues.
- `Learning rate scheduler`. The learning rate scheduler allows you to specify a schedule for the learning rate to 
change over time, with the possibility of starting with a higher learning rate and decreasing it as training progresses.

There is also an option called `Legacy NN` which, if enabled, will use a neural network architecture from a previous 
version of CellScanner.

Once your classifier has been trained, you can select it in the settings window. 

##### Autoencoder 

Autoencoder is the new approach to gating. To train an autoencoder, it is better to provide multiple monoculture files 
together with blank files. Same as before, the more references you provide, the better the results would be. However, 
from the experience, ratio of 10 to 1 (monoculture to blank) in terms of events is optimal.

There are some settings that are specific to autoencoder:
- `Number of clusters`. Training autoencoder is a semi-supervised learning task. Clusters, containing high number of
blanks observations, are removed to ensure that data is cleaned as much as possible.
- `Blank threshold`. This is the threshold for the number of blank events in a cluster. Clusters containing more blank
observations than the threshold will be removed.

##### General

After training model, the window with the training statistics will appear. You can consult it to see how well the model
was trained. It's important to return to the menu through the button because this action shuts down TensorBoard process,
which is independent of the CellScanner. Alternatively, it will continue running in background even when CellScanner is 
closed. If the page is not loaded, please press "Reload". Additionally, training statistics (same as in the 
visualisations) are saved in `training_logs/` as a `.csv` file.

Importantly, please select the Flow Cytometer that was used to collect the data prior model training. This is important
because autoencoder and classifier selection depends on selected FC. Question mark buttons in the settings menu can 
help you refresh memory which channels are available to use with the autoencoder model, and which labels classifier is
capable of predicting. 

The channels to drop option in the settings is only applicable to autoencoder. By default, it drops only "Time" channel
and preserves the rest. However, if you want to drop more channels, you can do so by specifying them there. When using
autoencoder for gating, the channels available for visualisations correspond to the features of the autoencoder. 

### Predicting

To predict results, first you need to select the model and gating approach. Nothing else is needed. However, there is 
two optional settings as well:
- `Reconstruction error` that is only applicable when autoencoder is selected. It is used to filter out events that are 
predicted to be blanks based on the reconstruction error. It's better to first see the distribution of reconstruction 
errors in the model diagnostics and then decide on the threshold. However, with the Accuri datasets 0.3 was optimal.
- Another setting is `Probability threshold`, which labels events as low-probability or not. The results of the 
prediction is class probability distribution, meaning that events are labeled as belonging to a certain class with a 
certain probability. For example, the output of the model for two class classification is `[0.32, 0.68]` for `label_1`, 
`label_2`, respectively. These probabilities saying that observation is 32% likely to be `label_1` and 68% likely to be
`label_2`. 
- Number of cores to compute UMAP controls how many CPU cores will be involved in the computation of UMAP. However, it's
limited by your CPU. 

After that you can just select files and see the results. After running prediction, it's possible to adjust 
reconstruction error prior saving results. However, at the moment, same threshold is applied to all files.

Results are saved as 3D or 2D plot of the predictions, MSE plot for each file in the specified directory. Additionally, 
`cell_counts.txt` is saved. Text file contains statistics for each file. For each predicted species the following 
structure is used:
- Number of cells. Total number of cells that are labeled as given class.
- Percentage. Number of cells / amount of events in the file.
- Number of blanks. How many of predicted events (Number of cells) are predicted to be blanks.
- Percentage of blanks. Number of blanks / number of cells.
- Number of high probability. Number of cells with probability higher than given threshold.
- Percentage of high probability. Number of high probability / number of cells.
- Results after gating. Number of cells - number of blanks

If ``Machine`` gating is selected, please provide reference files for the binary classifier. Files should include some
bacterial monocultures, preferably of same species used for the prediction, as well as some blank files. Additionally,
please add ``_ref`` to the end of the reference file's name. 

### Model diagnostics

Model diagnostics is a tool that allows you to see how well the model was trained. Please, provide at least one file per
class that model can predict including blanks files. Otherwise, is the error will bi raised and CellScanner 
will be closed. Files' name should follow same notation as when providing reference files for the training.

If ``Machine`` gating is selected, continue as described in the ``Predicting`` section.