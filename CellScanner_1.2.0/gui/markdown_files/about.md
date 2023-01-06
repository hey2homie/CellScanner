# Using CellScanner

### Training models

##### Classifier

To train your model, you will need to provide reference files in the "Training" window. It is recommended to provide at 
least one reference file for each class, although the more references you provide, the better the results will be. 
However, it is important to prioritize the quality of the reference files over their quantity. Additionally, the files 
used for training should be named either "Name_name-rest.fcs" or "Name-rest.fcs".

There are multiple settings that can be adjusted to improve the results However, they are optional. Here is the 
explanation of each option:
- **Number of epochs**. An epoch is a single pass through the entire training dataset. In general, the more epochs you 
use, the longer the training will take, but it may also result in a better model. On the other hand, if you use too 
many epochs, your model may overfit.
- **Batch size**. the batch size is the number of training examples used in one forward/backward pass. The larger the 
batch size, the more memory space you'll need. Batch size is also a factor in how long it takes to train a model. 
Smaller batch sizes are generally slower to train, because the model has to make more forward/backward passes to cover 
all the training examples. However, smaller batch sizes can also be more noise-resistant, since they average the error 
gradients over a smaller number of examples.
- **Learning rate**. The learning rate controls how much the model's weights are updated based on the loss gradient, 
with a higher learning rate leading to faster progress but also a higher risk of convergence issues, and a lower 
learning rate being slower but also less prone to such issues.
- **Learning rate scheduler**. The learning rate scheduler allows you to specify a schedule for the learning rate to 
change over time, with the possibility of starting with a higher learning rate and decreasing it as training progresses.

There is also an option called "Legacy NN" which, if enabled, will use a neural network architecture from a previous 
version of CellScanner.

Once your classifier has been trained, you can select it in the settings window. 

##### Autoencoder 

Autoencoder is the new approach to gating. To train an autoencoder, it is better to provide multiple monoculture files 
together with blank files. Same as before, the more references you provide, the better the results would be. However, 
from the experience, ratio of 10 to 1 (monoculture to blank) in terms of events is optimal.

There are some settings that are specific to autoencoder:
- **Number of clusters**. Training autoencoder is a semi-supervised learning task. Clusters, containing high number of
blanks observations, are removed to ensure that data is cleaned as much as possible.
- **Blank threshold**. This is the threshold for the number of blank events in a cluster. Clusters containing more blank
observations than the threshold will be removed.

##### General

After training model, the window with the training statistics will appear. You can consult it to see how well the model
was trained. It's important to return to the menu through the button because this action shuts down TensorBoard process,
which is independent of the CellScanner. Alternatively, it will continue running in background even when CellScanner is 
closed. Additionally, training statistics (same as in the visualisations) are saved in training logs directory as a 
.csv file.

Importantly, please select the Flow Cytometer that was used to collect the data prior model training. This is important
because autoencoder and classifier selection depends on selected FC. Question mark buttons in the settings menu can 
help you refresh memory which channels are available to use with the autoencoder model, and which labels classifier is
capable of predicting. 

The channels to drop option in the settings is only applicable to autoencoder. By default, it drops only "Time" channel
and preserves the rest. However, if you want to drop more channels, you can do so by specifying them there. When using
autoencoder for gating, the channels available for visualisations correspond to the features of the autoencoder. 

### Predicting

To predict results, first you need to select the model and gating approach. Additionally, there are "Reconstruction 
error" settings that are only applicable to autoencoder. They are used to filter out events that are predicted to be
blanks based on the reconstruction error. It's better to first see the distribution of reconstruction errors in the
model diagnostics and then decide on the threshold. However, with the Accuri datasets 0.3 was optimal. Another setting 
is "Probability threshold", which labels events as low-probability or not. The results of the prediction is class 
probability distribution, meaning that events are labeled as belonging to a certain class with a certain probability. 

After that you can just select files and see the results. After running prediction, it's possible to adjust 
reconstruction error prior saving results.

### Model diagnostics

Model diagnostics is a tool that allows you to see how well the model was trained. Please, provide at least one file per
class that model can predict. Also, include blank files too. Otherwise, is the error will bi raised and CellScanner 
will be closed.