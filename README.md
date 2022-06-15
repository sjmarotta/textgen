# TNG Text Generator

## Setup

#### Requirements
(at least, I ran this on a machine with the following):
* A machine with a GPU
* Python 3.7 or higher
* Keras 2.3.1
* numpy 1.18.3

#### Configuring your Environment
The training script uses the full set of ST:TNG scripts to train its text generator.
I did not include the TNG scripts for copyright reasons. But you, lucky reader,
can download them all for free [right here](https://www.st-minutiae.com/resources/scripts/scripts_tng.zip)!

I can provide a script that will automatically download the ZIP file and populate the
data directory. For now, you'll have to set it up manually. Unzip the contents of the
ZIP file to a directory called `data/tng_scripts` off of your main repo directory.

## Training the text generator

Run the `tng_train.py` script. It should be that simple. It will generate model
weights files that the text generator can use to generate text.

The training script can start from a previous set of weights instead of a brand-new
untrained model. To initialize the model with an existing set of weights, uncomment
line 64 and rename the weights file you want to use to `tng_init_weights.hdf5`.
Yes, you have to set it in the code. I'd have made it more user-friendly, but
I just wrote this for my own amusement and to share with anyone who might want
to kick it around.

## Running the text generator

Run the `tng_gen.py` script. First, make sure that the weights file you want to
use has been renamed to `tng_init_weights.hdf5`. Also, update line 103 with a
string of seed text you want to use to start the generated text. I usually start
with a phrase wrapped in escaped double-quotation marks, because the generator
treats that like an episode title.

## Notes
This project is a little rough around the edges. I don't have a .gitignore, and
the scripts have a lot of hard-coded stuff. Like I said, this is something I 
wrote to amuse myself and to share with other like-minded folks. Enjoy.