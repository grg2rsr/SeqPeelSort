# SeqPeelSort
A spike sorting algorithm for single sensillum recordings
## Author
Georg Raiser, University of Konstanz, Dept. Neurobiology, grg2rsr@gmail.com

## Summary
_SeqPeelSort_ is a spike sorting algorithm for single sensillum recordings (SSR).

Single sensillum recordings are a standard technique in insect neuroscience. In such electrophysiological recordings, a single electrode is inserted into a sensory sensillum, and the transepithelial potential is recorded. This technique is extensively (but not only) used in _Drosophila_ olfaction studies (see for example Hallem et al. (2004) or Lin et al. (2015) and citations therein).

Since insect olfactory sensilla can house multiple sensory neurons, the action potentials of all those neurons are recorded with the single electrode simultaneously. The individual units usually display different spike shapes and amplitudes. Additionally, the amplitudes of some units is substantially decreased when the unit exhibits a high firing rate.

As any spike sorting algorithm, _SeqPeelSorts_ goal is to detect spikes and based on their waveform, assign them to a unit. Different to most recording situations for which current spike sorting algorithms are designed, the recording configuration of SSR leads to specific constrains: The number of electrodes is only one, the number of units is known and low, but their spike shapes are quite variable. _SeqPeelSort_ is designed handle these specific constrains while being lightweight and simple.

## Installation
### Install Anaconda
Anaconda is a package managing software for python. Download and install [Anaconda](https://www.anaconda.com/download/) for your operating system. _SeqPeelSort_ will use python 3, but it doesn't matter which Anaconda version you install (recommended is 3).

### create a python environment
the `env` subfolder contains a conda environment file that you can use to create a environment with all required packages. Do do so, open a terminal and in the subfolder `env` type:

Linux/Mac:
```shell
conda env create -f SeqPeelSort.yml -n SeqPeelSort
```

Windows:
```shell
conda env create -f SeqPeelSort_win.yml -n SeqPeelSort
```

afterwards, activate the environment (linux/mac: `source activate SeqPeelSort`, windows: `activate SeqPeelSort`).

More on managing anaconda environments can be found in the  [official docs](https://conda.io/docs/user-guide/tasks/manage-environments.html) or [this blog post](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)

### Install the latest neo version
Note: this is only required until `neo==0.7` is available via pip.

in `src/neo`:
+ `pip uninstall neo` because `elephant` installs `neo==0.6`
+ `python setup.py develop`

## Usage
### Data format
SeqPeelSort is build around the python electrophysiology packages [neo](http://neuralensemble.org/neo/) and [elephant](http://elephant.readthedocs.io/en/latest/). As its data format, it uses the [Nix](http://g-node.github.io/nix/) format for electrophysiological data.

The input file is a `.nix` file containing a `Block` with one or more `Segments` (a detailed explanation of the terms can be found [here](http://neo.readthedocs.io/en/0.5.2/core.html)). If the `Block` consists of several `Segments`, they are considered to be individual trials and sorting is restricted to those, which speeds up processing and limits consumed memory substantially.

An example conversion function for converting Spike2 `.smr` files to a `.nix` file can be found in `examples/smr2nix.py`. Additional converters can be written upon request, if the data format can be handled by `neo.io`. Contact me and provide an example recording.

### Configuration file
This files specifies the settings and parameters used for the spike sorting. Description of the individual parameters can be found in the example configuration file: `examples/example_config.ini`.

### Example usage
+ [Download an example recording](https://web.gin.g-node.org/grg2rsr/SeqPeelSort_example_data/src/master/example.smr) in the `.smr` format into the subfolder `examples`.
+ Convert the `.smr` file to a `.nix` file. To do so, run `python smr2nix.py example.smr`. The file `smr2nix.py` contains an example conversion function, that can serve as a template for own data converters.
+ Run SeqPeelSort on the `.nix` file: `python SeqPeelSort.py ../examples/example_config.ini`. The `ini` file contains the path to the data.
+ The output is a `.nix` file named `example_sorted.nix` that contains the sorted spike trains. Alternatively, the output can be written in the `.csv` format, which generates a separate file for each unit/segment combination containing the time stamps of the sorted spikes.
+ The subfolder `plots` contains diagnostic plots of relevant computations of the algorithm (detailed below).

A more detailed usage is provided in the [USAGE.md](https://github.com/grg2rsr/SeqPeelSort/USAGE.md), along with a description of the settable parameters of the algorithm.

### Algorithm details
_SeqPeelSort_ is a template matching based spike sorter. Generally, a distribution of templates for each unit is generated, and the waveform of the best fitting template is subtracted from the recording in an iterative manner.

In detail, the following steps are applied:

1. median absolute derivation based thresholding to detect all spikes
2. sorting the detected spikes to units based on thresholding by user specified bounds
3. estimating the decrease of spike amplitude at high firing rates for each unit
4. calculation of time variable _adaptive_ thresholds based on the estimated decrease
5. resorting to units based on this adaptive thresholds
6. extracting spike waveforms (templates) based on the detected spikes of step 5.
7. performing a PCA on the templates from step 6., kicking outlier templates, and inverse PCA to generate simulated templates

Then, an iterative detecting and removing spikes ("_peeling_") loop starts with the unit of the largest amplitude:

1. using the simulated templates for template matching - resulting in a distribution of scores for each time point
2. reducing the distribution to a single value for each time point at a fixed percentile level
3. thresholding this time varying score, resulting in a new spike detection
4. based on these detected spike times and the best fitting template, a artificial voltage time series is generated and subtracted from the original recording.

the above steps above are then applied to the next smaller unit, using the "peeled" voltage for the next template matching step.

## Community Guidelines for future development
### expanding IO options based on users needs
Currently, more IO options are needed, but those would be added on a user by user basis. If you are willing to write an conversion from a data format to the `.nix`, have a look at the `smr2nix.py` file as a template.

### multithreaded template matching
In order to increase performance, the template matching step could be taken to a multithreaded computation. If you are interested in developing this, drop me a line - I have a semaphore based approach for a previous of the algorithm than can probably be adopted with not much effor.


## References
Hallem, E. A., Ho, M. G., & Carlson, J. R. (2004). The molecular basis of odor coding in the _Drosophila_ antenna. Cell, 117(7), 965-979.

Lin, C. C., & Potter, C. J. (2015). Re-classification of _Drosophila melanogaster_ trichoid and intermediate sensilla using fluorescence-guided single sensillum recording. PloS one, 10(10), e0139675.
