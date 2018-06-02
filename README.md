# SeqPeelSort
A spike sorting algorithm for single sensillum recordings
## Author
Georg Raiser, University of Konstanz, Dept. Neurobiology, grg2rsr@gmail.com

## Summary
_SeqPeelSort_ is a spike sorting algorithm for single sensillum recordings.

Single sensillum recordings are a standard technique in insect neuroscience. In such electrophysiological recordings, a single electrode is inserted into a sensory sensillum, and the transepithelial potential is recorded. This technique is extensively (but not only) used in _Drosophila_ olfaction studies (see for example Lin et al. (2015) and citations therein).

Since insect olfactory sensilla can house multiple sensory neurons, the action potentials of all those neurons are recorded with the single electrode simultaneously. The individual units usually display different spike shapes and amplitudes. Additionally, the amplitudes of some units is substantially decreased when the unit exhibits a high firing rate.

As any spike sorting algorithm, _SeqPeelSorts_ goal is to detect spikes and based on their waveform, assign them to a unit. Different to most recording situations for which current spike sorting algorithms are designed, the recording configuration of SSR leads to specific constrains: The number of electrodes is only one, the number of units is low and known, but their spike shapes are quite variable. _SeqPeelSort_ is designed handle these specific requirements while being lightweight and simple.

## Installation
### Install python 3.6 Anaconda distribution
download and install [Anaconda for python 3.6](https://www.anaconda.com/download/)

### create a python environment
the `env` subfolder contains a conda environment file that you can use to create a environment with all required packages. Do do so, open a terminal and in the subfolder `env` type
```shell
conda env create -f SeqPeelSort.yml -n SeqPeelSort
```
activate the environment (linux/mac: `source activate SeqPeelSort`, windows: `activate SeqPeelSort`).

More on managing anaconda environments can be found in the  [official docs](https://conda.io/docs/user-guide/tasks/manage-environments.html) or [this blog post](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)

## Usage
### Data format
SeqPeelSort is build around the python electrophysiology packages [neo](http://neuralensemble.org/neo/) and [elephant](http://elephant.readthedocs.io/en/latest/). As its data format, it uses [Nix](http://g-node.github.io/nix/).

The input file is a `.nix` file containing a `Block` with one or more `Segments` (a detailed explanation of the terms can be found [here](http://neo.readthedocs.io/en/0.5.2/core.html)). If the `Block` consists of several `Segments`, they are considered to be individual trials and sorting is restricted to those, which speeds up processing and limits consumed memory substantially.

An example conversion function for converting Spike2 `.smr` files to a `.nix` file can be found in `examples/smr2nix.py`. Additional converters can be written upon request, contact me and provide an example recording.

### Configuration file
This files specifies the settings and parameters used for the spike sorting. Description of the individual parameters can be found in the example configuration file: `examples/example_config.ini`. XX link it

### Example usage
+ [Download an example recording](https://web.gin.g-node.org/grg2rsr/SeqPeelSort_example_data/src/master/example.smr) in the `.smr` format into the subfolder `examples`.
+ Convert the `.smr` file to a `.nix` file. To do so, run `python smr2nix.py example.smr`. Inside `smr2nix.py`, an example conversion function is given that can serve as a template for own data conversion functions.
+ Run SeqPeelSort on the `.nix` file. `python SeqPeelSort.py ../examples/example_config.ini`
+ The output is a `.nix` file named `example_sorted.nix` that contains the sorted spike trains. Alternatively, the output can be written in the `.csv` format, which generates a separate file for each unit/segment combination containing the time stamps of the predicted spikes.
+ The subfolder `plots` contains diagnostic plots of relevant computations of the algorithm (detailed below).

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

## Benchmarking
To my knowledge, no ground truth data set for SSR recordings is available. If this is wrong and you know of or have a dataset of intracellular recordings of multiple neurons within one sensillum, please contact me.

In order to nevertheless estimate the sorting performance of the algorithm, a simulated dataset can be generated from extracted templates, and the sorting performance of the algorithm is assesed with this artificial ground truth dataset. This is done by generating an artificial spike train (homogeneous poisson process with a removal of spikes that would fall into the refractory period of a previous spike) and placing spike waveforms at the random time points.

Two benchmarks are available: 1) `benchmark_tm_params.py` can be used to determine optimal template matching parameters for the recording. Here, either threshold or percentile parameters of the algorithm are swept over the valid range in a brute force manner while the other is held fixed. 2) `benchmark_rates.py` can be used to estimate absolute template matching performance. For each unit combination, the firing rates are swept and the generated data is subjected to the template matching and peeling steps of the algorithm.

The results are placed in a subfolder `benchmark`.


### Benchmarking example
Run the algorithm once with properly set bounds in the config file and potentially adaptive thresholding. After the extraction of the templates, a file `results/templates.dill` is written. This file contains the templates that are used for the Benchmark. Run the `benchmark.py` script:
```shell
python benchmark_tm_params.py config.ini
```
this generates generates plots from which you can choose good values for the algorithms main parameters.

Then, update the `config.ini` with those values and run
```shell
python benchmark_rates.py config.ini
```
to get an estimate of the performance of the template matching step, including plots comparing the template matching result to a purely thresholding based result.

## Future development
+ multithreaded template match for performance increase
+ expanding IO options based on users needs

## References
Lin, C. C., & Potter, C. J. (2015). Re-classification of Drosophila melanogaster trichoid and intermediate sensilla using fluorescence-guided single sensillum recording. PloS one, 10(10), e0139675.
