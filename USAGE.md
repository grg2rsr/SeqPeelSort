# SeqPeelSort.py
`SeqPeelSort.py` is the main python script that runs the algorithm on your data. This is the only script that needs to be called in order to sort the spikes in the recording, such as `python SeqPeelSort.py ../examples/example_config.ini` from the command line interface.

## Input
### Arguments
`SeqPeelSort.py` takes the path to the `config.ini` file as a first and only command line argument. Within the `config.ini`, the path to the actual data is specified, along will all parameters that are needed for the algorithm. See below for more information.

## Output
Assuming your data file is named `data_filename.nix`, the following files are written:
+ `data_filename_sorted.nix`: This file's content are those of the initial `.nix` files with the spike trains written to each segment. The spike trains are given labels as specified within in the `config.ini` headers as unit names (see below). Alternatively, a `.csv` file can be written if set in the `config.ini`. In this case. a `.csv` file for each unit/segment combination is produced that contains the timestamps of the sorted spikes.

+ A subfolder `plots` containing:
  + a spike histogram, useful for selecting appropriate first boundaries of each units amplitudes
  + a plot of each units templates extracted and valid templates
  + an estimated decrease of firing amplitude as function of firing frequency for each unit where adaptive thresholding is set
  + a plot of each segment with the sorted spikes and zoomed version of that plot (if set in the `config.ini`). These plots are useful for assessing sorting quality and sorting algorithm parameter tweaking

# config.ini
The `config.ini` controls several options of the algorithm. It is strucutred in sections, a `[general]` section and one for eachc unit (such as '[ab3a]','[ab3B]' etc.). See the [example_config.ini](https://github.com/grg2rsr/SeqPeelSort/examples/example_config.ini) file for an example and default values.

## General Parameters
### experiment_name
The name of the experiment. This is used for writing metadata in the `nix` file.

### data_path
Location of the data file relative to the path of this file.

### mad_thresh
Median absolute deviation for initial spike detection. This defines the minimum deviation from the median for a peak to be considered as a spike and selected as a template for the first detection run. Higher values lead to more conservative spike detection, low values might be necessary in noisy recordings. 4-5 usually works.

### highpass_freq
Before anything else, the recording is high-pass filtered by a 4-th order butterworth signal with a cutoff at the specified frequency, in in [Hz]. 100 Hz usually works fine.

### peak_mode
Direction of the voltage deflection of the recorded spikes. Can be set to `negative` or `positive`, depending on your recording.

### fig_format
The desired file format of diagnostic output figures such as `svg`,`pdf` or `png`.

### zoom
If set, the diagnostic plots in the time domain are plotted once for a larger overview, and once zoomed to an area of the specified value in [ms]. If left empty, no zoomed plots are produced.

### output
The output file format. Can be set to `nix` or  or `csv`. In the case of `nix`, the sorted spike trains are written directly to the nix file, in the case of `csv`, a csv file for each unit/segment combination is produced that contains the timestamps of the sorted spikes.

## Unit specific parameters
The following parameters have to be specified for each unit present in the recording copy paste each of the following blocks and name them according to the unit that you want to sort

### bounds
The bounds that will be used for initial the thresholding, in [uV]. This value can be left empty, in this case only a spike histogram is produced that can be used to look for meaningful values to put into here.

### adaptive_threshold
If set to yes, adaptive thresholding is applied for this unit. This increases sorting performance for units that chance their amplitude for high firing rates, but also increases computational complexity. Advised to set to no if a unit does not show strong changes.

### wsize
The window size for templates, symmetric in [ms], This has to be an even number. Generally, 2 ms are fine.

### n_templates
The number of templates used for the PCA. The larger this number, the more data is used for estimating the distribution of possible spike shapes.

### n_sim
The number of templates that are then simulated from the statistical template model. As a template matching is computed for each of these simulated templates, this parameter is both sensitive in terms of computational complexity and in sorting precision. Recommended to start with 100.

### n_comp
the number of PCA components that are used. 5 are usually fine, can be increased for very complex spike shape variations.

### tm_percentile
the percentile level of the template match scores that is used for detecting spikes. A value of 80 has been a good starting point for me. See the Benchmarking section below for computationally finding good values.

### tm_thresh
the threshold that is applied to the percentile level of the template match scores that is used for detecting spikes. The closer this value to one, the more conservative is the sorting: More missed spikes, but fewer are erroneously predicted. A value of 0.5-0.8 is recommended. See the Benchmarking section below for computationally finding good values.

# Benchmarking
To my knowledge, no ground truth data set for SSR recordings is available. If this is wrong and you know of or have a dataset of intracellular recordings of multiple neurons within one sensillum, please contact me, such data would be of great help.

In order to nevertheless estimate the sorting performance of the algorithm, a simulated dataset can be generated from extracted templates, and the sorting performance of the algorithm is assessed with this artificial ground truth dataset. This is done by generating an artificial spike train (homogeneous poisson process with a removal of spikes that would fall into the refractory period of a previous spike) and placing spike waveforms at the random time points.

Two benchmarks are available:
1. `benchmark_tm_params.py` can be used to determine optimal template matching parameters for the recording. Here, either threshold or percentile parameters of the algorithm are swept over the valid range in a brute force manner while the other is held fixed.
2. `benchmark_rates.py` can be used to estimate absolute template matching performance. For each unit combination, the firing rates are swept and the generated data is subjected to the template matching and peeling steps of the algorithm.

The results are placed in a subfolder `benchmark`. Examples can be found [here](https://web.gin.g-node.org/grg2rsr/SeqPeelSort_example_data/src/master/SSR_ab3_example_results)

## Benchmarking example
Run the algorithm once with properly set bounds in the config file. After the extraction of the templates, a file `results/templates.dill` is written. This file contains the templates that are used for the Benchmark. Run the `benchmark.py` script:
```shell
python benchmark_tm_params.py config.ini
```
this generates generates plots from which you can choose good values for the algorithms main parameters: the template matching percentile level, and the threshold level that the template matching score has to cross in order to set a predicted spike at this timepoint.

Update the `config.ini` with those values and run
```shell
python benchmark_rates.py config.ini
```
This will simulate data with different rates for the units and runs the template matching part of SeqPeelSort on it to get an estimate of the performance, including plots comparing the template matching result to a purely thresholding based result.
