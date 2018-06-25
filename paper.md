---
title: 'SeqPeelSort: a spike sorting algorithm for single sensillum recordings'
tags:
  - Python
  - Neuroscience
  - Olfaction
  - Electrophysiology
  - Single sensillum recordings
authors:
  - name: Georg Raiser
    orcid: 0000-0002-9716-3513
    affiliation: "1"
affiliations:
 - name: Dept. of Neurobiology, University of Konstanz, Germany
   index: 1
date: 12 June 2018
bibliography: paper.bib
---

# Summary
_SeqPeelSort_ is a spike sorting algorithm for single sensillum recordings (SSR).

Single sensillum recordings are a standard technique in insect neuroscience. In such electrophysiological recordings, a single electrode is inserted into a sensory sensillum and the transepithelial potential is recorded. This technique is extensively (but not only) used in _Drosophila_ olfaction studies (see for example @hallem2004molecular or @lin2015re and citations therein).

Since insect olfactory sensilla can house multiple sensory neurons, the action potentials of all those neurons are recorded with the single electrode simultaneously. The individual units usually display different spike shapes and amplitudes. Additionally, the amplitudes of some units is substantially decreased when the unit exhibits a high firing rate.

As any spike sorting algorithm, _SeqPeelSorts_ goal is to detect spikes and based on their waveform, assign them to a unit. Different to most recording situations for which current spike sorting algorithms are designed, the recording configuration of SSR leads to specific constrains: The number of electrodes is only one, the number of units is known and low, but their spike shapes are quite variable. _SeqPeelSort_ is designed handle these specific constrains while being lightweight and simple.

# References
