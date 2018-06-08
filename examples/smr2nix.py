import sys
import os
import neo
from neo import Spike2IO
from neo import NixIO
import quantities as pq
import copy

"""
Data format requirements:
The neo.core.Block that is read by SeqPeelSort must:
1) consist of one or more segments - each segment is treated as a seperate trial
and is sorted seperately
2) in each segment, analogsignals[0] must contain the recording
"""


def smr2nix(path, cut_times):
    """
    converts a .smr file to a Nix file containing the neo.core.Block suitable
    for the SeqPeelSort run.

    This is a user specific function tailored to my recording settings, and
    servers here as an example.

    Args:
        path (str): the path to the .smr file
        cut_time (Quantity)(2,): specifying how much time before and after a
            trigger is defining the time of a trial. In my case, a trigger
            occured at the onset of stimulation
    """

    Reader = Spike2IO(filename=path)
    Blk = Reader.read_block()

    AnalogSignal = Blk.segments[0].analogsignals[0]
    AnalogSignal = AnalogSignal.rescale('uV')

    # Slice segments containing the trials
    Segments = []
    for t in Blk.segments[0].events[0].times:
        segment = neo.core.Segment()
        Asig = AnalogSignal.time_slice(*(t+cut_times))
        # see https://github.com/NeuralEnsemble/python-neo/issues/536
        Asig.annotations = dict()

        segment.analogsignals.append(Asig)
        Segments.append(segment)

    # put back to Block
    Blk.segments = Segments

    # DEBUG
    # for seg in Blk.segments:
        # print(seg.analogsignals[0][0])
        # print(seg.analogsignals[0].t_start, id(seg.analogsignals[0]), seg.analogsignals[0][0])

    # workaround for NixIO bug
    # https://github.com/NeuralEnsemble/python-neo/issues/535#issuecomment-389503181
    for chx in Blk.channel_indexes:
        chx.analogsignals = []

    # write to Nix
    outpath = os.path.splitext(path)[0]+'.nix'

    with NixIO(filename=outpath) as Writer:
        Writer.write_block(Blk)

if __name__ == '__main__':
    smr2nix(sys.argv[1], [-4, 4]*pq.s)
