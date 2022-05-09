from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    root_dir = '/home/airlab/PycharmProjects/pythonProject5/data/GOT-10K'
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    tracker = TrackerSiamFC()
    tracker.train_over(seqs)