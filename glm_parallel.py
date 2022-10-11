import nilearn
from nilearn.plotting import plot_carpet, plot_glass_brain, plot_anat, plot_stat_map, plot_design_matrix, plot_epi, plot_contrast_matrix
from nilearn import image, masking, input_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table, make_glm_report
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker
from nilearn import datasets
from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn import surface
from nilearn.decoding import Decoder
import seaborn as sns
from glm_functions import *
import multiprocessing
import sys

if __name__ == '__main__':
    pool = multiprocessing.Pool(4)
    participants = sys.argv[1:]
    
    pool.map(save_cont_maps, participants)
