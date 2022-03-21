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

def get_events_file(events_home_dir, subject_id, run):
    events_file = events_home_dir + 'sub-' + subject_id  + '/run-' + str(run).zfill(2) + '/events.csv'
    #events_file = 'events_run_' + str(i) + '.csv'

    events = pd.read_csv(events_file)
    events = events.drop('Unnamed: 0', 1)
    return events

def fit_glm(subject_id, run):
    events = get_events_file(subject_id, run)
    tr = 1.25
    n_scans = image.load_img(fmri_image[run-1]).shape[-1]
    frame_times = np.arange(n_scans) * tr
    motion = np.cumsum(np.random.randn(n_scans, 6), 0)
    add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    design_matrix = make_first_level_design_matrix(frame_times, events, 
                                               drift_model='polynomial', drift_order=3, 
                                               add_regs=motion, add_reg_names=add_reg_names, 
                                               hrf_model='spm')
    fmri_glm_model = FirstLevelModel(t_r=1.25, minimize_memory=False, noise_model='ar1', mask_img=mask_image[run-1])
    fmri_glm_model.fit(fmri_image[run-1], design_matrices=design_matrix)
    print("run done: ", run)
    return fmri_glm_model, design_matrix

def compute_no_diff_contrasts(glm, run):
    z_maps = list()
    conditions_label = list()
    sessions_label = list()
    events = get_events_file(subject_id, run)
    conditions = events.trial_type.unique()

    for condition_ in conditions:
        z_maps.append(glm[run-1].compute_contrast(condition_))
        conditions_label.append(condition_)
        sessions_label.append(str(run))
    return z_maps, conditions_label, sessions_label

def get_movement_minus_wait_contrasts(design_matrices, glms):
    z_map_movement_minus_wait = list()
    movement_minus_wait_labels = list()
    for run in range(1, 11):
        contrast_matrix = np.eye(design_matrices[run-1].shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(design_matrices[run-1].columns)])        
        movement_contrasts = basic_contrasts['movement_153'] + basic_contrasts['movement_207'] + basic_contrasts['movement_45'] + basic_contrasts['movement_99'] - basic_contrasts['wait']

        z_map_movement_minus_wait.append(glms[run-1].compute_contrast(movement_contrasts))
        movement_minus_wait_labels.append('Movement minus wait, run_' + str(run).zfill(2))
    
    return z_map_movement_minus_wait, movement_minus_wait_labels

def get_prep_minus_wait_contrasts(design_matrices, glms):
    z_map_prep_minus_wait = list()
    prep_minus_wait_labels = list()
    for run in range(1, 11):
        contrast_matrix = np.eye(design_matrices[run-1].shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(design_matrices[run-1].columns)])        
        movement_contrasts = basic_contrasts['go_153_prep'] + basic_contrasts['go_207_prep'] + basic_contrasts['go_45_prep'] + basic_contrasts['go_99_prep'] + basic_contrasts['nogo_153_prep'] + basic_contrasts['nogo_207_prep'] + basic_contrasts['nogo_45_prep'] + basic_contrasts['nogo_99_prep'] - basic_contrasts['wait']

        z_map_prep_minus_wait.append(glms[run-1].compute_contrast(movement_contrasts))
        prep_minus_wait_labels.append('Prep minus wait, run_' + str(run).zfill(2))
    
    return z_map_prep_minus_wait, prep_minus_wait_labels


def plot_contrast_maps(z_maps, z_map_no, condition_label, display_mode = 'ortho', correction = 'bonferroni', alpha = 0.05):    
    _, threshold = threshold_stats_img(
        z_maps[z_map_no], alpha= alpha, height_control=correction)
    print('Bonferroni-corrected, p<0.05 threshold: %.3f' % threshold)

    plot_map = plot_stat_map(z_maps[z_map_no], threshold = threshold, 
                             black_bg=True, display_mode=display_mode, draw_cross=False,
                             title = condition_label[z_map_no] + ' '+ correction + ' corrected, p<0.05')
    masker.fit(z_maps[z_map_no])
    #report = masker.generate_report()
    #plot_map.add_contours(image.index_img(atlas_filename, 11))
    plotting.show()
    return plot_map, masker
    