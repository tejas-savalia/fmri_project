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
import os


def get_fmri_files(subject_id):
    fmri_image = list()
    mask_image = list()
    for run in range(1, 11):
        if run%2 == 0:
            task = 'rotate'
        else:
            task = 'straight'
        fmri_file = f"data/derivatives/sub-{subject_id}/func/sub-{subject_id}_task-{task}_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        fmri_image.append(fmri_file)
        mask_file = f"data/derivatives/sub-{subject_id}/func/sub-{subject_id}_task-{task}_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        mask_image.append(mask_file)
    anat_image = f"data/derivatives/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    return fmri_image, mask_image, anat_image


events_home_dir = '/home/vm01/Documents/fmri_project_copy/fmri_behavioral_data/'

events_home_dir = '/home/vm01/Documents/fmri_project_copy/fmri_behavioral_data/'

def get_events_file(subject_id, run):
    events_file = events_home_dir + 'sub-' + subject_id  + '/run-' + str(run).zfill(2) + '/events.csv'
    #events_file = 'events_run_' + str(i) + '.csv'

    events = pd.read_csv(events_file)
    events = events.drop('Unnamed: 0', 1)
    
    #Uncomment below to collapse across go/nogo
    events['trial_type'] = events['trial_type'].replace({'nogo_45_prep':'45_prep', 
                                                 'nogo_99_prep':'99_prep',
                                                 'nogo_153_prep': '153_prep',
                                                 'nogo_207_prep':'207_prep',
                                                 'go_45_prep':'45_prep', 
                                                 'go_99_prep':'99_prep',
                                                 'go_153_prep': '153_prep',
                                                 'go_207_prep':'207_prep'                                                    
                                                })
    
#     Uncomment below to separate early and late preparation periods. 
#     late_prep_onset = (events[events.trial_type.str.contains('prep')]['onset'] + events[events.trial_type.str.contains('prep')]['duration']/2).values
#     late_prep_duration = (events[events.trial_type.str.contains('prep')]['duration']/2).values
#     late_prep_trial_type = 'late_'+ events[events.trial_type.str.contains('prep')]['trial_type'].values
#     late_prep_df = pd.DataFrame({'onset': late_prep_onset, 
#                             'trial_type': late_prep_trial_type, 
#                             'duration': late_prep_duration
#                             })
#     events.loc[events.trial_type.str.contains('prep'), 'duration'] = events.loc[events.trial_type.str.contains('prep'), 'duration']/2
#     events = events.append(late_prep_df).reset_index().drop('index', axis = 1)
    
#     Uncomment to do trial_wise PREP events
    events.loc[events.trial_type.str.contains('prep'), 'trial_type'] = events[events.trial_type.str.contains('prep')]['trial_type'].values + '_'+ np.arange(40).astype(str)

    #Uncomment to do trial_wise MOVEMENT events
#     events.loc[events.trial_type.str.startswith('movement'), 'trial_type'] = events[events.trial_type.str.startswith('movement')]['trial_type'].values + '_'+ np.arange(20).astype(str)

    
    #Uncomment to do odd-even events
    
    #sorted_ev = events.sort_values(['trial_type', 'onset'])
    #sorted_ev.loc[sorted_ev.trial_type.str.contains('prep'), 'trial_type'] = sorted_ev[sorted_ev.trial_type.str.contains('prep')].trial_type + '_' + np.tile(['odd', 'even'], 20)
    #events = sorted_ev.sort_index()
    
    return events


def fit_glm(subject_id, run):
    events = get_events_file(subject_id, run)
    fmri_image, mask_image, anat_image = get_fmri_files(subject_id)

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

def compute_no_diff_contrasts(subject_id, glm, run):
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
        #movement_contrasts = basic_contrasts['go_153_prep'] + basic_contrasts['go_207_prep'] + basic_contrasts['go_45_prep'] + basic_contrasts['go_99_prep'] + basic_contrasts['nogo_153_prep'] + basic_contrasts['nogo_207_prep'] + basic_contrasts['nogo_45_prep'] + basic_contrasts['nogo_99_prep'] - basic_contrasts['wait']
        movement_contrasts = basic_contrasts['45_prep'] + basic_contrasts['99_prep'] + basic_contrasts['153_prep'] + basic_contrasts['207_prep'] - basic_contrasts['wait']
        z_map_prep_minus_wait.append(glms[run-1].compute_contrast(movement_contrasts))
        prep_minus_wait_labels.append('Prep minus wait, run_' + str(run).zfill(2))
    
    return z_map_prep_minus_wait, prep_minus_wait_labels

def run_glms(subject_id):
    glms = list()
    glm_labels = list()
    design_matrices = list()
    for run in range(1, 11):
        g, d = fit_glm(subject_id, run)
        glms.append(g)
        glm_labels.append(run)
        design_matrices.append(d)
    return glms, glm_labels, design_matrices


def save_cont_maps(subject_id):
#     glms, glm
    print(subject_id)
    glms, glm_labels, design = run_glms(subject_id)
    for run in range(1, 11):
        run_no = str(run).zfill(2)
        #Modify paths based on the type of beta maps
        if not os.path.exists(f'analyses_results/sub-{subject_id}/beta_maps/trial_level/run-{run_no}/'):
            os.makedirs(f'analyses_results/sub-{subject_id}/beta_maps/trial_level/run-{run_no}/')
        z, l, s = compute_no_diff_contrasts(subject_id, glms, run)

        prep_45 = []
        prep_99 = []
        prep_153 = []
        prep_207 = []
        prep = {}

        
#         late_prep_45 = []
#         late_prep_99 = []
#         late_prep_153 = []
#         late_prep_207 = []
#         late_prep = {}

        
#         move_45 = []
#         move_99 = []
#         move_153 = []
#         move_207 = []
#         move = {}


        for i in range(len(l)):
            if l[i].startswith('45'):
                prep_45.append(z[i])
            elif l[i].startswith('99'):
                prep_99.append(z[i])
            elif l[i].startswith('153'):
                prep_153.append(z[i])
            elif l[i].startswith('207'):
                prep_207.append(z[i])
                
#         for i in range(len(l)):
#             if l[i].startswith('late_45'):
#                 late_prep_45.append(z[i])
#             elif l[i].startswith('late_99'):
#                 late_prep_99.append(z[i])
#             elif l[i].startswith('late_153'):
#                 late_prep_153.append(z[i])
#             elif l[i].startswith('late_207'):
#                 late_prep_207.append(z[i])

#         for i in range(len(l)):
#             if l[i].startswith("movement"):
#                 if '45' in  l[i]:
#                     move_45.append(z[i])
#                 elif '99' in l[i]:
#                     move_99.append(z[i])
#                 elif '153' in l[i]:
#                     move_153.append(z[i])
#                 elif '207' in l[i]:
#                     move_207.append(z[i])

        prep['45'] = image.concat_imgs(prep_45)
        prep['99'] = image.concat_imgs(prep_99)
        prep['153'] = image.concat_imgs(prep_153)
        prep['207'] = image.concat_imgs(prep_207)

#         late_prep['45'] = image.concat_imgs(late_prep_45)
#         late_prep['99'] = image.concat_imgs(late_prep_99)
#         late_prep['153'] = image.concat_imgs(late_prep_153)
#         late_prep['207'] = image.concat_imgs(late_prep_207)
        
        
#         move['45'] = image.concat_imgs(move_45)
#         move['99'] = image.concat_imgs(move_99)
#         move['153'] = image.concat_imgs(move_153)
#         move['207'] = image.concat_imgs(move_207)

        for i in prep.keys():
            prep[i].to_filename(f"analyses_results/sub-{subject_id}/beta_maps/trial_level/run-{run_no}/prep_{i}.nii.gz")

#         for i in late_prep.keys():
#             late_prep[i].to_filename(f"analyses_results/sub-{subject_id}/beta_maps/trial_level/run-{run_no}/prep_late_prep/late_prep_{i}.nii.gz")

            
#         for i in move.keys():
#             move[i].to_filename(f"analyses_results/sub-{subject_id}/beta_maps/trial_level/run-{run_no}/move_{i}.nii.gz")

        print("Run done: ", run)