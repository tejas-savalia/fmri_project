def read_data(subject_id):
    fmri_images = list()
    #mask_image = list()
    for run in range(1, 11):
        if run%2 == 0:
            task = 'rotate'
        else:
            task = 'straight'
        fmri_file = f"data/derivatives/sub-{subject_id}/func/sub-{subject_id}_task-{task}_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        fmri_images.append(fmri_file)
    return fmri_images

def extract_prep_events_and_indices(events):
    #extracting prep events
    prep_events = events[events['trial_type'].str.contains('prep')].reset_index().drop('index', 1)

    #Renaming conditions column
    prep_events['target_location'] = '207'
    prep_events.loc[prep_events['trial_type'].str.contains('45'), 'target_location'] = '45'
    prep_events.loc[prep_events['trial_type'].str.contains('99'), 'target_location'] = '99'
    prep_events.loc[prep_events['trial_type'].str.contains('153'), 'target_location'] = '153'
    prep_events = prep_events.drop('trial_type', 1)
    prep_events = prep_events.sort_values(by='onset')
    indices = np.floor(prep_events['onset']/1.25).astype(int).values + 3

    return prep_events, indices

