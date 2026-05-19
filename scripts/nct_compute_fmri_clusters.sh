########################################################################################################################
# settings
k_start=2
k_end=20
scriptsdir='/mnt/storage_ssd_raid/research_projects/intrinsic_neural_timescales/scripts'
########################################################################################################################

########################################################################################################################
# HCP-D
indir='/mnt/storage_ssd_raid/research_projects/intrinsic_neural_timescales/data/timeseries/HCP-D'
outdir='/mnt/storage_ssd_raid/research_projects/intrinsic_neural_timescales/data/nct/HCP-D'

for n_clusters in $(seq ${k_start} ${k_end}); do
    python ${scriptsdir}/nct_compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'HCP-D_Schaefer4007_task-rfMRIREST1PA_data-ts.npy' --file_prefix 'HCP-D_Schaefer4007_task-rfMRIREST1PA_' --n_clusters ${n_clusters}
    python ${scriptsdir}/nct_compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'HCP-D_Schaefer4007_task-tfMRICARITPA_data-ts.npy' --file_prefix 'HCP-D_Schaefer4007_task-tfMRICARITPA_' --n_clusters ${n_clusters}
done
########################################################################################################################

########################################################################################################################
# HCP-YA
indir='/mnt/storage_ssd_raid/research_projects/intrinsic_neural_timescales/data/timeseries/HCP-YA'
outdir='/mnt/storage_ssd_raid/research_projects/intrinsic_neural_timescales/data/nct/HCP-YA'

for n_clusters in $(seq ${k_start} ${k_end}); do
    python ${scriptsdir}/nct_compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'HCP-YA_Schaefer4007_task-rfMRIREST1LR_data-ts.npy' --file_prefix 'HCP-YA_Schaefer4007_task-rfMRIREST1LR_' --n_clusters ${n_clusters}
    python ${scriptsdir}/nct_compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'HCP-YA_Schaefer4007_task-tfMRIWMLR_data-ts.npy' --file_prefix 'HCP-YA_Schaefer4007_task-tfMRIWMLR_' --n_clusters ${n_clusters}
done
########################################################################################################################
