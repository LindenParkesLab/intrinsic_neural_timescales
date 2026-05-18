########################################################################################################################
# directories
projdir='/mnt/storage_ssd_raid/research_projects/intrinsic_neural_timescales'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data/timeseries/HCP-D'
outdir=${projdir}'/data/nct/HCP-D'

# k-means range (set both to the same value to run a single k)
k_start=2
k_end=20
########################################################################################################################

########################################################################################################################
# resting-state
for n_clusters in $(seq ${k_start} ${k_end}); do
    python ${scriptsdir}/nct_compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'HCP-D_Schaefer4007_task-rfMRIREST1PA_data-ts.npy' --file_prefix 'HCP-D_Schaefer4007_task-rfMRIREST1PA_' --n_clusters ${n_clusters}
done
########################################################################################################################

########################################################################################################################
# carit task
for n_clusters in $(seq ${k_start} ${k_end}); do
    python ${scriptsdir}/nct_compute_fmri_clusters.py --indir ${indir} --outdir ${outdir} --fmri_file 'HCP-D_Schaefer4007_task-tfMRICARITPA_data-ts.npy' --file_prefix 'HCP-D_Schaefer4007_task-tfMRICARITPA_' --n_clusters ${n_clusters}
done
########################################################################################################################
