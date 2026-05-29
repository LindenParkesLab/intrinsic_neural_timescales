########################################################################################################################
# settings
nct_script='/mnt/storage_ssd_raid/research_projects/nct_xr/scripts/compute_optimized_control_energy.py'
projdir='/mnt/storage_ssd_raid/research_projects/intrinsic_neural_timescales'
seg='Schaefer4007'
########################################################################################################################

########################################################################################################################
# HCP-D
dataset='HCP-D'
nct_dir=${projdir}'/data/nct/'${dataset}
outdir=${projdir}'/results/'${dataset}'/nct'

A_file=${nct_dir}'/'${dataset}'_'${seg}'_adjacency_mean.npy'

# Update k_combined after choosing k_rest and k_task from the elbow plot
# k_combined = k_rest + k_task + n_contrasts  (currently 5 + 5 + 6 = 16)
k_combined=16
fmri_clusters_file=${nct_dir}'/'${dataset}'_'${seg}'_REST-CARIT-kmeans-contrasts_fmri_clusters_k-'${k_combined}'.npy'
file_prefix=${dataset}'_'${seg}'_Am'

python ${nct_script} \
    --outdir ${outdir} \
    --A_file ${A_file} \
    --fmri_clusters_file ${fmri_clusters_file} \
    --file_prefix ${file_prefix} \
    --reference_state 'xf' \
    --compact_save True
########################################################################################################################

########################################################################################################################
# HCP-YA
dataset='HCP-YA'
nct_dir=${projdir}'/data/nct/'${dataset}
outdir=${projdir}'/results/'${dataset}'/nct'

A_file=${nct_dir}'/'${dataset}'_'${seg}'_adjacency_mean.npy'

# Update k_combined after choosing k_rest and k_task from the elbow plot
# k_combined = k_rest + k_task + n_contrasts  (currently 7 + 7 + 8 = 22)
k_combined=22
fmri_clusters_file=${nct_dir}'/'${dataset}'_'${seg}'_REST-WM-kmeans-contrasts_fmri_clusters_k-'${k_combined}'.npy'
file_prefix=${dataset}'_'${seg}'_Am'

python ${nct_script} \
    --outdir ${outdir} \
    --A_file ${A_file} \
    --fmri_clusters_file ${fmri_clusters_file} \
    --file_prefix ${file_prefix} \
    --reference_state 'xf' \
    --compact_save True
########################################################################################################################
