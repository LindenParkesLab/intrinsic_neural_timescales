import os, sys, time, getpass
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
from nilearn.glm.first_level import make_first_level_design_matrix, compute_regressor
from nilearn.glm.first_level import FirstLevelModel

username = getpass.getuser()
if username == 'lindenmp':
    sys.path.extend(['/home/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lindenmp/research_projects/nctpy/src'])
elif username == 'lp756':
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/nctpy/src'])

from snaplab_tools.derivs import compute_acf, compute_alff, compute_rlfp

class GroupLevelAnalysis:
    """

    """
    
    def __init__(self, data_dict, sa_axis):
        """
        Initialize with data dictionary and SA axis
        
        Parameters:
        -----------
        data_dict : dict
            {condition_name: numpy_array(n_subjects, n_regions)}
        sa_axis : array-like
            SA axis values (1=sensorimotor, 400=association)
        """
        self.data_dict = data_dict
        self.sa_axis = np.array(sa_axis)
        self.condition_names = list(data_dict.keys())
        self.n_subjects, self.n_regions = list(data_dict.values())[0].shape
        
        # Create SA axis deciles (10% each)
        self.sa_deciles = pd.qcut(self.sa_axis, q=10, labels=[f'D{i+1}' for i in range(10)])
        
        # Create long format dataframe
        self._create_dataframe()
        self.subject_ids = np.unique(list(self.df['subject']))
        
        print(f"Initialized: {self.n_subjects} subjects, {self.n_regions} regions, {len(self.condition_names)} conditions")
    
    def _create_dataframe(self):
        """Create long-format dataframe"""
        df_list = []
        for condition, data in self.data_dict.items():
            for subj in range(self.n_subjects):
                for region in range(self.n_regions):
                    df_list.append({
                        'subject': f'S{subj:03d}',
                        'condition': condition,
                        'region': region,
                        'sa_axis': self.sa_axis[region],
                        'sa_decile': self.sa_deciles[region],
                        'value': data[subj, region]
                    })
        self.df = pd.DataFrame(df_list)
        
    
    def _analyze_condition_changes_helper(self, data, return_mean=True):
        region_mean = []
        region_var = []
        sa_corr = []
        for subject in tqdm(self.subject_ids):
            vals = data[data['subject'] == subject]['value']
            region_mean.append(np.mean(vals))
            region_var.append(np.var(vals))
            sa_corr.append(sp.stats.spearmanr(self.sa_axis, vals)[0])

        region_mean = np.array(region_mean)
        region_var = np.array(region_var)
        sa_corr = np.array(sa_corr)

        if return_mean:
            region_mean = region_mean.mean()
            region_var = region_var.mean()
            sa_corr = sa_corr.mean()
            
        subject_var = []
        for region in range(self.n_regions):
            vals = data[data['region'] == region]['value']
            subject_var.append(np.var(vals))

        subject_var = np.array(subject_var)
        sa_corr_var = sp.stats.spearmanr(self.sa_axis, subject_var)[0]
        
        return region_mean, region_var, sa_corr, sa_corr_var
    
        
    def analyze_condition_changes(self, baseline=None):
        """
        Analyze changes in between-subjects variance and SA-axis coupling relative to baseline
        """
        
        if baseline is None:
            rest_conditions = [c for c in self.condition_names if 'rest' in c.lower()]
            baseline = rest_conditions[0] if rest_conditions else self.condition_names[0]
        print("Analyzing condition changes relative to {0}...".format(baseline))
        
        baseline_data = self.df[self.df['condition'] == baseline]
        baseline_region_mean, baseline_region_var, baseline_sa_corr, baseline_sa_corr_var = self._analyze_condition_changes_helper(baseline_data)

        results = {}
        for condition in self.condition_names:
            condition_data = self.df[self.df['condition'] == condition]
            condition_region_mean, condition_region_var, condition_sa_corr, condition_sa_corr_var = self._analyze_condition_changes_helper(condition_data)
            region_mean_ratio = condition_region_mean / baseline_region_mean
            region_var_ratio = condition_region_var / baseline_region_var
            sa_corr_delta = condition_sa_corr - baseline_sa_corr
            sa_corr_var_delta = condition_sa_corr_var - baseline_sa_corr_var
            
            # Subject variance ratio (condition/baseline)
            baseline_subj_var = np.var(baseline_data.groupby('subject')['value'].mean())
            condition_subj_var = np.var(condition_data.groupby('subject')['value'].mean())
            subject_var_ratio = condition_subj_var / baseline_subj_var
            
            results[condition] = {
                'region_mean_ratio': region_mean_ratio,
                'region_var_ratio': region_var_ratio,
                'sa_corr_delta': sa_corr_delta,
                'sa_corr_var_delta': sa_corr_var_delta,
                'subject_var_ratio': subject_var_ratio
            }
        
        return results, baseline

    def analyze_variance_along_sa_axis(self, use_decile=False):
        """
        Analyze variance components across SA axis deciles
        Returns data for Q3: Variance Along SA Axis plot
        """
        print("Analyzing variance along SA axis...")
        
        if use_decile is True:
            results = pd.DataFrame(index=self.condition_names, columns=self.sa_deciles)

            for condition in self.condition_names:
                print(condition)
                condition_data = self.df[self.df['condition'] == condition]
                for decile in self.sa_deciles:
                    decile_data = condition_data[condition_data['sa_decile'] == decile]
                    
                    # Total variance in this condition and decile
                    total_var = np.var(decile_data['value'])

                    # Between-subjects variance in this decile
                    subject_means = decile_data.groupby('subject')['value'].mean()
                    between_subjects_var = np.var(subject_means)
                    
                    results.loc[condition, decile] = between_subjects_var / total_var
        else:
            region_ids = self.df['region'].unique()
            results = pd.DataFrame(index=self.condition_names, columns=region_ids)

            for condition in self.condition_names:
                print(condition)
                condition_data = self.df[self.df['condition'] == condition]
                for region in region_ids:
                    region_data = condition_data[condition_data['region'] == region]['value']
                    
                    # Between-subjects variance in this region
                    between_subjects_var = np.var(region_data)
                    
                    results.loc[condition, region] = between_subjects_var
        
        return results


class SubjectLevelINTsMeasures:
    """
    Calculate subject-level measures of intrinsic neural timescales (INTs) that can be 
    correlated with behavioral variables from HCP dataset.
    
    Designed to work with pre-loaded df_behavior DataFrame and allow specification
    of behavioral columns for targeted correlation analysis.
    """
    
    def __init__(self, data_dict, sa_axis, df_behavior, baseline_condition=None):
        """
        Initialize with neural data, SA axis, and behavioral data
        
        Parameters:
        -----------
        data_dict : dict
            {condition_name: numpy_array(n_subjects, n_regions)}
        sa_axis : array
            SA (sensory-association) axis values for regions
        df_behavior : DataFrame
            Pre-loaded behavioral data with Subject column (pre-matched subjects)
        baseline_condition : str, optional
            Baseline condition name. If None, auto-detects rest condition.
        """
        self.data_dict = data_dict
        self.sa_axis = np.array(sa_axis)
        self.condition_names = list(data_dict.keys())
        self.n_subjects, self.n_regions = list(data_dict.values())[0].shape
        
        # Store behavioral data and subject IDs (pre-matched)
        self.df_behavior = df_behavior.copy()
        self.subject_ids = df_behavior['Subject'].astype(str).tolist()
        
        # Identify baseline condition
        if baseline_condition is None:
            rest_conditions = [c for c in self.condition_names if 'rest' in c.lower() or 'REST' in c]
            self.baseline = rest_conditions[0] if rest_conditions else self.condition_names[0]
        else:
            self.baseline = baseline_condition
            
        # Task conditions (non-baseline)
        self.task_conditions = [c for c in self.condition_names if c != self.baseline]
        
        print(f"Initialized SubjectLevelINTsMeasures:")
        print(f"  Neural data: {self.n_subjects} subjects, {self.n_regions} regions")
        print(f"  Behavioral data: {len(self.df_behavior)} subjects, {len(self.df_behavior.columns)} measures")
        print(f"  Subject IDs: {self.subject_ids[:5]}... (showing first 5)")
        print(f"  Baseline condition: {self.baseline}")
        print(f"  Task conditions: {self.task_conditions}")
    
    def calculate_individual_spatial_variability_changes(self, return_extra=False):
        """
        For each subject and condition: How much does their spatial brain variability change from baseline?
        
        Returns:
        --------
        measures_dict : dict
            Keys: '{condition}_spatial_var_change', '{condition}_spatial_var_ratio'
            Values: arrays of shape (n_subjects,)
        """
        baseline_data = self.data_dict[self.baseline]
        measures_dict = {}
        
        print("  Calculating spatial variability changes...")
        for condition in self.task_conditions:
            task_data = self.data_dict[condition]
            
            spatial_var_ratios = []
            spatial_var_changes = []
            spatial_range_ratios = []
            spatial_range_changes = []
            
            for subj in range(self.n_subjects):
                # Subject's spatial variability in baseline vs task
                baseline_spatial_var = np.var(baseline_data[subj, :])
                task_spatial_var = np.var(task_data[subj, :])
                spatial_var_ratios.append(task_spatial_var / baseline_spatial_var)
                spatial_var_changes.append(task_spatial_var - baseline_spatial_var)
                
                # Subject's spatial range in baseline vs task
                baseline_spatial_range = np.ptp(baseline_data[subj, :])
                task_spatial_range = np.ptp(task_data[subj, :])
                spatial_range_ratios.append(task_spatial_range / baseline_spatial_range)
                spatial_range_changes.append(task_spatial_range - baseline_spatial_range)
            
            measures_dict[f'{condition}_spatial_var_ratio'] = np.array(spatial_var_ratios)
            measures_dict[f'{condition}_spatial_range_ratio'] = np.array(spatial_range_ratios)
            if return_extra:
                measures_dict[f'{condition}_spatial_var_change'] = np.array(spatial_var_changes)
                measures_dict[f'{condition}_spatial_range_change'] = np.array(spatial_range_changes)
        
        return measures_dict
    
    def calculate_individual_sa_coupling_changes(self, return_extra=False):
        """
        For each subject and condition: How much does their SA axis coupling change from baseline?
        
        Returns:
        --------
        measures_dict : dict
            Keys: '{condition}_sa_coupling_change', '{condition}_sa_coupling_baseline', '{condition}_sa_coupling_task'
            Values: arrays of shape (n_subjects,)
        """
        baseline_data = self.data_dict[self.baseline]
        measures_dict = {}
        
        print("  Calculating SA coupling changes...")
        for condition in self.task_conditions:
            task_data = self.data_dict[condition]
            sa_coupling_changes = []
            sa_coupling_baseline = []
            sa_coupling_task = []
            
            for subj in range(self.n_subjects):
                # Subject's SA axis correlation in baseline vs task
                baseline_sa_corr, _ = spearmanr(self.sa_axis, baseline_data[subj, :])
                task_sa_corr, _ = spearmanr(self.sa_axis, task_data[subj, :])
                
                sa_coupling_changes.append(task_sa_corr - baseline_sa_corr)
                sa_coupling_baseline.append(baseline_sa_corr)
                sa_coupling_task.append(task_sa_corr)
            
            measures_dict[f'{condition}_sa_coupling_change'] = np.array(sa_coupling_changes)
            if return_extra:
                measures_dict[f'{condition}_sa_coupling_baseline'] = np.array(sa_coupling_baseline)
                measures_dict[f'{condition}_sa_coupling_task'] = np.array(sa_coupling_task)
        
        return measures_dict
    
    def calculate_individual_sa_differential_speedup(self, return_extra=False):
            """
            For each subject and condition: Calculate SA differential speedup in INTs.
            
            INTs speedup = task INTs - baseline INTs (more negative = faster timescales)
            SA differential speedup = association_speedup - sensorimotor_speedup
            
            Returns:
            --------
            measures_dict : dict
                Keys: '{condition}_sa_differential_speedup', '{condition}_sa_speedup_correlation'
                Values: arrays of shape (n_subjects,)
            """
            baseline_data = self.data_dict[self.baseline]
            measures_dict = {}
            
            # Define region types using SA axis percentiles
            sa_percentiles = np.percentile(self.sa_axis, [10, 90])
            sensorimotor_mask = self.sa_axis <= sa_percentiles[0]  # Bottom 10%
            association_mask = self.sa_axis >= sa_percentiles[1]   # Top 10%
            
            print("  Calculating SA differential speedup...")
            print(f"    Sensorimotor regions (SA ≤ {sa_percentiles[0]:.0f}): {np.sum(sensorimotor_mask)} regions")
            print(f"    Association regions (SA ≥ {sa_percentiles[1]:.0f}): {np.sum(association_mask)} regions")
            
            for condition in self.task_conditions:
                task_data = self.data_dict[condition]
                
                sa_differential_speedups = []
                association_speedups = []
                sensorimotor_speedups = []
                sa_speedup_correlations = []
                sa_speedup_rank_correlations = []
                overall_speedups = []
                
                for subj in range(self.n_subjects):
                    # Calculate INT change (task - baseline)
                    # Negative change = speedup (INTs became faster)
                    int_change = task_data[subj, :] - baseline_data[subj, :]
                    
                    # Mean speedup in each region type
                    association_speedup = np.mean(int_change[association_mask])
                    sensorimotor_speedup = np.mean(int_change[sensorimotor_mask])
                    
                    # Main measure: Differential speedup
                    # More negative = greater speedup in association regions
                    sa_differential_speedup = association_speedup - sensorimotor_speedup
                    
                    # Alternative measure: Correlation with SA axis
                    # More negative = greater speedup in association regions
                    sa_speedup_correlation, _ = pearsonr(self.sa_axis, int_change)
                    sa_speedup_rank_correlation, _ = spearmanr(self.sa_axis, int_change)
                    
                    # Overall speedup
                    overall_speedup = np.mean(int_change)
                    
                    sa_differential_speedups.append(sa_differential_speedup)
                    association_speedups.append(association_speedup)
                    sensorimotor_speedups.append(sensorimotor_speedup)
                    sa_speedup_correlations.append(sa_speedup_correlation)
                    sa_speedup_rank_correlations.append(sa_speedup_rank_correlation)
                    overall_speedups.append(overall_speedup)
                
                measures_dict[f'{condition}_sa_speedup_correlation'] = np.array(sa_speedup_correlations)
                measures_dict[f'{condition}_sa_speedup_rank_correlation'] = np.array(sa_speedup_rank_correlations)
                measures_dict[f'{condition}_sa_differential_speedup'] = np.array(sa_differential_speedups)
                if return_extra:
                    measures_dict[f'{condition}_association_speedup'] = np.array(association_speedups)
                    measures_dict[f'{condition}_sensorimotor_speedup'] = np.array(sensorimotor_speedups)
                    measures_dict[f'{condition}_overall_speedup'] = np.array(overall_speedups)
            
            return measures_dict

    def calculate_individual_sa_corr_differences(self, return_extra=False):
            """
            For each subject and condition: Calculate correlation between SA axis and INTs and subtract.
            
            Returns:
            --------
            measures_dict : dict
                Keys: '{condition}_sa_differential_speedup', '{condition}_sa_speedup_correlation'
                Values: arrays of shape (n_subjects,)
            """
            baseline_data = self.data_dict[self.baseline]
            measures_dict = {}
            
            print("  Calculating SA corr differences...")

            for condition in self.task_conditions:
                task_data = self.data_dict[condition]
                
                sa_rank_correlation_deltas = []
                
                for subj in range(self.n_subjects):
                    sa_rank_correlation_baseline, _ = spearmanr(self.sa_axis, baseline_data[subj, :])
                    sa_rank_correlation_condition, _ = spearmanr(self.sa_axis, task_data[subj, :])
                    sa_rank_correlation_delta = sa_rank_correlation_condition - sa_rank_correlation_baseline
                    
                    sa_rank_correlation_deltas.append(sa_rank_correlation_delta)
                
                measures_dict[f'{condition}_sa_rank_correlation_deltas'] = np.array(sa_rank_correlation_deltas)
            
            return measures_dict

    def calculate_individual_rank_changes(self, return_extra=False):
        """
        For each subject and condition: How much does their relative standing in the group change?
        
        Returns:
        --------
        measures_dict : dict
            Keys: '{condition}_rank_change', '{condition}_baseline_rank', '{condition}_task_rank'
            Values: arrays of shape (n_subjects,)
        """
        baseline_data = self.data_dict[self.baseline]
        baseline_means = np.mean(baseline_data, axis=1)
        baseline_ranks = stats.rankdata(baseline_means) / len(baseline_means) * 100
        
        measures_dict = {}
        
        print("  Calculating rank changes...")
        for condition in self.task_conditions:
            task_data = self.data_dict[condition]
            task_means = np.mean(task_data, axis=1)
            task_ranks = stats.rankdata(task_means) / len(task_means) * 100
            
            rank_changes = task_ranks - baseline_ranks
            
            measures_dict[f'{condition}_rank_change'] = rank_changes
            if return_extra:
                measures_dict[f'{condition}_baseline_rank'] = baseline_ranks
                measures_dict[f'{condition}_task_rank'] = task_ranks
        
        return measures_dict
    
    def calculate_individual_mean_level_changes(self, return_extra=False):
        """
        For each subject and condition: How much does their overall brain activity level change?
        
        Returns:
        --------
        measures_dict : dict
            Keys: '{condition}_mean_change', '{condition}_mean_ratio'
            Values: arrays of shape (n_subjects,)
        """
        baseline_data = self.data_dict[self.baseline]
        baseline_means = np.mean(baseline_data, axis=1)
        
        measures_dict = {}
        
        print("  Calculating mean level changes...")
        for condition in self.task_conditions:
            task_data = self.data_dict[condition]
            task_means = np.mean(task_data, axis=1)
            
            mean_changes = task_means - baseline_means
            mean_ratios = task_means / baseline_means
            
            measures_dict[f'{condition}_mean_change'] = mean_changes
            if return_extra:
                measures_dict[f'{condition}_mean_ratio'] = mean_ratios
        
        return measures_dict
    
    def calculate_individual_pattern_stability(self, return_extra=False):
        """
        For each subject and condition: How much does their spatial brain pattern change?
        
        Returns:
        --------
        measures_dict : dict
            Keys: '{condition}_pattern_correlation', '{condition}_pattern_distance'
            Values: arrays of shape (n_subjects,)
        """
        baseline_data = self.data_dict[self.baseline]
        measures_dict = {}
        
        print("  Calculating pattern stability...")
        for condition in self.task_conditions:
            task_data = self.data_dict[condition]
            
            rank_correlations = []
            pattern_correlations = []
            pattern_distances = []
            
            for subj in range(self.n_subjects):
                # Rank between subject's baseline and task patterns
                rank_corr, _ = spearmanr(baseline_data[subj, :], task_data[subj, :])
                rank_correlations.append(rank_corr)

                # Correlation between subject's baseline and task patterns
                pattern_corr, _ = pearsonr(baseline_data[subj, :], task_data[subj, :])
                pattern_correlations.append(pattern_corr)
                
                # Euclidean distance between patterns
                pattern_dist = np.linalg.norm(task_data[subj, :] - baseline_data[subj, :])
                pattern_distances.append(pattern_dist)
            
            measures_dict[f'{condition}_rank_correlation'] = np.array(rank_correlations)
            if return_extra:
                measures_dict[f'{condition}_pattern_correlation'] = np.array(pattern_correlations)
                measures_dict[f'{condition}_pattern_distance'] = np.array(pattern_distances)
        
        return measures_dict
    
    def calculate_individual_extremeness_changes(self, return_extra=False):
        """
        For each subject and condition: How much does their distance from the group center change?
        
        Returns:
        --------
        measures_dict : dict
            Keys: '{condition}_extremeness_change'
            Values: arrays of shape (n_subjects,)
        """
        baseline_data = self.data_dict[self.baseline]
        baseline_group_mean = np.mean(baseline_data, axis=0)
        
        measures_dict = {}
        
        print("  Calculating extremeness changes...")
        for condition in self.task_conditions:
            task_data = self.data_dict[condition]
            task_group_mean = np.mean(task_data, axis=0)
            
            extremeness_changes = []
            
            for subj in range(self.n_subjects):
                # Distance from group mean in baseline vs task
                baseline_distance = np.linalg.norm(baseline_data[subj, :] - baseline_group_mean)
                task_distance = np.linalg.norm(task_data[subj, :] - task_group_mean)
                
                extremeness_changes.append(task_distance - baseline_distance)
            
            measures_dict[f'{condition}_extremeness_change'] = np.array(extremeness_changes)
        
        return measures_dict
    
    def create_subject_measures_dataframe(self):
        """
        Create comprehensive DataFrame with all subject-level measures for behavioral correlations
        
        Returns:
        --------
        df_measures : DataFrame
            Rows = subjects, Columns = all subject-level measures across conditions
        """
        print(f"Calculating subject-level measures for {len(self.task_conditions)} conditions...")
        
        # Initialize with actual subject IDs from behavioral data
        measures_dict = {'Subject': self.subject_ids}
        
        # Calculate all measure types
        measures_dict.update(self.calculate_individual_sa_differential_speedup(return_extra=True))
        # measures_dict.update(self.calculate_individual_mean_level_changes(return_extra=False))
        # measures_dict.update(self.calculate_individual_sa_coupling_changes(return_extra=False))
        measures_dict.update(self.calculate_individual_spatial_variability_changes(return_extra=False))
        measures_dict.update(self.calculate_individual_pattern_stability(return_extra=True))
        measures_dict.update(self.calculate_individual_sa_corr_differences(return_extra=False))
        # measures_dict.update(self.calculate_individual_rank_changes(return_extra=False))
        # measures_dict.update(self.calculate_individual_extremeness_changes(return_extra=False))
        
        # Create DataFrame
        df_measures = pd.DataFrame(measures_dict)
        df_measures.set_index('Subject', inplace=True)
        
        print(f"Created subject measures DataFrame: {df_measures.shape}")
        print(f"Measures per condition: {df_measures.shape[1] // len(self.task_conditions)}")
        
        return df_measures
    
    def correlate_with_behavior(self, df_measures, behavioral_columns=None, 
                               neural_columns=None, method='pearson', min_valid_pairs=10):
        """
        Correlate subject-level neural measures with specified behavioral variables
        
        Parameters:
        -----------
        df_measures : DataFrame
            Subject-level neural measures (from create_subject_measures_dataframe)
        behavioral_columns : list, optional
            Specific behavioral columns to correlate. If None, uses all numeric columns.
        neural_columns : list, optional
            Specific neural columns to correlate. If None, uses all columns.
        method : str
            Correlation method ('pearson' or 'spearman')
        min_valid_pairs : int
            Minimum number of valid (non-NaN) pairs required for correlation
            
        Returns:
        --------
        correlation_results : DataFrame
            Correlation matrix (neural_vars × behavioral_vars)
        pvalue_results : DataFrame
            P-values for all correlations
        significant_results : DataFrame
            Only significant correlations (p < 0.05)
        """
        # Set up behavioral data with Subject as index for alignment
        df_behavior_indexed = self.df_behavior.set_index('Subject')
        
        # Both DataFrames now have matching Subject IDs as index
        df_measures_aligned = df_measures
        df_behavior_aligned = df_behavior_indexed
        
        print(f"Correlating neural measures with behavior for {len(df_measures_aligned)} subjects...")
        
        # Select behavioral columns
        if behavioral_columns is None:
            behavioral_columns = df_behavior_aligned.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Validate that specified columns exist
            available_cols = df_behavior_aligned.columns.tolist()
            behavioral_columns = [col for col in behavioral_columns if col in available_cols]
            if len(behavioral_columns) == 0:
                raise ValueError("None of the specified behavioral columns found in data")
        
        # Select neural columns
        if neural_columns is None:
            neural_columns = df_measures_aligned.columns.tolist()
        
        print(f"  Neural measures: {len(neural_columns)}")
        print(f"  Behavioral measures: {len(behavioral_columns)}")
        
        # Choose correlation function
        corr_func = pearsonr if method == 'pearson' else spearmanr
        
        # Calculate correlations
        correlation_matrix = np.zeros((len(neural_columns), len(behavioral_columns)))
        pvalue_matrix = np.zeros((len(neural_columns), len(behavioral_columns)))
        
        for i, neural_var in enumerate(tqdm(neural_columns, desc="Correlating")):
            for j, behav_var in enumerate(behavioral_columns):
                # Get data and remove NaN values
                neural_data = df_measures_aligned[neural_var].values
                behav_data = df_behavior_aligned[behav_var].values
                
                valid_mask = ~(np.isnan(neural_data) | np.isnan(behav_data))
                
                if np.sum(valid_mask) >= min_valid_pairs:
                    try:
                        corr, pval = corr_func(neural_data[valid_mask], behav_data[valid_mask])
                        correlation_matrix[i, j] = corr
                        pvalue_matrix[i, j] = pval
                    except:
                        correlation_matrix[i, j] = np.nan
                        pvalue_matrix[i, j] = np.nan
                else:
                    correlation_matrix[i, j] = np.nan
                    pvalue_matrix[i, j] = np.nan
        
        # Create results DataFrames
        correlation_results = pd.DataFrame(
            correlation_matrix, 
            index=neural_columns, 
            columns=behavioral_columns
        )
        
        pvalue_results = pd.DataFrame(
            pvalue_matrix,
            index=neural_columns,
            columns=behavioral_columns
        )
        
        # Significant results (p < 0.05)
        significant_mask = pvalue_results < 0.05
        significant_results = correlation_results.where(significant_mask)
        
        n_significant = significant_mask.sum().sum()
        print(f"  Significant correlations found: {n_significant}")
        
        return correlation_results, pvalue_results, significant_results
    
    def plot_behavioral_correlations(self, correlation_results, significant_results, 
                                   title="Neural INTs - Behavioral Correlations", 
                                   figsize=(16, 10), save_path=None):
        """
        Plot heatmap of neural-behavioral correlations
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # All correlations
        sns.heatmap(correlation_results, cmap='RdBu_r', center=0, 
                   annot=True, fmt='.2f', ax=ax1, 
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax1.set_title(f'{title}\n(All Correlations)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Behavioral Measures', fontsize=10)
        ax1.set_ylabel('Neural INTs Measures', fontsize=10)
        
        # Significant correlations only
        sns.heatmap(significant_results, cmap='RdBu_r', center=0,
                   annot=True, fmt='.2f', ax=ax2, 
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax2.set_title(f'{title}\n(Significant Only, p < 0.05)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Behavioral Measures', fontsize=10)
        ax2.set_ylabel('Neural INTs Measures', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def get_top_correlations(self, correlation_results, pvalue_results, n_top=10, 
                           p_threshold=0.05, abs_corr_threshold=0.1):
        """
        Get the strongest significant correlations
        
        Parameters:
        -----------
        correlation_results : DataFrame
        pvalue_results : DataFrame
        n_top : int
            Number of top correlations to return
        p_threshold : float
            P-value threshold for significance
        abs_corr_threshold : float
            Minimum absolute correlation threshold
            
        Returns:
        --------
        top_correlations : DataFrame
            Top correlations with neural_var, behavioral_var, correlation, p_value columns
        """
        significant_correlations = []
        
        for neural_var in correlation_results.index:
            for behav_var in correlation_results.columns:
                corr_val = correlation_results.loc[neural_var, behav_var]
                p_val = pvalue_results.loc[neural_var, behav_var]
                
                if (not np.isnan(corr_val) and 
                    p_val < p_threshold and 
                    abs(corr_val) >= abs_corr_threshold):
                    significant_correlations.append({
                        'neural_measure': neural_var,
                        'behavioral_measure': behav_var,
                        'correlation': corr_val,
                        'p_value': p_val,
                        'abs_correlation': abs(corr_val)
                    })
        
        # Convert to DataFrame and sort
        df_corr = pd.DataFrame(significant_correlations)
        
        if len(df_corr) > 0:
            df_corr = df_corr.sort_values('abs_correlation', ascending=False)
            return df_corr.head(n_top)
        else:
            print("No significant correlations found meeting the criteria")
            return pd.DataFrame()


def run_ints_behavioral_analysis(data_dict, sa_axis, df_behavior, 
                                 behavioral_columns=None, baseline_condition=None,
                                 correlation_method='pearson', plot_results=True):
    """
    Complete pipeline for INTs subject-level measures and behavioral correlations
    
    Parameters:
    -----------
    data_dict : dict
        {condition_name: numpy_array(n_subjects, n_regions)}
    sa_axis : array
        SA axis values for regions
    df_behavior : DataFrame
        Behavioral data with subject information
    behavioral_columns : list, optional
        Specific behavioral columns to correlate with
    baseline_condition : str, optional
        Baseline condition name
    correlation_method : str
        'pearson' or 'spearman'
    plot_results : bool
        Whether to plot correlation results
        
    Returns:
    --------
    analyzer : SubjectLevelINTsMeasures
    df_measures : DataFrame
        Subject-level neural measures
    correlation_results : DataFrame
        Neural-behavioral correlations
    top_correlations : DataFrame
        Top significant correlations
    """
    print("Starting INTs Subject-Level Measures & Behavioral Analysis...")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SubjectLevelINTsMeasures(data_dict, sa_axis, df_behavior, baseline_condition)
    
    # Calculate all subject-level measures
    df_measures = analyzer.create_subject_measures_dataframe()
    
    print(f"\nSubject-level measures calculated:")
    print(f"  Shape: {df_measures.shape}")
    print(f"  Sample columns: {list(df_measures.columns[:5])}")
    
    # Run behavioral correlations
    print(f"\nRunning behavioral correlations using {correlation_method} method...")
    correlation_results, pvalue_results, significant_results = analyzer.correlate_with_behavior(
        df_measures, behavioral_columns=behavioral_columns, method=correlation_method
    )
    
    # Get top correlations
    top_correlations = analyzer.get_top_correlations(
        correlation_results, pvalue_results, n_top=15
    )
    
    if len(top_correlations) > 0:
        print(f"\nTop {len(top_correlations)} significant correlations:")
        for i, row in top_correlations.iterrows():
            print(f"  {i+1}. {row['neural_measure']} × {row['behavioral_measure']}: "
                  f"r = {row['correlation']:.3f}, p = {row['p_value']:.4f}")
    
    # Plot results
    if plot_results and len(significant_results.dropna(how='all', axis=1)) > 0:
        fig = analyzer.plot_behavioral_correlations(
            correlation_results, significant_results,
            title="Subject-level INTs measures - HCP behavioral correlations"
        )
    
    return analyzer, df_measures, correlation_results, significant_results, top_correlations, fig


def compute_autocorrelation_function(data):
    """
    Compute autocorrelation function for time series data.
    
    Parameters:
    - data: 2D array (voxels x timepoints) or 1D array (single time series)
    
    Returns:
    - autocorr: autocorrelation function(s)
    - zero_crossings: first zero-crossing indices
    """
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    autocorr_list = []
    zero_crossings = []
    
    for voxel_ts in data:
        if np.std(voxel_ts) > 0:  # Avoid division by zero
            # Standardize the time series
            standardized = (voxel_ts - np.mean(voxel_ts)) / np.std(voxel_ts)
            
            # Compute full autocorrelation
            acf = np.correlate(standardized, standardized, mode='full')
            acf = acf[len(acf)//2:]  # Take only non-negative lags
            acf = acf / acf[0]  # Normalize by lag-0 value
            
            autocorr_list.append(acf)
            
            # Find first zero-crossing
            negative_indices = np.where(acf < 0)[0]
            if len(negative_indices) > 0:
                zero_crossings.append(negative_indices[0])
            else:
                zero_crossings.append(len(acf) - 1)  # No zero crossing found
        else:
            # Handle constant time series
            n_timepoints = len(voxel_ts)
            autocorr_list.append(np.ones(n_timepoints))
            zero_crossings.append(n_timepoints - 1)
    
    return np.array(autocorr_list), np.array(zero_crossings)


def compute_timescales(timeseries, which_version='acf', return_mean=True, tr=np.nan):
    n_parcels = timeseries.shape[1]
    n_subs = timeseries.shape[-1]
    print(n_subs, n_parcels)
    timescales = np.zeros((n_parcels, n_subs))
    for i in tqdm(np.arange(n_subs)):
        if which_version == 'acf':
            try:
                autocorr_raw, timescales[:, i] = compute_autocorrelation_function(timeseries[:, :, i].T)
            except:
                timescales[:, i] = np.nan
        elif which_version == 'alff':
            try:
                timescales[:, i] = compute_alff(timeseries[:, :, i].T, low_pass=0.08, high_pass=0.01, TR=tr, sample_mask=None)
            except:
                timescales[:, i] = np.nan
        else:
            for j in np.arange(n_parcels):
                try:
                    timescales[j, i] = compute_rlfp(timeseries[:, j, i], tr=tr, low=0.01, high=0.08)
                    # time_series = signal.detrend(timeseries[:, j, i])
                    # if which_version == 'acf':
                    #     _, timescales[j, i] = compute_acf(timeseries[:, j, i], nlags=20)
                    # elif which_version == 'rlfp':
                    #     timescales[j, i] = compute_rlfp(timeseries[:, j, i], tr=tr, low=0.01, high=0.08)
                except:
                    timescales[j, i] = np.nan
    if return_mean:
        return np.nanmean(timescales, axis=1)
    else:
        return timescales


def get_task_events(ev_dir):
    task_events = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])

    conditions = os.listdir(ev_dir)
    for condition in conditions:
        task_events_tmp = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])

        try:
            ev_data = np.genfromtxt(os.path.join(ev_dir, condition))

            if ev_data.ndim == 1:
                task_events_tmp.loc[0, 'onset'] = ev_data[0]
                task_events_tmp.loc[0, 'duration'] = ev_data[1]
                task_events_tmp.loc[0, 'trial_type'] = condition.replace('.txt', '')

            elif ev_data.ndim == 2:
                for block in np.arange(ev_data.shape[0]):
                    task_events_tmp.loc[block, 'onset'] = ev_data[block, 0]
                    task_events_tmp.loc[block, 'duration'] = ev_data[block, 1]
                    task_events_tmp.loc[block, 'trial_type'] = condition.replace('.txt', '')
        except:
            pass
        
        task_events = pd.concat((task_events, task_events_tmp), axis=0)

    task_events.reset_index(inplace=True, drop=True)    
    
    return task_events

    
def create_design_matrix_nilearn(events, n_timepoints, tr, hrf_model='glover', drift_model='cosine', 
                                 high_pass=0.01, add_regs=None):
    """
    Create design matrix using nilearn's implementation.
    
    Parameters:
    - events: pandas DataFrame with events (onset, duration, trial_type)
    - n_timepoints: number of timepoints
    - tr: repetition time in seconds
    - hrf_model: HRF model ('glover', 'spm', 'glover + derivative', etc.)
    - drift_model: drift model ('polynomial', 'cosine', None)
    - high_pass: high-pass filter cutoff in Hz
    - add_regs: additional regressors (optional)
    
    Returns:
    - design_matrix: nilearn DesignMatrix object
    """
    
    # Create frame times
    frame_times = np.arange(n_timepoints) * tr
    
    # Use nilearn to create design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        add_regs=add_regs
    )
    
    return design_matrix


def fit_glm(fmri_data, design_matrix):
    """
    Fit GLM.
    
    Parameters:
    - fmri_data: 2D array (voxels x timepoints)
    - design_matrix: nilearn DesignMatrix object
    
    Returns:
    - residuals: residual time series after GLM fitting
    - predicted: predicted time series
    - betas: regression coefficients
    """
    
    try:
        # Convert design matrix to numpy array
        X = design_matrix.values
        
        # Fit GLM using least squares
        betas, residuals_sum, rank, s = np.linalg.lstsq(X, fmri_data.T, rcond=None)
        
        # Compute predicted and residual signals
        predicted = np.dot(X, betas).T
        residuals = fmri_data - predicted
        
        return residuals, predicted, betas.T
        
    except np.linalg.LinAlgError as e:
        # GLM fitting failed, use original data as residuals
        return fmri_data.copy(), np.zeros_like(fmri_data), np.zeros((fmri_data.shape[0], design_matrix.shape[1]))


def compute_autocorrelation_with_hrf(fmri_data, events, tr, mask=None, hrf_model='glover', drift_model='cosine'):
    """
    Compute autocorrelation of fMRI data using nilearn for block design analysis.
    
    Parameters:
    - fmri_data: 4D numpy array (x,y,z,time) or 2D array (voxels,time)
    - events: pandas DataFrame with events (onset, duration, trial_type) or path to CSV
    - tr: repetition time in seconds
    - mask: binary mask for voxel selection (optional)
    - hrf_model: HRF model for nilearn ('glover', 'spm', 'glover + derivative', etc.)
    - drift_model: drift model for nilearn ('cosine', 'polynomial', None)
    
    Returns:
    - autocorr_raw: autocorrelation before task removal (n_voxels x n_lags)
    - autocorr_resid: autocorrelation after task removal (n_voxels x n_lags)  
    - zero_crossings: first zero-crossing indices (n_voxels,)
    - residuals: residual time series after GLM (n_voxels x n_timepoints)
    - design_matrix: nilearn DesignMatrix object
    """

    # Handle input data shape
    if fmri_data.ndim == 4:
        original_shape = fmri_data.shape
        n_voxels = np.prod(original_shape[:3])
        n_timepoints = original_shape[3]
        fmri_data_2d = fmri_data.reshape((n_voxels, n_timepoints))
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)
            fmri_data_2d = fmri_data_2d[mask_flat > 0, :]
    elif fmri_data.ndim == 2:
        fmri_data_2d = fmri_data.copy()
        n_timepoints = fmri_data_2d.shape[1]
    else:
        raise ValueError("fMRI data must be 2D (voxels x time) or 4D (x,y,z,time)")

    # Compute raw autocorrelation (before task removal)
    autocorr_raw, zero_crossings_raw = compute_autocorrelation_function(fmri_data_2d)

    # Create design matrix using nilearn
    design_matrix = create_design_matrix_nilearn(events, n_timepoints, tr, hrf_model=hrf_model, drift_model=drift_model)
    
    # Fit GLM using nilearn approach
    residuals, predicted, betas = fit_glm(fmri_data_2d, design_matrix)
    
    # Compute autocorrelation on residuals
    autocorr_resid, zero_crossings_resid = compute_autocorrelation_function(residuals)

    return autocorr_raw, zero_crossings_raw, autocorr_resid, zero_crossings_resid, residuals, design_matrix


