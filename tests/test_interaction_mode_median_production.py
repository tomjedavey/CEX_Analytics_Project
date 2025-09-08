"""
Unit tests for interaction_mode_median_production_source.py
Covers config loading, cluster selection, scoring, and median calculation logic.
"""
import os
import tempfile
import shutil
import yaml
import numpy as np
import pandas as pd
import pytest

from source_code_package.features import interaction_mode_median_production_source as immp

def test_load_config_and_target_features():
    # Test default config loading and target features
    config = immp.load_config()
    features = immp.get_target_features(config)
    assert isinstance(config, dict)
    assert set(features) == {"DEX_EVENTS", "CEX_EVENTS", "DEFI_EVENTS", "BRIDGE_EVENTS"}

def test_calculate_enhanced_selection_score():
    # Test scoring above and below activity threshold
    score = immp.calculate_enhanced_selection_score(100, 1.0, 10.0, 0.5, min_activity_threshold=0.3)
    assert score == 5.0
    score_zero = immp.calculate_enhanced_selection_score(100, 1.0, 10.0, 0.2, min_activity_threshold=0.3)
    assert score_zero == 0.0

def test_select_strongest_cluster_for_feature():
    # Create synthetic DataFrame with two clusters and a feature
    df = pd.DataFrame({
        'cluster': [0]*60 + [1]*60 + [-1]*5,
        'DEX_EVENTS': [5]*30 + [0]*30 + [10]*60 + [0]*5
    })
    # Cluster 0: 30/60 nonzero, median_nonzero=5
    # Cluster 1: 60/60 nonzero, median_nonzero=10
    cluster_id, stats = immp.select_strongest_cluster_for_feature(
        df, 'DEX_EVENTS', min_activity_threshold=0.3, min_cluster_size=50)
    assert cluster_id == 1
    assert stats['median_nonzero_value'] == 10.0
    # Test fallback: all clusters below threshold (all feature values zero)
    df2 = pd.DataFrame({
        'cluster': [0]*60,
        'DEX_EVENTS': [0]*60
    })
    fallback_cluster_id, fallback_stats = immp.select_strongest_cluster_for_feature(
        df2, 'DEX_EVENTS', min_activity_threshold=0.3, min_cluster_size=50)
    # Should return the only cluster, with zero activity and zero median
    assert fallback_cluster_id == 0
    assert fallback_stats['non_zero_proportion'] == 0.0
    assert fallback_stats['median_nonzero_value'] == 0.0

def test_calculate_median_feature_values_for_clusters(tmp_path):
    # Setup synthetic results dir and files
    results_dir = tmp_path / "interaction_mode_results"
    os.makedirs(results_dir / "main_clustering")
    # Create base data and cluster labels
    base_data = pd.DataFrame({
        'DEX_EVENTS': [1,2,3,4,5,6,7,8,9,10],
        'CEX_EVENTS': [0,0,0,0,0,0,0,0,0,0],
        'DEFI_EVENTS': [1]*10,
        'BRIDGE_EVENTS': [0]*10
    })
    raw_data_dir = results_dir.parent / "raw_data"
    os.makedirs(raw_data_dir, exist_ok=True)
    base_data_path = raw_data_dir / "new_raw_data_polygon.csv"
    base_data.to_csv(base_data_path, index=False)
    clusters = pd.DataFrame({'cluster_label': [0,0,1,1,1,1,1,1,1,1]})
    clusters.to_csv(results_dir / "main_clustering/cluster_labels.csv", index=False)
    # Write a config file with the correct base data path
    config_path = tmp_path / "test_config.yaml"
    config = {
        'main_base_data_path': str(base_data_path)
    }
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    # Patch config loader to use our test config
    orig_load_config = immp.load_config
    immp.load_config = lambda: config
    try:
        results = immp.calculate_median_feature_values_for_clusters(
            results_dir=str(results_dir),
            min_activity_threshold=0.1,
            min_cluster_size=2,
            output_path=str(results_dir / "output.yaml")
        )
        assert 'datasets' in results
        assert 'main' in results['datasets']
        assert 'DEX_EVENTS' in results['datasets']['main']['feature_selections']
    finally:
        immp.load_config = orig_load_config

def test_generate_summary_statistics():
    # Test summary stats with mock results
    results = {
        'datasets': {
            'main': {
                'feature_selections': {
                    'DEX_EVENTS': {'median_nonzero_value': 5, 'feature_stats': {'non_zero_proportion': 0.5}, 'meets_activity_threshold': True},
                    'CEX_EVENTS': {'median_nonzero_value': 0, 'feature_stats': {'non_zero_proportion': 0.0}, 'meets_activity_threshold': False}
                }
            }
        }
    }
    summary = immp.generate_summary_statistics(results, ["DEX_EVENTS", "CEX_EVENTS"])
    assert summary['features_analyzed'] == 2
    assert summary['successful_selections'] == 2
    assert summary['activity_threshold_met'] >= 1

def test_save_and_print_cluster_selection_results(tmp_path):
    # Test saving and printing results
    results = {
        'selection_parameters': {'algorithm': 'test', 'min_activity_threshold': 0.1, 'min_cluster_size': 2},
        'datasets': {},
        'summary': {'total_datasets_processed': 1, 'successful_selections': 1, 'activity_threshold_met': 1, 'fallback_selections': 0}
    }
    output_path = tmp_path / "results.yaml"
    immp.save_cluster_selection_results(results, str(output_path))
    assert os.path.exists(output_path)
    immp.print_cluster_selection_summary(results)

def main():
    test_load_config_and_target_features()
    test_calculate_enhanced_selection_score()
    test_select_strongest_cluster_for_feature()
    test_generate_summary_statistics()
    print("All unit tests for interaction_mode_median_production_source passed.")

if __name__ == "__main__":
    main()
