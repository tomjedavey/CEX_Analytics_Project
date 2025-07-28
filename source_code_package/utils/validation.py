#!/usr/bin/env python3
"""
Validation functionality for AS_1 multi-dataset configuration and processing.

This module provides comprehensive validation functions for:
1. Configuration file validation
2. Dataset validation (raw and processed)
3. Data consistency checks
4. Directory structure validation
5. Pipeline validation

Author: Tom Davey
Date: July 2025
"""

import os
import pandas as pd
import yaml
from typing import Dict, List, Tuple
import json


class AS1ValidationSuite:
    """
    Comprehensive validation suite for AS_1 multi-dataset pipeline.
    """
    
    def __init__(self, project_root: str = None):
        """
        Initialize the validation suite.
        
        Parameters:
        -----------
        project_root : str, optional
            Root directory of the project. If None, will try to detect automatically.
        """
        if project_root is None:
            # Try to detect project root
            current_dir = os.path.abspath(os.path.dirname(__file__))
            while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    raise FileNotFoundError('Could not find project root (pyproject.toml)')
                current_dir = parent_dir
            project_root = current_dir
        
        self.project_root = project_root
        self.validation_results = {}
    
    def validate_configurations(self, config_names: List[str] = None) -> Dict[str, bool]:
        """
        Validate that all required configuration files exist and are valid.
        
        Parameters:
        -----------
        config_names : List[str], optional
            List of configuration file names to validate. If None, uses default set.
            
        Returns:
        --------
        Dict[str, bool]
            Validation results for each configuration
        """
        print("VALIDATING CONFIGURATION FILES")
        print("-" * 40)
        
        if config_names is None:
            config_names = [
                'config_AS_1.yaml',
                'config_AS_1_full_dataset.yaml',
                'config_AS_1_cluster_0.yaml',
                'config_AS_1_cluster_1.yaml'
            ]
        
        config_dir = os.path.join(self.project_root, 'source_code_package', 'config')
        results = {}
        
        for config_file in config_names:
            config_path = os.path.join(config_dir, config_file)
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check required sections
                    required_sections = ['data', 'features', 'train_test_split', 'output']
                    has_all_sections = all(section in config for section in required_sections)
                    
                    if has_all_sections:
                        print(f"✅ {config_file}: Valid")
                        results[config_file] = True
                    else:
                        missing_sections = [s for s in required_sections if s not in config]
                        print(f"❌ {config_file}: Missing sections: {missing_sections}")
                        results[config_file] = False
                        
                except yaml.YAMLError as e:
                    print(f"❌ {config_file}: YAML parsing error - {e}")
                    results[config_file] = False
            else:
                print(f"❌ {config_file}: File not found")
                results[config_file] = False
        
        self.validation_results['configurations'] = results
        return results
    
    def validate_raw_datasets(self) -> Dict[str, Dict]:
        """
        Validate that all raw datasets exist and have correct structure.
        
        Returns:
        --------
        Dict[str, Dict]
            Validation results for each raw dataset
        """
        print("\nVALIDATING RAW DATASETS")
        print("-" * 40)
        
        datasets = {
            'full_dataset': os.path.join(self.project_root, 'data/raw_data/new_raw_data_polygon.csv'),
            'cluster_0': os.path.join(self.project_root, 'data/raw_data/cluster_datasets/new_raw_data_polygon_cluster_0.csv'),
            'cluster_1': os.path.join(self.project_root, 'data/raw_data/cluster_datasets/new_raw_data_polygon_cluster_1.csv')
        }
        
        results = {}
        
        for dataset_name, file_path in datasets.items():
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Check structure
                    expected_columns = 22  # Original dataset has 22 columns
                    actual_columns = len(df.columns)
                    
                    # Check for required columns
                    required_columns = ['WALLET', 'TX_PER_MONTH', 'TOTAL_TRANSFER_USD']
                    has_required_columns = all(col in df.columns for col in required_columns)
                    
                    results[dataset_name] = {
                        'exists': True,
                        'records': len(df),
                        'columns': actual_columns,
                        'structure_valid': actual_columns == expected_columns,
                        'has_required_columns': has_required_columns,
                        'missing_columns': [col for col in required_columns if col not in df.columns]
                    }
                    
                    status = "✅" if results[dataset_name]['structure_valid'] and has_required_columns else "⚠️"
                    print(f"{status} {dataset_name}: {len(df):,} records, {actual_columns} columns")
                    
                    if not has_required_columns:
                        print(f"    Missing columns: {results[dataset_name]['missing_columns']}")
                    
                except Exception as e:
                    print(f"❌ {dataset_name}: Error reading file - {e}")
                    results[dataset_name] = {'exists': True, 'error': str(e)}
            else:
                print(f"❌ {dataset_name}: File not found at {file_path}")
                results[dataset_name] = {'exists': False, 'file_path': file_path}
        
        self.validation_results['raw_datasets'] = results
        return results
    
    def validate_processed_datasets(self) -> Dict[str, Dict]:
        """
        Validate that all processed datasets exist and have correct engineered features.
        
        Returns:
        --------
        Dict[str, Dict]
            Validation results for each processed dataset
        """
        print("\nVALIDATING PROCESSED DATASETS")
        print("-" * 40)
        
        datasets = {
            'full_dataset': os.path.join(self.project_root, 'data/processed_data/AS_1_feature_data_full_dataset.csv'),
            'cluster_0': os.path.join(self.project_root, 'data/processed_data/AS_1_feature_data_cluster_0.csv'),
            'cluster_1': os.path.join(self.project_root, 'data/processed_data/AS_1_feature_data_cluster_1.csv')
        }
        
        results = {}
        
        # Expected engineered features
        expected_features = [
            'REVENUE_PROXY',  # Target variable
            'ESTIMATED_TOTAL_VOLUME',
            'TRADING_EVENTS_TOTAL',
            'PROTOCOL_EXPERTISE',
            'WALLET_MATURITY_SCORE',
            'IS_BRIDGE_USER',
            'LOG_AVG_TRANSFER'
        ]
        
        for dataset_name, file_path in datasets.items():
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Check for engineered features
                    has_target = 'REVENUE_PROXY' in df.columns
                    present_features = [f for f in expected_features if f in df.columns]
                    missing_features = [f for f in expected_features if f not in df.columns]
                    
                    results[dataset_name] = {
                        'exists': True,
                        'records': len(df),
                        'columns': len(df.columns),
                        'has_target_variable': has_target,
                        'present_features': present_features,
                        'missing_features': missing_features,
                        'feature_completeness': len(present_features) / len(expected_features)
                    }
                    
                    status = "✅" if has_target and len(missing_features) == 0 else "⚠️"
                    print(f"{status} {dataset_name}: {len(df):,} records, {len(df.columns)} columns")
                    print(f"    Features: {len(present_features)}/{len(expected_features)} present")
                    
                    if missing_features:
                        print(f"    Missing features: {missing_features}")
                    
                except Exception as e:
                    print(f"❌ {dataset_name}: Error reading file - {e}")
                    results[dataset_name] = {'exists': True, 'error': str(e)}
            else:
                print(f"❌ {dataset_name}: File not found at {file_path}")
                results[dataset_name] = {'exists': False, 'file_path': file_path}
        
        self.validation_results['processed_datasets'] = results
        return results
    
    def validate_data_consistency(self) -> Dict[str, bool]:
        """
        Validate data consistency between datasets and clustering results.
        
        Returns:
        --------
        Dict[str, bool]
            Consistency validation results
        """
        print("\nVALIDATING DATA CONSISTENCY")
        print("-" * 40)
        
        results = {}
        
        try:
            # Load clustering summary for expected counts
            summary_path = os.path.join(self.project_root, 'data/raw_data/cluster_datasets/cluster_datasets_summary.json')
            
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                expected_cluster_0 = summary['cluster_statistics']['cluster_0']['count']
                expected_cluster_1 = summary['cluster_statistics']['cluster_1']['count']
                expected_total = expected_cluster_0 + expected_cluster_1
                
                # Load datasets for consistency checks
                datasets_to_check = {
                    'full_raw': os.path.join(self.project_root, 'data/raw_data/new_raw_data_polygon.csv'),
                    'cluster_0_raw': os.path.join(self.project_root, 'data/raw_data/cluster_datasets/new_raw_data_polygon_cluster_0.csv'),
                    'cluster_1_raw': os.path.join(self.project_root, 'data/raw_data/cluster_datasets/new_raw_data_polygon_cluster_1.csv'),
                    'full_proc': os.path.join(self.project_root, 'data/processed_data/AS_1_feature_data_full_dataset.csv'),
                    'cluster_0_proc': os.path.join(self.project_root, 'data/processed_data/AS_1_feature_data_cluster_0.csv'),
                    'cluster_1_proc': os.path.join(self.project_root, 'data/processed_data/AS_1_feature_data_cluster_1.csv')
                }
                
                loaded_datasets = {}
                for name, path in datasets_to_check.items():
                    if os.path.exists(path):
                        loaded_datasets[name] = pd.read_csv(path)
                    else:
                        print(f"⚠️  Dataset not found: {name}")
                        loaded_datasets[name] = None
                
                # Perform consistency checks
                if all(df is not None for df in loaded_datasets.values()):
                    results['cluster_0_count_matches'] = len(loaded_datasets['cluster_0_raw']) == expected_cluster_0
                    results['cluster_1_count_matches'] = len(loaded_datasets['cluster_1_raw']) == expected_cluster_1
                    results['cluster_sum_matches_expected'] = (len(loaded_datasets['cluster_0_raw']) + len(loaded_datasets['cluster_1_raw'])) == expected_total
                    results['raw_processed_consistency_full'] = len(loaded_datasets['full_raw']) == len(loaded_datasets['full_proc'])
                    results['raw_processed_consistency_cluster_0'] = len(loaded_datasets['cluster_0_raw']) == len(loaded_datasets['cluster_0_proc'])
                    results['raw_processed_consistency_cluster_1'] = len(loaded_datasets['cluster_1_raw']) == len(loaded_datasets['cluster_1_proc'])
                    
                    # Print results
                    for check, passed in results.items():
                        status = "✅" if passed else "❌"
                        print(f"{status} {check.replace('_', ' ').title()}")
                    
                    print(f"\nRecord counts:")
                    for name, df in loaded_datasets.items():
                        if df is not None:
                            print(f"  {name}: {len(df):,} records")
                    
                else:
                    results['validation_error'] = True
                    print("❌ Some datasets missing - cannot validate consistency")
                    
            else:
                results['no_summary_file'] = True
                print("❌ Cluster summary file not found - cannot validate consistency")
                
        except Exception as e:
            print(f"❌ Error during consistency validation: {e}")
            results['validation_error'] = True
        
        self.validation_results['data_consistency'] = results
        return results
    
    def validate_directory_structure(self) -> Dict[str, bool]:
        """
        Validate that all required directories exist.
        
        Returns:
        --------
        Dict[str, bool]
            Directory validation results
        """
        print("\nVALIDATING DIRECTORY STRUCTURE")
        print("-" * 40)
        
        required_dirs = [
            'data/raw_data/cluster_datasets',
            'data/processed_data',
            'data/scores',
            'data/logs',
            'source_code_package/config',
            'scripts',
            'scripts/AS_1'
        ]
        
        results = {}
        
        for rel_dir_path in required_dirs:
            abs_dir_path = os.path.join(self.project_root, rel_dir_path)
            exists = os.path.exists(abs_dir_path) and os.path.isdir(abs_dir_path)
            results[rel_dir_path] = exists
            
            status = "✅" if exists else "❌"
            print(f"{status} {rel_dir_path}")
        
        self.validation_results['directory_structure'] = results
        return results
    
    def generate_validation_report(self) -> Dict:
        """
        Generate a comprehensive validation report.
        
        Returns:
        --------
        dict
            Complete validation report
        """
        print("\n" + "=" * 60)
        print("AS_1 PIPELINE VALIDATION REPORT")
        print("=" * 60)
        
        # Run all validations if not already done
        if 'configurations' not in self.validation_results:
            self.validate_configurations()
        if 'raw_datasets' not in self.validation_results:
            self.validate_raw_datasets()
        if 'processed_datasets' not in self.validation_results:
            self.validate_processed_datasets()
        if 'data_consistency' not in self.validation_results:
            self.validate_data_consistency()
        if 'directory_structure' not in self.validation_results:
            self.validate_directory_structure()
        
        # Analyze overall status
        config_results = self.validation_results.get('configurations', {})
        raw_results = self.validation_results.get('raw_datasets', {})
        processed_results = self.validation_results.get('processed_datasets', {})
        consistency_results = self.validation_results.get('data_consistency', {})
        directory_results = self.validation_results.get('directory_structure', {})
        
        all_configs_valid = all(config_results.values()) if config_results else False
        all_raw_valid = all(r.get('exists', False) and not r.get('error') for r in raw_results.values()) if raw_results else False
        all_processed_valid = all(r.get('exists', False) and not r.get('error') for r in processed_results.values()) if processed_results else False
        all_consistent = all(consistency_results.values()) if consistency_results and 'validation_error' not in consistency_results else False
        all_dirs_exist = all(directory_results.values()) if directory_results else False
        
        overall_success = all([all_configs_valid, all_raw_valid, all_processed_valid, all_consistent, all_dirs_exist])
        
        # Generate report
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'overall_success': overall_success,
            'component_status': {
                'configurations': 'PASS' if all_configs_valid else 'FAIL',
                'raw_datasets': 'PASS' if all_raw_valid else 'FAIL',
                'processed_datasets': 'PASS' if all_processed_valid else 'FAIL',
                'data_consistency': 'PASS' if all_consistent else 'FAIL',
                'directory_structure': 'PASS' if all_dirs_exist else 'FAIL'
            },
            'detailed_results': self.validation_results,
            'next_steps': self._generate_next_steps(overall_success, all_configs_valid, all_raw_valid, 
                                                   all_processed_valid, all_consistent, all_dirs_exist)
        }
        
        # Print summary
        status = "🎉 PIPELINE VALIDATION SUCCESSFUL!" if overall_success else "⚠️  PIPELINE HAS ISSUES"
        print(f"\n{status}")
        
        print(f"\nComponent Status:")
        for component, status in report['component_status'].items():
            icon = "✅" if status == 'PASS' else "❌"
            print(f"  {icon} {component.replace('_', ' ').title()}: {status}")
        
        if overall_success:
            print(f"\n📋 READY FOR MODEL TRAINING:")
            for step in report['next_steps']:
                print(f"  - {step}")
        else:
            print(f"\n🔧 ISSUES TO RESOLVE:")
            for step in report['next_steps']:
                print(f"  - {step}")
        
        return report
    
    def _generate_next_steps(self, overall_success, all_configs_valid, all_raw_valid, 
                           all_processed_valid, all_consistent, all_dirs_exist) -> List[str]:
        """Generate next steps based on validation results."""
        if overall_success:
            return [
                "AS_1 model training for each dataset",
                "Performance comparison across clusters",
                "Feature importance analysis",
                "Business insights generation"
            ]
        else:
            steps = []
            if not all_configs_valid:
                steps.append("Fix configuration file issues")
            if not all_raw_valid:
                steps.append("Ensure all raw datasets are created properly")
            if not all_processed_valid:
                steps.append("Re-run feature engineering for missing processed datasets")
            if not all_consistent:
                steps.append("Check data consistency between raw and processed datasets")
            if not all_dirs_exist:
                steps.append("Create missing directories")
            return steps


# Convenience functions
def validate_full_pipeline(project_root: str = None) -> Dict:
    """
    Convenience function to run full pipeline validation.
    
    Parameters:
    -----------
    project_root : str, optional
        Root directory of the project
        
    Returns:
    --------
    dict
        Complete validation report
    """
    validator = AS1ValidationSuite(project_root)
    return validator.generate_validation_report()


def quick_validation_check(project_root: str = None) -> bool:
    """
    Quick validation check that returns True if pipeline is ready.
    
    Parameters:
    -----------
    project_root : str, optional
        Root directory of the project
        
    Returns:
    --------
    bool
        True if pipeline is ready for model training
    """
    validator = AS1ValidationSuite(project_root)
    report = validator.generate_validation_report()
    return report['overall_success']
