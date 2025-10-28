#!/usr/bin/env python3
"""
Enhanced feature engineering functionality for AS_1 analysis supporting multiple datasets.

This module provides functionality to:
1. Process single datasets with feature engineering
2. Process multiple datasets in batch
3. Load and validate configurations
4. Handle various dataset types (full, cluster-specific)

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from typing import Optional, List, Dict, Tuple

# Import the base feature engineering functionality
try:
    from ..features.AS_1_features_source import engineer_features
except ImportError:
    # Handle relative import issues
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    features_path = os.path.join(project_root, 'source_code_package', 'features', 'AS_1_features_source.py')
    
    if os.path.exists(features_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("AS_1_features_source", features_path)
        features_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(features_module)
        engineer_features = features_module.engineer_features
    else:
        raise ImportError("Could not import engineer_features function")


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering class that supports multiple datasets and configurations.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the EnhancedFeatureEngineer.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress information
        """
        self.verbose = verbose
        self.processed_datasets = {}
        
    def load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
            
        Returns:
        --------
        dict
            Loaded configuration
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate that configuration has required sections.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
            
        Returns:
        --------
        tuple
            (is_valid, list_of_errors)
        """
        required_sections = ['data', 'features']
        errors = []
        
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Check data section
        if 'data' in config:
            required_data_keys = ['raw_data_path', 'processed_data_path']
            for key in required_data_keys:
                if key not in config['data']:
                    errors.append(f"Missing required data key: {key}")
        
        # Check features section
        if 'features' in config:
            required_feature_keys = ['dependent_variable']
            for key in required_feature_keys:
                if key not in config['features']:
                    errors.append(f"Missing required features key: {key}")
        
        return len(errors) == 0, errors
    
    def resolve_paths(self, config_path: str, config: Dict) -> Tuple[str, str]:
        """
        Resolve relative paths to absolute paths.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
        config : dict
            Configuration dictionary
            
        Returns:
        --------
        tuple
            (raw_data_path, processed_data_path)
        """
        raw_data_path = config['data']['raw_data_path']
        processed_data_path = config['data']['processed_data_path']
        
        # Convert relative paths to absolute
        if not os.path.isabs(raw_data_path):
            config_dir = os.path.dirname(config_path)
            raw_data_path = os.path.join(config_dir, '../../', raw_data_path)
            raw_data_path = os.path.normpath(raw_data_path)
        
        if not os.path.isabs(processed_data_path):
            config_dir = os.path.dirname(config_path)
            processed_data_path = os.path.join(config_dir, '../../', processed_data_path)
            processed_data_path = os.path.normpath(processed_data_path)
        
        return raw_data_path, processed_data_path
    
    def process_dataset(self, config_path: str) -> Dict:
        """
        Process a single dataset with feature engineering.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
            
        Returns:
        --------
        dict
            Processing results including success status and metadata
        """
        try:
            # Load and validate configuration
            config = self.load_config(config_path)
            is_valid, errors = self.validate_config(config)
            
            if not is_valid:
                return {
                    'success': False,
                    'error': f"Configuration validation failed: {'; '.join(errors)}",
                    'config_path': config_path
                }
            
            # Get dataset information
            dataset_type = config.get('metadata', {}).get('dataset_type', 'unknown')
            description = config.get('metadata', {}).get('description', 'No description')
            
            if self.verbose:
                print(f"\nProcessing dataset: {dataset_type}")
                print(f"Description: {description}")
            
            # Resolve file paths
            raw_data_path, processed_data_path = self.resolve_paths(config_path, config)
            
            # Check if input file exists
            if not os.path.exists(raw_data_path):
                return {
                    'success': False,
                    'error': f"Input file not found: {raw_data_path}",
                    'config_path': config_path,
                    'dataset_type': dataset_type
                }
            
            # Load and process data
            if self.verbose:
                print(f"Loading data from: {raw_data_path}")
            
            data = pd.read_csv(raw_data_path)
            original_shape = data.shape
            
            if self.verbose:
                print(f"Original data shape: {original_shape}")
            
            # Apply feature engineering
            if self.verbose:
                print("Applying feature engineering...")
            
            processed_data = engineer_features(data)
            final_shape = processed_data.shape
            
            if self.verbose:
                print(f"Processed data shape: {final_shape}")
                print(f"New features added: {final_shape[1] - original_shape[1]}")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(processed_data_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save processed data
            if self.verbose:
                print(f"Saving processed data to: {processed_data_path}")
            
            processed_data.to_csv(processed_data_path, index=False)
            
            # Store results
            result = {
                'success': True,
                'config_path': config_path,
                'dataset_type': dataset_type,
                'description': description,
                'raw_data_path': raw_data_path,
                'processed_data_path': processed_data_path,
                'original_shape': original_shape,
                'final_shape': final_shape,
                'features_added': final_shape[1] - original_shape[1],
                'processed_data': processed_data  # Store in memory for further use
            }
            
            self.processed_datasets[dataset_type] = result
            
            if self.verbose:
                print(f"✅ Successfully processed {dataset_type} dataset")
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'config_path': config_path,
                'dataset_type': config.get('metadata', {}).get('dataset_type', 'unknown') if 'config' in locals() else 'unknown'
            }
    
    def process_multiple_datasets(self, config_paths: List[str]) -> Dict[str, Dict]:
        """
        Process multiple datasets in batch.
        
        Parameters:
        -----------
        config_paths : List[str]
            List of configuration file paths
            
        Returns:
        --------
        Dict[str, Dict]
            Results for each dataset
        """
        results = {}
        
        if self.verbose:
            print("BATCH FEATURE ENGINEERING")
            print("=" * 50)
        
        for config_path in config_paths:
            config_name = os.path.basename(config_path)
            
            if self.verbose:
                print(f"\n--- Processing {config_name} ---")
            
            result = self.process_dataset(config_path)
            results[config_name] = result
            
            if not result['success'] and self.verbose:
                print(f"❌ Failed to process {config_name}: {result.get('error', 'Unknown error')}")
        
        # Generate summary
        if self.verbose:
            self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: Dict[str, Dict]):
        """
        Print a formatted summary of batch processing results.
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            Dictionary containing processing results for each dataset,
            where each result dict contains a 'success' boolean key
        
        Returns:
        --------
        None
            Prints summary information to console
        
        Notes:
        ------
        - Displays total success/failure counts
        - Lists failed datasets if any
        - Used internally for batch processing feedback
        """
        print("\n" + "=" * 50)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 50)
        
        successful = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        print(f"Processed: {successful}/{total} datasets successfully")
        
        if successful < total:
            print("\nFailed datasets:")
            for config_name, result in results.items():
                if not result['success']:
                    print(f"  - {config_name}: {result.get('error', 'Unknown error')}")
        
        if successful > 0:
            print("\nSuccessful datasets:")
            for config_name, result in results.items():
                if result['success']:
                    dataset_type = result['dataset_type']
                    features_added = result['features_added']
                    records = result['final_shape'][0]
                    print(f"  - {config_name} ({dataset_type}): {records:,} records, +{features_added} features")
    
    def get_processed_data(self, dataset_type: str) -> Optional[pd.DataFrame]:
        """
        Get processed data for a specific dataset type.
        
        Parameters:
        -----------
        dataset_type : str
            Type of dataset to retrieve
            
        Returns:
        --------
        pd.DataFrame or None
            Processed data if available
        """
        if dataset_type in self.processed_datasets:
            return self.processed_datasets[dataset_type].get('processed_data')
        return None
    
    def get_processing_summary(self) -> Dict:
        """
        Get summary of all processed datasets.
        
        Returns:
        --------
        dict
            Summary of processing results
        """
        summary = {
            'total_datasets_processed': len(self.processed_datasets),
            'datasets': {}
        }
        
        for dataset_type, result in self.processed_datasets.items():
            summary['datasets'][dataset_type] = {
                'success': result['success'],
                'records': result['final_shape'][0] if result['success'] else 0,
                'features': result['final_shape'][1] if result['success'] else 0,
                'features_added': result['features_added'] if result['success'] else 0,
                'file_path': result['processed_data_path'] if result['success'] else None
            }
        
        return summary


# Convenience functions for backward compatibility and easier usage
def process_single_dataset(config_path: str, verbose: bool = True) -> Dict:
    """
    Convenience function to process a single dataset.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict
        Processing results
    """
    engineer = EnhancedFeatureEngineer(verbose=verbose)
    return engineer.process_dataset(config_path)


def process_multiple_datasets_batch(config_paths: List[str], verbose: bool = True) -> Dict[str, Dict]:
    """
    Convenience function to process multiple datasets in batch.
    
    Parameters:
    -----------
    config_paths : List[str]
        List of configuration file paths
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    Dict[str, Dict]
        Results for each dataset
    """
    engineer = EnhancedFeatureEngineer(verbose=verbose)
    return engineer.process_multiple_datasets(config_paths)


def validate_dataset_consistency(config_paths: List[str]) -> Dict:
    """
    Validate consistency across multiple dataset configurations.
    
    Parameters:
    -----------
    config_paths : List[str]
        List of configuration file paths
        
    Returns:
    --------
    dict
        Validation results
    """
    engineer = EnhancedFeatureEngineer(verbose=False)
    consistency_results = {
        'configs_valid': [],
        'configs_invalid': [],
        'feature_consistency': True,
        'target_variable_consistency': True,
        'errors': []
    }
    
    target_variables = set()
    independent_variables = set()
    
    for config_path in config_paths:
        try:
            config = engineer.load_config(config_path)
            is_valid, errors = engineer.validate_config(config)
            
            if is_valid:
                consistency_results['configs_valid'].append(config_path)
                
                # Check feature consistency
                target_var = config['features']['dependent_variable']
                target_variables.add(target_var)
                
                indep_vars = config['features'].get('independent_variables', [])
                if independent_variables:
                    if set(indep_vars) != independent_variables:
                        consistency_results['feature_consistency'] = False
                else:
                    independent_variables = set(indep_vars)
                    
            else:
                consistency_results['configs_invalid'].append({
                    'config_path': config_path,
                    'errors': errors
                })
                
        except Exception as e:
            consistency_results['configs_invalid'].append({
                'config_path': config_path,
                'errors': [str(e)]
            })
    
    # Check target variable consistency
    if len(target_variables) > 1:
        consistency_results['target_variable_consistency'] = False
        consistency_results['errors'].append(
            f"Multiple target variables found: {list(target_variables)}"
        )
    
    return consistency_results
