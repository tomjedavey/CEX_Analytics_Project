#!/usr/bin/env python3
"""
AS_1 Comparative Analysis Module

This module provides comprehensive functionality for comparing AS_1 linear regression models
across different datasets (full dataset vs cluster-specific models). It includes performance
comparison, feature importance analysis, and business insights generation.

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


class AS1ComparativeAnalyzer:
    """Comprehensive comparative analysis for AS_1 models."""
    
    def __init__(self, config_paths: List[str]):
        """Initialize analyzer with multiple configuration paths."""
        self.config_paths = config_paths
        self.configs = self._load_configs()
        self.model_results = {}
        self.comparison_results = {}
        
    def _load_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files."""
        configs = {}
        for config_path in self.config_paths:
            with open(config_path, 'r') as f:
                configs[config_path] = yaml.safe_load(f)
        return configs
    
    def _get_dataset_name(self, config_path: str) -> str:
        """Extract dataset name from config path."""
        return Path(config_path).stem.replace('config_AS_1_', '')
    
    def load_model_results(self) -> Dict[str, Any]:
        """Load training and testing results for all models."""
        print("Loading model results...")
        
        for config_path in self.config_paths:
            dataset_name = self._get_dataset_name(config_path)
            config = self.configs[config_path]
            
            try:
                # Load training metrics
                training_metrics_path = config.get('output', {}).get('metrics_path', 'data/scores/AS_1_metrics.json')
                if not os.path.isabs(training_metrics_path):
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                    training_metrics_path = os.path.join(project_root, training_metrics_path)
                
                with open(training_metrics_path, 'r') as f:
                    training_data = json.load(f)
                
                # Load testing metrics
                test_metrics_path = training_metrics_path.replace('metrics', 'test_metrics')
                
                test_data = {}
                if os.path.exists(test_metrics_path):
                    with open(test_metrics_path, 'r') as f:
                        test_data = json.load(f)
                
                # Load test results CSV
                test_results_path = config.get('output', {}).get('test_results_path', 'data/scores/AS_1_test_results.csv')
                if not os.path.isabs(test_results_path):
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                    test_results_path = os.path.join(project_root, test_results_path)
                
                test_predictions = None
                if os.path.exists(test_results_path):
                    test_predictions = pd.read_csv(test_results_path)
                
                self.model_results[dataset_name] = {
                    'config_path': config_path,
                    'training_data': training_data,
                    'test_data': test_data,
                    'test_predictions': test_predictions,
                    'dataset_size': self._get_dataset_size(config),
                    'feature_count': len(config['features']['independent_variables'])
                }
                
                print(f"✅ Loaded results for {dataset_name}")
                
            except Exception as e:
                print(f"❌ Failed to load results for {dataset_name}: {e}")
                self.model_results[dataset_name] = None
        
        return self.model_results
    
    def _get_dataset_size(self, config: Dict[str, Any]) -> int:
        """Get dataset size from processed data."""
        try:
            data_path = config['data']['processed_data_path']
            if not os.path.isabs(data_path):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                data_path = os.path.join(project_root, data_path)
            
            df = pd.read_csv(data_path)
            return len(df)
        except:
            return 0
    
    def compare_model_performance(self) -> Dict[str, Any]:
        """Compare performance metrics across all models."""
        print("\nComparing model performance...")
        
        performance_comparison = {
            'summary_table': {},
            'best_performers': {},
            'performance_differences': {},
            'statistical_significance': {}
        }
        
        # Create summary table
        summary_data = []
        for dataset_name, results in self.model_results.items():
            if results and results['training_data']:
                training_metrics = results['training_data']['metrics']
                test_metrics = results['test_data'].get('performance_metrics', {}) if results['test_data'] else {}
                
                summary_data.append({
                    'Dataset': dataset_name,
                    'Dataset_Size': results['dataset_size'],
                    'Features': results['feature_count'],
                    'Train_R2': training_metrics.get('train_r2', 0),
                    'Test_R2': training_metrics.get('test_r2', 0),
                    'Train_RMSE': training_metrics.get('train_rmse', 0),
                    'Test_RMSE': training_metrics.get('test_rmse', 0),
                    'Train_MAE': training_metrics.get('train_mae', 0),
                    'Test_MAE': training_metrics.get('test_mae', 0),
                    'Overfitting_Gap_R2': training_metrics.get('train_r2', 0) - training_metrics.get('test_r2', 0),
                    'Training_Time': training_metrics.get('training_time_seconds', 0)
                })
        
        summary_df = pd.DataFrame(summary_data)
        performance_comparison['summary_table'] = summary_df.to_dict('records')
        
        # Identify best performers
        if not summary_df.empty:
            performance_comparison['best_performers'] = {
                'highest_test_r2': {
                    'dataset': summary_df.loc[summary_df['Test_R2'].idxmax(), 'Dataset'],
                    'value': summary_df['Test_R2'].max()
                },
                'lowest_test_rmse': {
                    'dataset': summary_df.loc[summary_df['Test_RMSE'].idxmin(), 'Dataset'],
                    'value': summary_df['Test_RMSE'].min()
                },
                'lowest_overfitting': {
                    'dataset': summary_df.loc[summary_df['Overfitting_Gap_R2'].idxmin(), 'Dataset'],
                    'value': summary_df['Overfitting_Gap_R2'].min()
                },
                'fastest_training': {
                    'dataset': summary_df.loc[summary_df['Training_Time'].idxmin(), 'Dataset'],
                    'value': summary_df['Training_Time'].min()
                }
            }
        
        # Calculate performance differences
        if len(summary_df) > 1:
            cluster_models = summary_df[summary_df['Dataset'].str.contains('cluster')]
            full_model = summary_df[summary_df['Dataset'] == 'full_dataset']
            
            if not cluster_models.empty and not full_model.empty:
                full_r2 = full_model['Test_R2'].iloc[0]
                full_rmse = full_model['Test_RMSE'].iloc[0]
                
                performance_comparison['performance_differences'] = {
                    'cluster_vs_full': {
                        'r2_improvements': {
                            row['Dataset']: row['Test_R2'] - full_r2 
                            for _, row in cluster_models.iterrows()
                        },
                        'rmse_improvements': {
                            row['Dataset']: full_rmse - row['Test_RMSE'] 
                            for _, row in cluster_models.iterrows()
                        }
                    }
                }
        
        self.comparison_results['performance'] = performance_comparison
        return performance_comparison
    
    def compare_feature_importance(self) -> Dict[str, Any]:
        """Compare feature importance across models."""
        print("Comparing feature importance...")
        
        feature_comparison = {
            'coefficient_comparison': {},
            'feature_rankings': {},
            'feature_stability': {},
            'unique_features': {}
        }
        
        all_coefficients = {}
        feature_rankings = {}
        
        for dataset_name, results in self.model_results.items():
            if results and results['training_data']:
                metrics = results['training_data']['metrics']
                coefficients = metrics.get('feature_coefficients', {})
                
                all_coefficients[dataset_name] = coefficients
                
                # Rank features by absolute coefficient value
                if coefficients:
                    ranked_features = sorted(
                        coefficients.items(), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )
                    feature_rankings[dataset_name] = ranked_features
        
        feature_comparison['coefficient_comparison'] = all_coefficients
        feature_comparison['feature_rankings'] = feature_rankings
        
        # Analyze feature stability (how consistent are rankings across models)
        if len(feature_rankings) > 1:
            all_features = set()
            for rankings in feature_rankings.values():
                all_features.update([f[0] for f in rankings])
            
            feature_stability = {}
            for feature in all_features:
                positions = []
                for dataset_name, rankings in feature_rankings.items():
                    feature_names = [f[0] for f in rankings]
                    if feature in feature_names:
                        positions.append(feature_names.index(feature) + 1)
                    else:
                        positions.append(len(feature_names) + 1)  # Rank as last if not present
                
                feature_stability[feature] = {
                    'mean_rank': np.mean(positions),
                    'std_rank': np.std(positions),
                    'rank_range': max(positions) - min(positions),
                    'consistency_score': 1 - (np.std(positions) / len(all_features))
                }
            
            feature_comparison['feature_stability'] = feature_stability
        
        self.comparison_results['features'] = feature_comparison
        return feature_comparison
    
    def analyze_residual_patterns(self) -> Dict[str, Any]:
        """Analyze residual patterns across models."""
        print("Analyzing residual patterns...")
        
        residual_analysis = {
            'residual_statistics': {},
            'outlier_comparison': {},
            'prediction_quality': {}
        }
        
        for dataset_name, results in self.model_results.items():
            if results and results['test_predictions'] is not None:
                df = results['test_predictions']
                
                residuals = df['residual'].values
                abs_residuals = df['abs_residual'].values
                
                residual_analysis['residual_statistics'][dataset_name] = {
                    'mean_residual': float(np.mean(residuals)),
                    'std_residual': float(np.std(residuals)),
                    'mean_abs_residual': float(np.mean(abs_residuals)),
                    'median_abs_residual': float(np.median(abs_residuals)),
                    'q95_abs_residual': float(np.percentile(abs_residuals, 95)),
                    'max_abs_residual': float(np.max(abs_residuals)),
                    'residual_skewness': float(self._calculate_skewness(residuals)),
                    'residual_kurtosis': float(self._calculate_kurtosis(residuals))
                }
                
                # Outlier analysis
                q75 = np.percentile(abs_residuals, 75)
                q25 = np.percentile(abs_residuals, 25)
                iqr = q75 - q25
                outlier_threshold = q75 + 1.5 * iqr
                n_outliers = np.sum(abs_residuals > outlier_threshold)
                
                residual_analysis['outlier_comparison'][dataset_name] = {
                    'outlier_threshold': float(outlier_threshold),
                    'n_outliers': int(n_outliers),
                    'outlier_percentage': float(n_outliers / len(abs_residuals) * 100)
                }
        
        self.comparison_results['residuals'] = residual_analysis
        return residual_analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def generate_business_insights(self) -> Dict[str, Any]:
        """Generate business insights from model comparisons."""
        print("Generating business insights...")
        
        insights = {
            'key_findings': [],
            'cluster_characteristics': {},
            'recommendations': [],
            'model_selection_guidance': {}
        }
        
        # Analyze performance differences
        if 'performance' in self.comparison_results:
            perf_data = self.comparison_results['performance']
            
            # Key findings
            if 'best_performers' in perf_data:
                best_r2 = perf_data['best_performers']['highest_test_r2']
                insights['key_findings'].append(
                    f"Best performing model: {best_r2['dataset']} with R² = {best_r2['value']:.4f}"
                )
                
                lowest_rmse = perf_data['best_performers']['lowest_test_rmse']
                insights['key_findings'].append(
                    f"Most accurate model: {lowest_rmse['dataset']} with RMSE = {lowest_rmse['value']:.4f}"
                )
            
            # Cluster vs full dataset comparison
            if 'performance_differences' in perf_data:
                cluster_diff = perf_data['performance_differences']['cluster_vs_full']
                
                r2_improvements = cluster_diff['r2_improvements']
                rmse_improvements = cluster_diff['rmse_improvements']
                
                best_cluster_r2 = max(r2_improvements, key=r2_improvements.get)
                best_cluster_rmse = max(rmse_improvements, key=rmse_improvements.get)
                
                if r2_improvements[best_cluster_r2] > 0:
                    insights['key_findings'].append(
                        f"Cluster-specific models outperform full dataset model. "
                        f"Best improvement: {best_cluster_r2} (+{r2_improvements[best_cluster_r2]:.4f} R²)"
                    )
                else:
                    insights['key_findings'].append(
                        "Full dataset model performs better than cluster-specific models"
                    )
        
        # Model selection guidance
        summary_df = pd.DataFrame(self.comparison_results.get('performance', {}).get('summary_table', []))
        if not summary_df.empty:
            best_overall = summary_df.loc[summary_df['Test_R2'].idxmax()]
            
            insights['model_selection_guidance'] = {
                'recommended_model': best_overall['Dataset'],
                'reasoning': f"Highest test R² ({best_overall['Test_R2']:.4f}) with acceptable overfitting gap",
                'alternative_considerations': []
            }
            
            # Check for overfitting
            if best_overall['Overfitting_Gap_R2'] > 0.1:
                insights['model_selection_guidance']['alternative_considerations'].append(
                    "Consider model with lower overfitting gap for better generalization"
                )
            
            # Check for computational efficiency
            fastest_model = summary_df.loc[summary_df['Training_Time'].idxmin()]
            if fastest_model['Dataset'] != best_overall['Dataset']:
                insights['model_selection_guidance']['alternative_considerations'].append(
                    f"For faster training, consider {fastest_model['Dataset']} "
                    f"(training time: {fastest_model['Training_Time']:.2f}s)"
                )
        
        # Recommendations
        insights['recommendations'] = [
            "Use cluster-specific models if they show significant performance improvements",
            "Monitor for overfitting in smaller cluster datasets",
            "Consider ensemble approaches combining predictions from multiple models",
            "Focus feature engineering on consistently important features across models",
            "Validate model performance on completely new data before deployment"
        ]
        
        self.comparison_results['insights'] = insights
        return insights
    
    def save_comparison_report(self, output_path: str = None) -> str:
        """Save comprehensive comparison report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/reports/AS_1_comparative_analysis_{timestamp}.json"
        
        if not os.path.isabs(output_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            output_path = os.path.join(project_root, output_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Compile comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {
                'models_analyzed': len(self.config_paths),
                'config_paths': self.config_paths,
                'successful_models': len([r for r in self.model_results.values() if r is not None])
            },
            'comparison_results': self.comparison_results,
            'model_results_summary': {
                name: {
                    'dataset_size': results['dataset_size'] if results else 0,
                    'feature_count': results['feature_count'] if results else 0,
                    'has_training_data': bool(results and results['training_data']),
                    'has_test_data': bool(results and results['test_data'])
                }
                for name, results in self.model_results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comprehensive analysis report saved to: {output_path}")
        return output_path
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete comparative analysis."""
        print("RUNNING COMPREHENSIVE AS_1 MODEL COMPARISON")
        print("=" * 60)
        
        # Load all model results
        self.load_model_results()
        
        # Run all comparisons
        performance_results = self.compare_model_performance()
        feature_results = self.compare_feature_importance()
        residual_results = self.analyze_residual_patterns()
        insights = self.generate_business_insights()
        
        # Save comprehensive report
        report_path = self.save_comparison_report()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print(f"Report saved to: {report_path}")
        
        return {
            'performance': performance_results,
            'features': feature_results,
            'residuals': residual_results,
            'insights': insights,
            'report_path': report_path
        }


def compare_as1_models(config_paths: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Compare AS_1 models across multiple configurations.
    
    Args:
        config_paths: List of configuration file paths
        verbose: Whether to enable verbose output
        
    Returns:
        Dictionary containing comparative analysis results
    """
    analyzer = AS1ComparativeAnalyzer(config_paths)
    return analyzer.run_full_analysis()


# Example usage
if __name__ == "__main__":
    # Example comparative analysis
    config_paths = [
        "/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_AS_1_full_dataset.yaml",
        "/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_AS_1_cluster_0.yaml",
        "/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_AS_1_cluster_1.yaml"
    ]
    
    results = compare_as1_models(config_paths, verbose=True)
