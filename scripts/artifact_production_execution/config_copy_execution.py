"""
Execution script to copy config YAML files from source_code_package/config/ to artifacts/Configurations_and_Pipeline_Metadata.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../source_code_package/utils')))
from source_code_package.utils.output_config_loader import save_config_files

if __name__ == "__main__":
    config_files = [
        "config_behavioural_volatility.yaml",
        "config_cluster.yaml",
        "config_cross_domain_engagement.yaml",
        "config_interaction_mode.yaml",
        "config_revenue_proxy.yaml"
    ]
    save_config_files(config_files)
