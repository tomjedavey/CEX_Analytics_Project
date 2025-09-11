import os
import shutil
from typing import List

def save_config_files(
    config_filenames: List[str],
    source_dir: str = os.path.join(os.path.dirname(__file__), '../config'),
    dest_dir: str = os.path.join(os.path.dirname(__file__), '../../artifacts/Configurations_and_Pipeline_Metadata')
) -> None:
    """
    Copies specified config YAML files from the source config directory to the destination directory in an organised way.
    """
    # Do not create the destination directory; assume it already exists
    for filename in config_filenames:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_dir):
            print(f"Destination directory does not exist: {dest_dir}")
            continue
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        else:
            print(f"Config file not found: {src_path}")

if __name__ == "__main__":
    config_files = [
        "config_behavioural_volatility.yaml",
        "config_cluster.yaml",
        "config_cross_domain_engagement.yaml",
        "config_interaction_mode.yaml",
        "config_revenue_proxy.yaml"
    ]
    save_config_files(config_files)
