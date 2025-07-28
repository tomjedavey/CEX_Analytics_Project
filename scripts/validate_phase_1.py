#!/usr/bin/env python3
"""
Execution script for Phase 1 validation - AS_1 multi-dataset configuration.

This script serves as a simple execution wrapper around the core functionality
in source_code_package.utils.validation module.

Author: Tom Davey
Date: July 2025
"""

import os
import sys

# Add source_code_package to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
source_package_path = os.path.join(project_root, 'source_code_package')
sys.path.insert(0, source_package_path)

try:
    from utils.validation import validate_full_pipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure source_code_package structure is correct")
    exit(1)


def main():
    """Main validation function."""
    print("PHASE 1 VALIDATION - AS_1 MULTI-DATASET CONFIGURATION")
    print("=" * 60)
    
    # Run full pipeline validation using core functionality
    report = validate_full_pipeline(project_root)
    
    # Return success status
    return report['overall_success']


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
