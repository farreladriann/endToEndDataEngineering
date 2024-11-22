#!/usr/bin/env python
# run_tests.py

import pytest
import sys
import os

def run_tests():
    """Run all tests and return exit code"""
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Run pytest with verbosity
    exit_code = pytest.main([
        '-v',
        '--capture=no',
        '--tb=short',
        'tests/'
    ])
    
    return exit_code

if __name__ == '__main__':
    sys.exit(run_tests())