"""
Utility functions for wildlife classification pipeline
"""

from .cuda_fixes import setup_cuda_multiprocessing

__all__ = ['setup_cuda_multiprocessing']