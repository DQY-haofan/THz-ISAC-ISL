#!/usr/bin/env python3
"""
diagnostics.py - Unified diagnostics and data collection system
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List
import os

class DiagnosticsCollector:
    """Collect and analyze key metrics for debugging."""
    
    def __init__(self):
        self.results = {}
        self.warnings = []
        self.key_metrics = {}
        
    def check_value_sanity(self, name: str, value: float, 
                          expected_range: tuple = None,
                          units: str = "") -> bool:
        """Check if a value is within reasonable range."""
        is_sane = True
        
        # Check for NaN or Inf
        if np.isnan(value) or np.isinf(value):
            warning = f"ERROR: {name} is NaN or Inf!"
            self.warnings.append(warning)
            print(f"  ⚠️ {warning}")
            return False
        
        # Check for complex values that shouldn't be
        if isinstance(value, complex):
            warning = f"WARNING: {name} is complex: {value}, using real part {np.real(value)}"
            self.warnings.append(warning)
            print(f"  ⚠️ {warning}")
            value = np.real(value)
        
        # Check expected range if provided
        if expected_range:
            min_val, max_val = expected_range
            if value < min_val or value > max_val:
                warning = f"WARNING: {name} = {value:.3e} {units} outside expected range [{min_val:.3e}, {max_val:.3e}]"
                self.warnings.append(warning)
                print(f"  ⚠️ {warning}")
                is_sane = False
        
        return is_sane
    
    def add_key_metric(self, category: str, name: str, value: Any, 
                      expected_range: tuple = None, units: str = ""):
        """Add a key metric with sanity check."""
        if category not in self.key_metrics:
            self.key_metrics[category] = {}
        
        # Handle arrays
        if isinstance(value, np.ndarray):
            if value.size < 10:
                self.key_metrics[category][name] = value.tolist()
            else:
                # Store statistics for large arrays
                self.key_metrics[category][name] = {
                    'min': float(np.min(np.real(value))),
                    'max': float(np.max(np.real(value))),
                    'mean': float(np.mean(np.real(value))),
                    'std': float(np.std(np.real(value))),
                    'size': value.size
                }
        else:
            # Single value
            if isinstance(value, complex):
                value = float(np.real(value))
            self.key_metrics[category][name] = value
            
            # Check sanity if it's a number
            if isinstance(value, (int, float, np.number)):
                self.check_value_sanity(f"{category}/{name}", value, expected_range, units)
    
    def print_summary(self):
        """Print key metrics summary."""
        print("\n" + "="*70)
        print("KEY METRICS SUMMARY")
        print("="*70)
        
        for category, metrics in self.key_metrics.items():
            print(f"\n{category}:")
            for name, value in metrics.items():
                if isinstance(value, dict):
                    print(f"  {name}: min={value['min']:.3e}, max={value['max']:.3e}, "
                          f"mean={value['mean']:.3e}")
                elif isinstance(value, (int, float)):
                    print(f"  {name}: {value:.3e}")
                else:
                    print(f"  {name}: {value}")
        
        if self.warnings:
            print("\n" + "="*70)
            print(f"WARNINGS ({len(self.warnings)} issues found):")
            print("="*70)
            for warning in self.warnings:
                print(f"  • {warning}")
    
    def save_comprehensive_report(self, filename: str = "comprehensive_results"):
        """Save all results to a single comprehensive file."""
        report = {
            'timestamp': str(datetime.now()),
            'key_metrics': self.key_metrics,
            'warnings': self.warnings,
            'all_results': self.results
        }
        
        # Save as JSON
        json_path = f'results/{filename}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save as readable text
        txt_path = f'results/{filename}.txt'
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE THz ISL ISAC ANALYSIS REPORT\n")
            f.write(f"Generated: {report['timestamp']}\n")
            f.write("="*70 + "\n\n")
            
            # Key metrics
            f.write("KEY METRICS:\n")
            f.write("-"*40 + "\n")
            for category, metrics in self.key_metrics.items():
                f.write(f"\n{category}:\n")
                for name, value in metrics.items():
                    if isinstance(value, dict):
                        f.write(f"  {name}:\n")
                        for k, v in value.items():
                            f.write(f"    {k}: {v}\n")
                    else:
                        f.write(f"  {name}: {value}\n")
            
            # Warnings
            if self.warnings:
                f.write("\n" + "="*40 + "\n")
                f.write(f"WARNINGS ({len(self.warnings)} issues):\n")
                f.write("-"*40 + "\n")
                for warning in self.warnings:
                    f.write(f"• {warning}\n")
            
            # Detailed results
            f.write("\n" + "="*40 + "\n")
            f.write("DETAILED RESULTS:\n")
            f.write("-"*40 + "\n")
            for key, value in self.results.items():
                f.write(f"\n{key}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")
        
        print(f"\n✅ Comprehensive report saved to:")
        print(f"  - {json_path}")
        print(f"  - {txt_path}")
        
        return json_path, txt_path

# Global instance
diagnostics = DiagnosticsCollector()