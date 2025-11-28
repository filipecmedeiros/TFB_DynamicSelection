#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to plot and compare forecasting results from multiple models in TFB.

This script reads the evaluation results, decodes the predictions and actual values,
and creates comparison plots for all models.
"""

import os
import sys
import base64
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Add TFB to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ts_benchmark.recording import load_record_data


def decode_data(encoded_str: str) -> Any:
    """
    Decode base64-encoded pickle data.
    
    Args:
        encoded_str: Base64-encoded string
        
    Returns:
        Decoded Python object (DataFrame or list of DataFrames)
    """
    if pd.isna(encoded_str):
        return None
    try:
        decoded = base64.b64decode(encoded_str.encode('utf-8'))
        return pickle.loads(decoded)
    except Exception as e:
        print(f"Error decoding data: {e}")
        return None


def extract_values(data: Any, variable_idx: int = 0) -> np.ndarray:
    """
    Extract values from decoded data.
    
    Args:
        data: Decoded data (DataFrame or list of DataFrames)
        variable_idx: Index of variable to extract (for multivariate)
        
    Returns:
        Numpy array of values
    """
    if data is None:
        return None
        
    # Handle rolling forecast (list of DataFrames)
    if isinstance(data, list):
        data = pd.concat(data, axis=0)
    
    # Extract values
    if isinstance(data, pd.DataFrame):
        if data.shape[1] > variable_idx:
            return data.iloc[:, variable_idx].values
        else:
            return data.iloc[:, 0].values
    else:
        return np.array(data).flatten()


def plot_single_series(
    results_df: pd.DataFrame,
    series_name: str,
    variable_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 8)
) -> plt.Figure:
    """
    Plot forecast comparison for a single time series.
    
    Args:
        results_df: DataFrame with evaluation results
        series_name: Name of the time series to plot
        variable_idx: Index of variable to plot (for multivariate)
        save_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Filter results for the specific series
    series_results = results_df[results_df['file_name'] == series_name].copy()
    
    if len(series_results) == 0:
        print(f"No results found for series: {series_name}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values (same for all models)
    first_row = series_results.iloc[0]
    actual_data = decode_data(first_row['actual_data'])
    
    if actual_data is None:
        print(f"No actual data found for {series_name}. Make sure save_true_pred was enabled.")
        return None
    
    actual_values = extract_values(actual_data, variable_idx)
    
    if actual_values is None:
        print(f"Could not extract actual values for {series_name}")
        return None
    
    # Plot actual values
    time_steps = np.arange(len(actual_values))
    ax.plot(time_steps, actual_values, label='Actual', 
            linewidth=2.5, color='black', alpha=0.8, zorder=100)
    
    # Plot predictions for each model
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(series_results))))
    if len(series_results) > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, len(series_results)))
    
    for (idx, row), color in zip(series_results.iterrows(), colors):
        model_name = row['model_name']
        predicted_data = decode_data(row['inference_data'])
        
        if predicted_data is None:
            print(f"No prediction data for {model_name}")
            continue
        
        predicted_values = extract_values(predicted_data, variable_idx)
        
        if predicted_values is None:
            print(f"Could not extract predictions for {model_name}")
            continue
        
        # Get metrics for legend
        mae = row.get('mae', np.nan)
        mse = row.get('mse', np.nan)
        
        label = f'{model_name} (MAE: {mae:.3f}, MSE: {mse:.3f})'
        
        pred_time_steps = np.arange(len(predicted_values))
        ax.plot(pred_time_steps, predicted_values, label=label,
                linewidth=1.5, alpha=0.7, color=color)
    
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Forecast Comparison - {series_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def plot_all_series(
    results_df: pd.DataFrame,
    output_dir: str = 'plots',
    variable_idx: int = 0,
    create_pdf: bool = True
) -> None:
    """
    Plot forecast comparisons for all time series in the results.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
        variable_idx: Index of variable to plot (for multivariate)
        create_pdf: Whether to create a combined PDF
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique series names
    series_names = results_df['file_name'].unique()
    print(f"Found {len(series_names)} time series to plot")
    
    if create_pdf:
        pdf_path = os.path.join(output_dir, 'all_forecasts_comparison.pdf')
        pdf = PdfPages(pdf_path)
    
    for series_name in series_names:
        print(f"\nPlotting {series_name}...")
        
        # Create individual PNG
        png_path = os.path.join(output_dir, f'{series_name.replace(".csv", "")}_comparison.png')
        fig = plot_single_series(results_df, series_name, variable_idx, save_path=png_path)
        
        # Add to PDF
        if create_pdf and fig is not None:
            pdf.savefig(fig, bbox_inches='tight')
        
        if fig is not None:
            plt.close(fig)
    
    if create_pdf:
        pdf.close()
        print(f"\n✓ Combined PDF saved to: {pdf_path}")
    
    print(f"\n✓ All plots saved to: {output_dir}/")


def create_summary_plot(results_df: pd.DataFrame, output_path: str = 'plots/summary.png') -> None:
    """
    Create a summary plot showing model performance across all series.
    
    Args:
        results_df: DataFrame with evaluation results
        output_path: Path to save the summary plot
    """
    # Group by model and calculate mean metrics
    metrics = ['mae', 'mse', 'rmse', 'mape']
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if not available_metrics:
        print("No metrics found for summary plot")
        return
    
    summary = results_df.groupby('model_name')[available_metrics].mean()
    
    # Create subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, available_metrics):
        summary[metric].sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel(metric.upper(), fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(f'Average {metric.upper()} Across All Series', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Summary plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot forecast comparisons from TFB evaluation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='result/daily',
        help='Directory containing result files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--series-name',
        type=str,
        default=None,
        help='Specific series to plot (if None, plots all)'
    )
    parser.add_argument(
        '--variable-idx',
        type=int,
        default=0,
        help='Index of variable to plot for multivariate series'
    )
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Do not create combined PDF'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only create summary plot'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results_df = load_record_data([args.results_dir])
    
    print(f"Loaded {len(results_df)} evaluation records")
    print(f"Models: {results_df['model_name'].unique().tolist()}")
    print(f"Series: {results_df['file_name'].nunique()} unique time series")
    
    # Check if predictions are available
    if 'actual_data' not in results_df.columns or 'inference_data' not in results_df.columns:
        print("\n⚠️  ERROR: No prediction data found in results!")
        print("Make sure you ran the evaluation with save_true_pred=true")
        return
    
    # Create summary plot
    if args.summary_only or not args.series_name:
        summary_path = os.path.join(args.output_dir, 'summary.png')
        os.makedirs(args.output_dir, exist_ok=True)
        create_summary_plot(results_df, summary_path)
    
    if args.summary_only:
        return
    
    # Plot specific series or all series
    if args.series_name:
        output_path = os.path.join(args.output_dir, f'{args.series_name.replace(".csv", "")}_comparison.png')
        os.makedirs(args.output_dir, exist_ok=True)
        plot_single_series(results_df, args.series_name, args.variable_idx, output_path)
    else:
        plot_all_series(results_df, args.output_dir, args.variable_idx, create_pdf=not args.no_pdf)


if __name__ == '__main__':
    main()