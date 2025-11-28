# TFB Dash App with Integrated Forecast Plots

This guide explains how to use the enhanced Dash reporting interface with integrated forecast visualization.

## What's New

The Dash app now includes a **Forecast Plots** page that allows you to:
- Interactively visualize predictions from all models
- Compare model forecasts against actual values
- View performance metrics in hover tooltips
- Filter by time series and variables
- See model performance summaries

## Quick Start

### 1. Run Evaluation with Predictions Enabled

Make sure your evaluation saves predictions:

```bash
python scripts/run_benchmark.py \
  --config-path "fixed_forecast_config_daily.json" \
  --data-name-list "your_data.csv" \
  --model-name "time_series_library.DLinear" "time_series_library.Informer" \
  --save-path "MyExperiment" \
  --save-true-pred True \
  --report-method dash
```

**Important**: The `--save-true-pred True` flag is required for forecast plots!

### 2. Launch Dash App for Existing Results

Create a Python script (`view_dash.py`):

```python
from ts_benchmark.report import report

report_config = {
    "log_files_list": ["result/daily/"],  # Your results directory
    "report_metrics": ["mae", "mse", "rmse", "mape", "smape"],
    "aggregate_type": "mean",
    "fill_type": "mean_value",
    "null_value_threshold": 0.3,
    "host": "0.0.0.0",
    "port": "12345",
    "debug": False
}

report(report_config, report_method="dash")
```

Run it:
```bash
python view_dash.py
```

### 3. Access the Dash App

Open your browser and navigate to:
- **Leaderboard**: `http://localhost:12345/leaderboard`
- **Query Interface**: `http://localhost:12345/query`
- **Forecast Plots**: `http://localhost:12345/forecast-plots` ⭐ NEW!

## Forecast Plots Page Features

### Interactive Visualization

The Forecast Plots page provides:

1. **Time Series Selection**
   - Dropdown menu to select which time series to visualize
   - Automatically populated with all series in your results

2. **Variable Selection**
   - For multivariate time series, select which variable to plot
   - Default is variable 0 (first variable)

3. **Interactive Plot**
   - Zoom, pan, and hover over data points
   - Actual values shown in bold black line
   - Each model's predictions in different colors
   - Hover tooltips show:
     - Model name
     - Time step
     - Predicted value
     - MAE, MSE, RMSE metrics

4. **Performance Summary Table**
   - Shows all models ranked by performance
   - Displays key metrics for the selected series
   - Automatically updates when you change series

### Plot Features

- **Zoom**: Click and drag to zoom into a region
- **Pan**: Hold shift and drag to pan
- **Reset**: Double-click to reset view
- **Toggle Models**: Click on legend items to show/hide models
- **Hover**: Hover over lines to see detailed information

## Configuration Options

### Enable/Disable Prediction Storage

By default, the app now keeps prediction data. To optimize memory:

Edit `ts_benchmark/report/report_dash/app.py`:

```python
# To drop predictions and save memory:
ARTIFACT_COLUMNS = [
    FieldNames.ACTUAL_DATA,
    FieldNames.INFERENCE_DATA,
]

# To keep predictions (default):
ARTIFACT_COLUMNS = []
```

### Customize Plot Appearance

Edit `ts_benchmark/report/report_dash/pages/forecast_plots.py`:

```python
# Change plot height
fig.update_layout(height=800)  # Default is 600

# Change color scheme
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# Modify hover template
hovertemplate=(
    f'<b>{model_name}</b><br>'
    'Custom info here'
)
```

## Memory Considerations

### For Large Datasets

If you have many time series or large predictions:

1. **Option 1**: Drop predictions after viewing
   ```python
   ARTIFACT_COLUMNS = [
       FieldNames.ACTUAL_DATA,
       FieldNames.INFERENCE_DATA,
   ]
   ```

2. **Option 2**: Use the standalone plotting script
   ```bash
   python plot_forecast_comparison.py --results-dir result/daily
   ```

3. **Option 3**: Filter results before loading
   ```python
   # Load only specific models
   results_df = load_record_data(['result/daily'])
   filtered = results_df[results_df['model_name'].isin(['DLinear', 'Informer'])]
   
   report_config = {
       "log_files_list": filtered,  # Pass DataFrame directly
       ...
   }
   ```

## Comparison: Dash vs Standalone Plots

| Feature | Dash App | Standalone Script |
|---------|----------|-------------------|
| Interactive | ✅ Yes (Plotly) | ❌ No (Matplotlib) |
| Real-time filtering | ✅ Yes | ❌ No |
| Memory usage | 🟡 Higher | ✅ Lower |
| Batch export | ❌ No | ✅ Yes (PDF) |
| Customization | 🟡 Moderate | ✅ High |
| Ease of use | ✅ Very easy | 🟡 Moderate |

## Troubleshooting

### "Prediction data not available"

**Cause**: Evaluation was run without `save_true_pred=true`

**Solution**: Re-run evaluation with:
```bash
--save-true-pred True
```

Or in config:
```json
{
  "evaluation_config": {
    "strategy_args": {
      "save_true_pred": true
    }
  }
}
```

### Dash app is slow

**Cause**: Large prediction data in memory

**Solutions**:
1. Filter to fewer models before loading
2. Use standalone plotting script instead
3. Enable prediction dropping in `app.py`

### Plot shows wrong variable

**Cause**: Variable index doesn't match your data

**Solution**: Adjust the "Variable Index" input in the Dash interface

### Can't see all models in legend

**Cause**: Too many models to display

**Solution**: 
1. Click and drag legend to reposition
2. Filter results to fewer models
3. Use standalone script with custom layout

## Advanced Usage

### Custom Dash Page

Create your own visualization page:

```python
# ts_benchmark/report/report_dash/pages/my_custom_page.py
import dash
from dash import html, dcc
from ts_benchmark.report.report_dash.memory import READONLY_MEMORY

dash.register_page(__name__, name="My Custom Page")

raw_data = READONLY_MEMORY["raw_data"]

# Your custom visualization code here
layout = html.Div([
    html.H3("My Custom Analysis"),
    # Add your components
])
```

### Export Plots from Dash

While viewing a plot in Dash:
1. Hover over the plot
2. Click the camera icon in the top-right
3. Save as PNG

## Best Practices

1. **Always enable `save_true_pred`** when you want to visualize forecasts
2. **Use Dash for exploration**, standalone script for publication-quality plots
3. **Filter data** before loading if you have many models
4. **Check memory usage** if working with large datasets
5. **Use variable index 0** for univariate series

## Examples

### Example 1: Quick Exploration

```bash
# Run with Dash reporting
python scripts/run_benchmark.py \
  --config-path "config.json" \
  --data-name-list "series1.csv" "series2.csv" \
  --model-name "DLinear" "Informer" "NBEATS" \
  --save-true-pred True \
  --report-method dash
```

Then navigate to `http://localhost:12345/forecast-plots`

### Example 2: Analyze Existing Results

```python
from ts_benchmark.report import report

report({
    "log_files_list": ["result/my_experiment/"],
    "report_metrics": ["mae", "mse"],
    "aggregate_type": "mean",
}, report_method="dash")
```

### Example 3: Compare Specific Models

```python
from ts_benchmark.recording import load_record_data
from ts_benchmark.report import report

# Load and filter
results = load_record_data(['result/daily'])
top_3 = results[results['model_name'].isin(['DLinear', 'Informer', 'NBEATS'])]

# Launch Dash with filtered data
report({
    "log_files_list": top_3,
    "report_metrics": ["mae", "mse"],
}, report_method="dash")
```

## Support

For issues:
1. Check that predictions were saved during evaluation
2. Verify the Dash app is running on the correct port
3. Check browser console for JavaScript errors
4. Try the standalone plotting script as an alternative