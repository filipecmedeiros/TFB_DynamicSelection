# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_daily.json" --data-name-list "m4_daily_dataset_368.csv" --model-name  "darts.AutoARIMA" "darts.LinearRegressionModel" "darts.NBEATSModel" "darts.NHiTSModel" "time_series_library.Informer" --save-path "daily" --gpus 5 --num-workers 3 --timeout 60000
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_daily.json" --data-name-list "m4_daily_dataset_368.csv" --model-name  "time_series_library.Informer" --model-hyper-params "{\"d_model\":8,\"d_ff\":8,\"factor\":3}" --adapter "transformer_adapter" --save-path "daily"   --gpus 5  --num-workers 3 --timeout 60000

# python ./scripts/run_benchmark.py \
#     --config-path "fixed_forecast_config_daily.json" \
#     --data-name-list "m4_daily_dataset_368.csv" \
#     --model-name "dynamic_selection.DynamicSelection" \
#     --model-hyper-params '{"k": 3, "n": 3, "h": 5}' \
#     --save-path "daily" \
#     --gpus 5 \
#     --num-workers 3

python ./scripts/run_benchmark.py \
    --config-path "fixed_forecast_config_daily.json" \
    --data-name-list "m4_daily_dataset_368.csv" \
    --model-name   "dynamic_selection.DynamicSelection" "time_series_library.Informer" "darts.AutoARIMA" "darts.LinearRegressionModel" "darts.NBEATSModel" "darts.NHiTSModel" \
    --model-hyper-params "{\"k\": 3, \"n\": 3, \"h\": 30}" "{\"d_model\":8,\"d_ff\":8,\"factor\":3,\"num_epochs\":5}" \
    --adapter "None" "transformer_adapter" \
    --strategy-args '{"horizon": 30}' \
    --save-path "daily" \
    --gpus 5 \
    --num-workers 3


