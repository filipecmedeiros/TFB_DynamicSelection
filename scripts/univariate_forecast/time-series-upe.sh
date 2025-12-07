EXPERIMENT="test"
CLEAN="TRUE"


if [ "$CLEAN" = "TRUE" ]; then
    rm -rf ./plots/${EXPERIMENT}
    rm -rf ./result/${EXPERIMENT}/*
fi


# Dynamic Selection with Airline
python ./scripts/run_benchmark.py \
    --config-path "fixed_forecast_config_daily.json" \
    --data-name-list "airline.csv" \
    --model-name   "dynamic_selection.DynamicSelection" \
    --model-hyper-params "{\"window_size\":10, \"similar_windows\": 5, \"n_models\": 3, \"h\": 30}" \
    --strategy-args '{"horizon": 30}' \
    --save-path "${EXPERIMENT}"


# All models with airline
# python ./scripts/run_benchmark.py \
#     --config-path "fixed_forecast_config_daily.json" \
#     --data-name-list "airline.csv" \
#     --model-name   "dynamic_selection.DynamicSelection" "time_series_library.Informer" "darts.AutoARIMA" "darts.LinearRegressionModel" "darts.NBEATSModel" "darts.NHiTSModel" \
#     --model-hyper-params "{\"window_size\":10, \"similar_windows\": 3, \"n_models\": 3, \"h\": 30}" "{\"d_model\":8,\"d_ff\":8,\"factor\":3,\"num_epochs\":5}" \
#     --adapter "None" "transformer_adapter" \
#     --strategy-args '{"horizon": 30}' \
#     --save-path "${EXPERIMENT}" \


# All models with all datasets
# python ./scripts/run_benchmark.py \
#     --config-path "fixed_forecast_config_daily.json" \
#     --data-name-list "APPLE.csv" "henon.csv" "river.csv" "airline.csv" "lake.csv" "sp.csv" "amz.csv" "laser.csv" "star.csv" "carsales.csv" "lynx.csv" "sunspot.csv" "coloradoRiver.csv" "msft.csv" "traffic.csv" "electricity.csv" "nordic.csv" "vehicle.csv" "eletric.csv" "pigs.csv" "volume_vendas_varejo_pe.csv" "gas.csv" "pollutions.csv" "wine.csv" "goldman.csv" "redwine.csv" \
#     --model-name   "dynamic_selection.DynamicSelection" "time_series_library.Informer" "darts.AutoARIMA" "darts.LinearRegressionModel" "darts.NBEATSModel" "darts.NHiTSModel" \
#     --model-hyper-params "{\"window_size\":10, \"similar_windows\": 3, \"n_models\": 3, \"h\": 30}" "{\"d_model\":8,\"d_ff\":8,\"factor\":3,\"num_epochs\":5}" \
#     --adapter "None" "transformer_adapter" \
#     --strategy-args '{"horizon": 30}' \
#     --save-path "research" \


python plot_forecast_comparison.py --results-dir result/${EXPERIMENT} --output-dir result/${EXPERIMENT}