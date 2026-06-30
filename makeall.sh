#!/bin/bash

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.gemm \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir gemm_MINI_baseline

cd ./benches/experiments/gemm_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.gemver \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir gemver_MINI_baseline

cd ./benches/experiments/gemver_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.gesummv \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir gesummv_MINI_baseline

cd ./benches/experiments/gesummv_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.symm \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir symm_MINI_baseline

cd ./benches/experiments/symm_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.syr2k \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir syr2k_MINI_baseline

cd ./benches/experiments/syr2k_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.syrk \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir syrk_MINI_baseline

cd ./benches/experiments/syrk_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.trmm \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir trmm_MINI_baseline

cd ./benches/experiments/trmm_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.kernels.atax \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir atax_MINI_baseline

cd ./benches/experiments/atax_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.kernels.bicg \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir bicg_MINI_baseline

cd ./benches/experiments/bicg_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.kernels.doitgen \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir doitgen_MINI_baseline

cd ./benches/experiments/doitgen_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.kernels.mvt \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir mvt_MINI_baseline

cd ./benches/experiments/mvt_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.kernels.threemm \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir threemm_MINI_baseline

cd ./benches/experiments/threemm_MINI_baseline
pixi run make
cd ../../../

pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.kernels.twomm \
    --dataset MINI \
    --dtype float32 \
    --device xcu280-2Lfsvh2892-VVD \
    --clock_period 5 \
    --memory_policy NO_BRAM \
    --target verilog_results \
    --output_dir twomm_MINI_baseline

cd ./benches/experiments/twomm_MINI_baseline
pixi run make
cd ../../../
