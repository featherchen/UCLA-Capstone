# Comparative Evaluation of CPU and NPU Inference Frameworks for Edge LLMs

This repository contains the workflow and evaluation scripts for the Capstone Project

It demonstrates using two inference frameworks: llama.cpp & ExecuTorch on Snapdragon 8 Elite. 

## llama.cpp

We use Qualcomm's fork of llama.cpp. A specific [version](https://github.com/qualcomm/llama.cpp/tree/1f5accb8d0056e6099cd5b772b1cb787dd590a13) has been noted, since Qualcomm keeps updating the repo. A newer repo is expected to have better support for the NPU.

Then follow the [instructions](https://github.com/OswaldHe/cs259-hexagon-llama-cpp/tree/68c91822e2099fd9426d1c4e539b6843f722a558) to run tests on different models.

## ExecuTorch

We only discuss the key commands in this section, the remaining details can be found at the following resources

* [Qualcomm AI Engine Backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
* [example/model/llama](https://github.com/pytorch/executorch/tree/faa59036a9bd874761fb642b79ead5954408cee5/examples/models/llama)
* [XNNPACK](https://github.com/pytorch/executorch/tree/faa59036a9bd874761fb642b79ead5954408cee5/backends/xnnpack)

### Cross-Compiling the C++ Runners
Before executing on the edge device, we must cross-compile the ExecuTorch C++ runners for the target architecture (ARM64) using CMake. This requires the Android NDK and the Qualcomm AI Engine Direct SDK.


Prerequisites
* Android NDK: Required for cross-compiling the C++ code into Android-compatible binaries.

* Qualcomm AI Engine Direct SDK (QNN SDK): Required to link the Hexagon DSP libraries for the NPU runner.

#### 1. Compile the CPU Baseline Runner (llama_main)
This step builds the standard ExecuTorch runner used for the Portable and XNNPACK CPU backends.

```Bash
# env
export ANDROID_NDK=<path-to-android-ndk>

# Build executorch and associated libraries for android
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -Bcmake-out-android .

cmake --build cmake-out-android -j16 --target install --config Release

# Build llama runner for android
cmake  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DSUPPORT_REGEX_LOOKAHEAD=ON \
    -Bcmake-out-android/examples/models/llama \
    examples/models/llama

cmake --build cmake-out-android/examples/models/llama -j16 --config Release

# runner can be found at cmake-out-android/examples/models/llama/llama_main
ls cmake-out-android/examples/models/llama/
```

#### 2. Compile the NPU Runner (qnn_llama_runner)
This step builds the highly optimized runner that delegates operations to the Hexagon NPU. You must explicitly enable the QNN build flag and provide the path to your QNN SDK.

```Bash
# env 
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..

# build
cd $EXECUTORCH_ROOT
# android target
./backends/qualcomm/scripts/build.sh

# assuming the AOT component is built.
cd $EXECUTORCH_ROOT/build-x86
cmake ../examples/qualcomm \
  -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DPYTHON_EXECUTABLE=python3 \
  -Bexamples/qualcomm

cmake --build examples/qualcomm -j$(nproc)

# qnn_llama_runner can be found under build-android/examples/qualcomm/oss_scripts/llama/
ls build-android/examples/qualcomm/oss_scripts/llama/
```

Note: After compilation, remember to push these binaries to your Snapdragon device via adb push.

### Host Machine (Ahead-of-Time Compilation)

The ExecuTorch pipeline relies on AOT compilation to perform static memory planning and operator lowering, ensuring zero runtime compilation overhead on the edge device.

#### QNN

We use the ExecuTorch export utility to trace the PyTorch eager model into EXIR, apply 16a4w Post-Training Quantization (PTQ) for the NPU, and serialize it into a Flatbuffer (`.pte`) file.

```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
-b build-android \
-s ${SERIAL_NUM} \
-m ${SOC_MODEL} \
--checkpoint consolidated.00.pth \
--params params.json \
--tokenizer_model tokenizer.model \
--decoder_model llama3_2-1b_instruct \
--model_mode hybrid \
--prefill_ar_len 128 \
--max_seq_len 1024 \
--prompt "I would like to learn python, could you teach me with a simple example?" \
--tasks wikitext \
--limit 1
```

*Note: The `--max_seq_len 1024` flag is critical here as it forces the AOT compiler to statically allocate the exact memory footprint required for the KV-cache.*

#### Portable

```bash
python -m executorch.examples.models.llama.export_llama \
    --model_name llama3_1b \
    --seq_len 1024 \
    --output_dir ./build_portable
```

#### XNNPACK

```bash
# QAT+LoRA
# Set these paths to point to the exported files
LLAMA_QUANTIZED_CHECKPOINT=path/to/qlora/consolidated.00.pth.pth
LLAMA_PARAMS=path/to/qlora/params.json

python -m extension.llm.export.export_llm \
    --config examples/models/llama/config/llama_xnnpack_qat.yaml \
    +base.model_class="llama3_2" \
    +base.checkpoint="${LLAMA_QUANTIZED_CHECKPOINT:?}" \
    +base.params="${LLAMA_PARAMS:?}" 
```


### Edge Device (On-Device Inference)

Once the `.pte` binaries and the `tokenizer.model` are pushed to the Snapdragon device via `adb`, we execute them using the C++ reference runners.

#### Option A: Hexagon NPU Execution (QNN Backend)

Run the model on the NPU.

**Note:** You must explicitly pass `--decoder_model_version "llama3"`. Without this, the QNN runner defaults to Llama-2's EOS token ID, causing the model to ignore `<|eot_id|>` and fall into a degenerate repetition loop (hallucination) until the sequence length is exhausted.

```bash
# at the device or via adb
./qnn_llama_runner \
    --model_path <qnn_model>.pte \
    --tokenizer_path tokenizer.model \
    --prompt "If property values have gone up over the last twenty years, what do we know will happen in the next twenty years?" \
    --seq_len 1024 \
    --eval_mode 0 \
    --shared_buffer \
    --decoder_model_version "llama3"

```

The output can be found at `./outputs.txt`

### Option B: CPU Execution (Portable Backend)

To run the model the Oryon CPU, use the standard `llama_main` runner.

```bash
# at the device or via adb
./llama_main \
    --model_path <cpu_model>.pte \
    --tokenizer_path llama3/tokenizer.model \
    --prompt "Once upon a time" \
    --seq_len 1024"

```

## Evaluation (TruthfulQA)

For factual accuracy evaluation, we utilize the BLEURT semantic metric rather than traditional n-gram matching (e.g., BLEU) to properly evaluate the cognitive integrity of the 16a4w quantized model.

Shell scripts to run `llama_main` and `qnn_llama_runner` are provided (`run-cpu.sh` and `run-executorch.sh`), specified corresponding script in `truthful_qa_eval.py` for the experiments.  



