#!/bin/bash

# --- basic conventions ---
export EXECUTORCH_ROOT=$(pwd)/executorch/
export QNN_SDK_ROOT=$(pwd)/qairt/2.37.0.250724/
export ANDROID_NDK_ROOT=$(pwd)/android-ndk-r27d/

# --- environment variables---
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..

# export ADB_SERVER_SOCKET=tcp:host.docker.internal:5037
export DEVICE_DIR=/data/local/tmp/executorch_qualcomm_tutorial/
source /opt/venv/bin/activate
# --- status ---
echo "ExecuTorch setup completed"
echo "QNN_SDK_ROOT: $QNN_SDK_ROOT"
echo "ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"

