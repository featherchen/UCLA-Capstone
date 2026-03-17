#!/bin/sh

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

# specify the model
# model="kv_llama3_cpu_1024.pte"
model="kv_llama3_x_1024.pte"

[ "$M" != "" ] && model="$M"

tokenizer="llama3/tokenizer.model"
[ "$T" != "" ] && tokenizer="$T"

prompt=""
seq_len=25

while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) prompt="$2"; shift 2;;
    --seq_len) seq_len="$2"; shift 2;;
    *) shift;;
  esac
done

adb $adbserial shell " \
  cd $DEVICE_DIR && \
  export LD_LIBRARY_PATH=. && \
  ./llama_main \
    --model_path $model \
    --tokenizer_path $tokenizer \
    --prompt \"$prompt\" \
    --temperature 0.4 \
    --seq_len $seq_len
"