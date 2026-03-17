#!/bin/sh

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

model="kv_llama3_qnn_64.pte"
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
  export ADSP_LIBRARY_PATH=. && \
  ./qnn_llama_runner \
    --model_path $model \
    --tokenizer_path $tokenizer \
    --prompt \"$prompt\" \
    --seq_len $seq_len \
    --decoder_model_version llama3 \
    --eval_mode 0 \
    --temperature 0.4 \
    --shared_buffer > /dev/null 2>&1 && \
  cat outputs.txt \
"

