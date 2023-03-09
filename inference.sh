#!/usr/bin/env bash

# do enhance(denoise)
CUDA_VISIBLE_DEVICES='5' python -m speech_enhance.tools.inference \
  -C config/inference.toml \
  -M /cjcode/ft_local/intersubnet/intersubnet.tar \
  -I /DNS_2021/eval/testclips \
  -O /enhance_data/dns4_testclips/inter_subnet


# Normalized to -6dB (optional)
sdir="/dockerdata/thujunchen/enhance_data/dns4_testclips/inter_subnet"
fdir="/dockerdata/thujunchen/enhance_data/dns4_testclips/inter_subnet_norm"

softfiles=$(find $sdir -name "*.wav")
for file in ${softfiles}
do
  length=${#sdir}+1
  file=${file:$length}
  f=$sdir/$file
  echo $f
  dstfile=$fdir/$file
  echo $dstfile
  sox $f -b16 $dstfile rate -v -b 99.7 16k norm -6
done


