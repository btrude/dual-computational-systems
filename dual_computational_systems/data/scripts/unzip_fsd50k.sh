#!/bin/bash

zip -s 0 datasets/fsd50k_zipped/FSD50K.dev_audio.zip --out unsplit.zip
zip -s 0 datasets/fsd50k_zipped/FSD50K.eval_audio.zip --out unsplit_eval.zip
mkdir -p datasets/fsd50k
unzip unsplit.zip -d datasets/fsd50k
unzip unsplit_eval.zip -d datasets/fsd50k
mv datasets/fsd50k/FSD50K.dev_audio datasets/fsd50k/train
mv datasets/fsd50k/FSD50K.eval_audio datasets/fsd50k/eval
rm unsplit.zip
rm unsplit_eval.zip
unzip datasets/fsd50k_zipped/FSD50K.ground_truth.zip -d datasets/fsd50k
mv datasets/fsd50k/FSD50K.ground_truth datasets/fsd50k/ground_truth
