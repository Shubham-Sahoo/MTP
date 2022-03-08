#!/usr/bin/env bash

python generate_vocab_subwords.py ../examples/conformer/libris_trans1.tsv --config ../examples/conformer/config_lib.yml --output_file ../examples/conformer/libris_speech.subwords

python create_tfrecords.py --config ../examples/conformer/config_lib.yml --tfrecords_dir ../examples/conformer/tfrec/ --subwords ../examples/conformer/libris_speech.subwords --mode train ../examples/conformer/libris_trans1.tsv

python generate_metadata.py --stage train --config ../examples/conformer/config_lib.yml --metadata ../examples/conformer/metadata_libris.txt --subwords ../examples/conformer/libris_speech.subwords ../examples/conformer/libris_trans1.tsv

cd ..
cd examples/conformer/

python train.py --config config_lib.yml --metadata metadata_libris.txt --tfrecords --spx 1 --bs 1
