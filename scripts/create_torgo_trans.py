# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import argparse
import librosa
from tqdm.auto import tqdm
import unicodedata

from tensorflow_asr.utils.file_util import preprocess_paths

parser = argparse.ArgumentParser(prog="Setup LibriSpeech Transcripts")

parser.add_argument("--dir", "-d", type=str, default=None, help="Directory of dataset")

# parser.add_argument("--dir_head", "-dh", type=str, default=None, help="Directory of head mic dataset")

# parser.add_argument("--dir_p", "-dp", type=str, default=None, help="Directory of prompts for the dataset")

parser.add_argument("output", type=str, default=None, help="The output .tsv transcript file path")

args = parser.parse_args()

assert args.dir and args.output

args.dir = preprocess_paths(args.dir, isdir=True)
args.output = preprocess_paths(args.output)

transcripts = []

text_files = glob.glob(os.path.join(args.dir, "prompts", "**", "*.txt"), recursive=True)

audio_files_arr = glob.glob(os.path.join(args.dir,"wav_arrayMic"), recursive=True)

audio_files_head = glob.glob(os.path.join(args.dir,"wav_headMic"), recursive=True)

align_file = glob.glob(os.path.join(args.dir,"alignment.txt"), recursive=True)[0]
# print(text_files)




for i in tqdm(range(len(text_files)), desc="[Loading]"):
    text_file = text_files[i]
    current_dir_text = os.path.dirname(text_file)
    file_name = (text_file.split("/")[-1]).split(".txt")
    # print(file_name)
    current_dir_audio_a = audio_files_arr[0]
    current_dir_audio_h = audio_files_head[0]

    offset = 0
    with open(align_file,"r", encoding="utf-8") as aln:
        for line_n, line in enumerate(aln):

            if line_n == (int(file_name[0])+1):
                # print((line.split()))
                offset = float(line.split()[1])/16000

    # print(current_dir_audio)

    with open(text_file, "r", encoding="utf-8") as txt:
        lines = (txt.read()).split('\n')[0]
        # line = line.split(" ", maxsplit=1)
        audio_file_arr = os.path.join(current_dir_audio_a, file_name[0] + ".wav")
        audio_file_head = os.path.join(current_dir_audio_h, file_name[0] + ".wav")

        
        if(offset>0):
            y2, sr2 = librosa.load(audio_file_head, sr=None, offset=offset)
            y1, sr1 = librosa.load(audio_file_arr, sr=None)
            duration1 = librosa.get_duration(y1, sr1)
            duration2 = librosa.get_duration(y2, sr2)
            
        else:
            y1, sr1 = librosa.load(audio_file_arr, sr=None, offset=offset)
            y2, sr2 = librosa.load(audio_file_head, sr=None)
            duration1 = librosa.get_duration(y1, sr1)
            duration2 = librosa.get_duration(y2, sr2)
        

        text = unicodedata.normalize("NFC", text_file.lower())
        transcripts.append(f"{audio_file_arr}\t{duration1}\t{audio_file_head}\t{duration2}\t{offset}\t{lines}\n")

with open(args.output, "w", encoding="utf-8") as out:
    out.write("PATH1\tDURATION1\tPATH2\tDURATION2\tOFFSET\tTRANSCRIPT\n")
    for line in tqdm(transcripts, desc="[Writing]"):
        out.write(line)
