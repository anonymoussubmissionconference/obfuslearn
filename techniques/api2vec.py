import time
from pathlib import Path

import r2pipe
import re
from collections import Counter
import networkx as nx
import json
from gensim.models import Word2Vec
import numpy as np

import os


def api_calls_to_vector(api_list, api_vocab):
    vec = [0] * len(api_vocab)
    for api in api_list:
        if api in api_vocab:
            vec[api_vocab[api]] = 1
    return vec


def api_calls_to_freq_vector(api_list, api_vocab):
    count = Counter(api_list)
    vec = [0] * len(api_vocab)
    for api, freq in count.items():
        if api in api_vocab:
            vec[api_vocab[api]] = freq
    return vec




def extract_api_calls(binary_path):
    r2 = r2pipe.open(binary_path)
    r2.cmd('aaa')  # Perform full analysis

    funcs = json.loads(r2.cmd('aflj'))  # List functions in JSON
    api_calls = {}

    for func in funcs:
        name = func['name']
        offset = func['offset']
        disasm_json = json.loads(r2.cmd(f'pdfj @{offset}'))

        calls = []
        for op in disasm_json.get('ops', []):
            if op.get('type') == 'call':
                call_target = op.get('disasm', '')
                calls.append(call_target)

        api_calls[name] = calls

    r2.quit()
    return api_calls



def process_directory_to_images(root_folder: str, output_folder: str, vector_size: int, log_folder: str = "timing_logs"):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue

        start_time = time.time()

        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            if not os.path.isfile(file_path):
                continue

            float_array = word2vector(file_path, vector_size)
            if float_array is not None:
                relative_output_path = os.path.join(output_folder, subfolder_name, file + ".npy")
                os.makedirs(os.path.dirname(relative_output_path), exist_ok=True)
                np.save(relative_output_path, float_array)

        elapsed_time = time.time() - start_time

        # Log timing to a file
        with open(os.path.join(log_folder, f"{subfolder_name}_time.txt"), "w") as log_file:
            log_file.write(f"Processed in {elapsed_time:.2f} seconds\n")
        print(f"Finished processing {subfolder_name} in {elapsed_time:.2f} seconds.")


def word2vector(file_path, vector_size=128):
    try:
        call_dict = extract_api_calls(file_path)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return np.zeros((224, 224))
    sequences = []

    for start_func, calls in call_dict.items():
        seq = [start_func] + [line.split()[-1] for line in calls if line.startswith("call")]
        if len(seq) > 1:
            sequences.append(seq)
    if len(sequences)>0:
        # Step 2: Train Word2Vec model
        model = Word2Vec(sentences=sequences, vector_size=vector_size, window=5, min_count=1, sg=1)

        # Step 3: Embed each call sequence with start function
        embedded_sequences = {
            func: embed_sequence([func] + [line.split()[-1] for line in calls], model)
            for func, calls in call_dict.items()
        }
        matrix = np.stack(list(embedded_sequences.values()))
        return matrix
    else:
        return np.zeros((224, 224))
def embed_sequence(seq, model):
    vectors = [model.wv[token] for token in seq if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)




root_folder=r"../data/malimgbinary"
output_folder=r'../features/apis/malfiner-apisequence-512'
log_path=last_part = os.path.join('timing_logs', output_folder.split('/')[-1])
process_directory_to_images(root_folder, output_folder, vector_size=512, log_folder=log_path)
