import time

#import r2pipe
import networkx as nx
import numpy as np
import os
import angr
from node2vec import Node2Vec
import pyhidra



def flow_graph_angr(binary_path) -> nx.DiGraph:
        project = angr.Project(binary_path, auto_load_libs=False)
        cfg = project.analyses.CFGFast()
        graph = nx.DiGraph()

        # nodes
        for node in cfg.graph.nodes():
            graph.add_node(node.addr)

        # edges
        for node in cfg.graph.nodes():
            for successor in cfg.graph.successors(node):
                graph.add_edge(node.addr, successor.addr)

        return graph


def flow_graph_ghidra(binary_path) -> nx.DiGraph:
    project_path = r"../project"

    try:
        with pyhidra.open_program(binary_path, project_location=project_path) as api:
            print(binary_path)
            import ghidra
            graph = nx.DiGraph()
            program = api.getCurrentProgram()
            block_model = ghidra.program.model.block.SimpleBlockModel(program)


            blocks = block_model.getCodeBlocks(api.getMonitor())
            for block in blocks:
                start_addr = int(block.getFirstStartAddress().getOffset())
                graph.add_node(start_addr)

            for block in blocks:
                src = int(block.getFirstStartAddress().getOffset())
                dests = block.getDestinations(api.getMonitor())
                while dests.hasNext():
                    dest = dests.next().getDestinationBlock()
                    dst = int(dest.getFirstStartAddress().getOffset())
                    graph.add_edge(src, dst, weight=1.0)
    except Exception as e:
        print(f"[ERROR] Failed to import {binary_path}: {e}")
        return None
    return graph


def graph_embedding_node2vec(G, dimensions):
    if G is None or not isinstance(G, (nx.Graph, nx.DiGraph)) or len(G.nodes) < 2:
        return np.zeros(dimensions)

    try:
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=20, num_walks=100, workers=1, quiet=True)

        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Aggregate node embeddings
        embeddings = [model.wv[str(node)] for node in G.nodes() if str(node) in model.wv]
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(dimensions)

    except Exception as e:
        print(f"[ERROR] Node2Vec embedding failed: {e}")
        return np.zeros(dimensions)


def process_directory_to_images(root_folder: str, output_folder: str, vector_size: int, log_folder: str = "timing_logs"):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue

        start_time = time.time()
        file_count = 0
        pyhidra.start(True)
        for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if not os.path.isfile(file_path):
                    continue

                graph = flow_graph_ghidra(file_path)
                float_array = graph_embedding_node2vec(graph, dimensions=vector_size)
                print(float_array.shape)
                #print(file_path)
                relative_output_path = os.path.join(output_folder, subfolder_name, file + ".npy")
                os.makedirs(os.path.dirname(relative_output_path), exist_ok=True)
                np.save(relative_output_path, float_array)
                file_count += 1

        elapsed_time = time.time() - start_time
        log_path = os.path.join(log_folder, f"{subfolder_name}_time.txt")
        with open(log_path, "w") as log_file:
            log_file.write(f"Processed {file_count} files in {elapsed_time:.2f} seconds\n")

        print(f"[INFO] Finished processing {subfolder_name}: {file_count} files, {elapsed_time:.2f} seconds")
    return output_folder


root_folder=r"../data/malimg"
output_folder=r'../features/graphs/malimg-cfg-512'
log_path=last_part = os.path.join('timing_logs', output_folder.split('/')[-1])
process_directory_to_images(root_folder, output_folder, vector_size=512, log_folder=log_path)

