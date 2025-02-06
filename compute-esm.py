import torch
import esm
import numpy as np
import os
from Bio import SeqIO
import gc

def main(input_path, output_dir):
    # Load ESM-2 model
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    # model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device(f"cuda:0" if (torch.cuda.is_available()) else "cpu")
    model.to(device)

    i = 0
    files_list = os.listdir(f'{input_path}')
    for filename in files_list:
        i = i + 1
        name, ext = os.path.splitext(filename)

        print(f"Processing {filename} ... {i}/{len(files_list)}")
        with open(f'{input_path}/{filename}', 'r') as f:
            sequence = f.read()
            treshold = 1022
            vectors = []
            while len(sequence) > 0:
                sequence1 = sequence[:treshold]
                sequence = sequence[treshold:]
                data = [
                    (name, sequence1)
                ]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                # Extract per-residue representations
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[36], return_contacts=True)
                token_representations = results["representations"][36]
                vectors1 = token_representations.detach().cpu().numpy()[0][1:-1]
                if len(vectors) > 0:
                    vectors = np.concatenate((vectors, vectors1))
                else:
                    vectors = vectors1

            if not os.path.exists(f"{output_dir}/"): 
                os.makedirs(f"{output_dir}/") 
            np.save(f'{output_dir}/{name}.npy', vectors)    
#            del results, token_representations, batch_tokens
        if i % 500 == 0:
            if 'cuda' in device.type:
                torch.cuda.empty_cache()   
            gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', metavar='path', required=True,
                        help='the path to the fasta files')
    parser.add_argument('--output', metavar='path', required=True,
                        help='output path')
    args = parser.parse_args()
    main(args.input, args.output)
