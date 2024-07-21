# Create the train-test splits for the SCDB/Songer tasks

import json
from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--sc_file', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    assert args.test_size + args.val_size < 1

    train_size = 1 - args.test_size - args.val_size

    # Load the data file
    ids = []
    print("Loading the Supreme Court opinions...")
    with open(args.sc_file, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            data = json.loads(line)
            ids.append(data['caselaw']['id'])

    # Assert that all ids are unique
    assert len(ids) == len(set(ids))

    # Randomly shuffle the ids
    import random
    random.seed(args.seed)
    random.shuffle(ids)

    # Split the ids
    train_ids = ids[:int(train_size * len(ids))]
    val_ids = ids[int(train_size * len(ids)):int((train_size + args.val_size) * len(ids))]
    test_ids = ids[int((train_size + args.val_size) * len(ids)):]
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(ids)

    # Save the dataset
    with open(args.save_file, "w") as jsonl_file:
        json.dump({'train': train_ids, 'val': val_ids, 'test': test_ids}, jsonl_file)