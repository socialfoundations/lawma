# Download the court opinions from the Case Law Access Project corresponding to the Court of Appeals cases in the Songer dataset.
# The Songer dataset contains the citations to the cases, which can be used to match the cases in the Case Law Access Project.
# The resulting dataset is saved as a jsonl file, where each line is a dictionary with the keys:
#   - 'caselaw': full case law data from the Case Law Access Project API
#   - 'songer': metadata from the Songer dataset

import json
import requests
import jsonlines
import pyreadstat
from tqdm import tqdm

caselaw_api_token = ''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # songer data obtained from http://www.songerproject.org/us-courts-of-appeals-databases.html
    parser.add_argument('--songer_file', type=str, default='cta96_stata.dta')
    # metadata for f2d obtained from https://case.law/download/bulk_exports/20200604/by_reporter/case_metadata/f2d/
    parser.add_argument('--metadata_file', type=str, default='f2d_metadata.dta')
    parser.add_argument('--save_file', type=str)  # will be saved as a .jsonl

    args = parser.parse_args()

    # Load the Songer data
    dataframe, meta = pyreadstat.read_dta(args.data_file)

    # Obtain citations from the Songer data
    match_data = {}
    seen_keys = set()
    for i in range(len(dataframe)):
        row = dataframe.iloc[i]
        songer_id = row['casenum']
        citation = f"{int(row['vol'])} F.2d {int(row['beginpg'])}"
        date = f"{row['year']}-{row['month']:02d}-{row['day']:02d}"
        key = f"{citation} :: {date}"
        if key not in seen_keys:
            seen_keys.add(key)
            match_data[key] = {'songer_id': songer_id}
        else:
            if key in match_data:
                del match_data[key]

    print("Kept {}% of the data".format(len(match_data) / len(dataframe) * 100))

    # Match citations with the Case Law Access Project metadata
    metadata_file = 'f2d_metadata.jsonl'
    n_found = 0
    n_duplicates = 0
    with jsonlines.open(args.metadata_file) as reader:
        for obj in reader:
            for citation in obj['citations']:
                if 'F.2d' in citation['cite']:
                    key = f"{citation['cite']} :: {obj['decision_date']}"
                    if key in match_data:
                        if 'caselaw_id' not in match_data[key]:
                            match_data[key]['caselaw_id'] = obj['id']
                            n_found += 1
                        else:  # delete duplicate
                            n_duplicates += 1
                            del match_data[key]

    print(f"Found {n_found} unique cases, and deleted {n_duplicates} duplicates")

    # Remove cases without caselaw_id
    match_data = {k: v for k, v in match_data.items() if 'caselaw_id' in v}
    print("Kept {}% of the data".format(len(match_data) / len(dataframe) * 100))


    # Individually download each of the cases
    dataset = []
    for match_data_ in tqdm(match_data.values()):
        # Extract the songer labels
        songer_row = dataframe[dataframe['casenum'] == match_data_['songer_id']]
        assert len(songer_row) == 1
        songer_row = songer_row.iloc[0].to_dict()

        # Download from the API
        caselaw_ip_call = requests.get(
            'https://api.case.law/v1/cases/' + str(match_data_['caselaw_id']) + '/?full_case=true',
            headers={'Authorization': f'Token {caselaw_api_token}'})
        caselaw = json.loads(caselaw_ip_call.text)

        # Put togeher
        data_row = {'caselaw': caselaw, 'songer': songer_row}
        dataset.append(data_row)

    # Save as a jsonl file
    with open(args.save_file, "w") as jsonl_file:
        for item in dataset:
            jsonl_file.write(json.dumps(item) + "\n")