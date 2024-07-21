# Download the court opinions from the Case Law Access Project corresponding to the Supreme Court cases in the SCDB dataset.
# The SCDB dataset contains the citations to the cases, which can be used to match the cases in the Case Law Access Project.
# The resulting dataset is saved as a jsonl file, where each line is a dictionary with the keys:
#   - 'caselaw': full case law data from the Case Law Access Project API
#   - 'sc_db': metadata from the SCDB dataset

import json
import requests
import pandas as pd
from tqdm import tqdm

caselaw_api_token = ''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Download and extract the SCDB data file from...
    # http://scdb.wustl.edu/data.php > 
    #                       Case Centered Data > 
    #                               Cases Organized by Supreme Court Citation > 
    #                                                       SCDB_2023_01_caseCentered_Citation.csv.zip
    parser.add_argument('--scdb_file', type=str, default='sc_labels.csv')
    # Download the Caselaw Project metadata from the caselaw us bulk export 
    # > https://case.law/download/bulk_exports/20200604/by_jurisdiction/case_metadata/us/
    parser.add_argument('--metadata_file', type=str, default='caselaw_us_metadata.jsonl')
    parser.add_argument('--save_file', type=str)  # will be saved as a .jsonl
    args = parser.parse_args()

    # Load the meta data corresponding to the US jurisdiction
    metadata = []
    with open(args.metadata_file,  'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            data = json.loads(line)
            metadata.append(data)

    # Load the Supreme Court labels
    sc_labels = pd.read_csv(args.scdb_file, encoding='Windows-1252')

    # ------------------------------------------
    # Match the SC cases with labels to caselaw ids
    def unique_citations(sc_labels, style):
        """ Returns cases for which the citation is unique (e.g., does not appear in multiple rows of the SC data) """
        return {v: k for k, v in sc_labels[style][sc_labels.groupby(style)[style].transform(len) == 1].items()}

    def get_all_citations(metadata, citations):
        """ Try to find citation matches in the caswlaw data """
        cit_sc_metadata = {}
        for i, row in tqdm(enumerate(metadata)):
            for cit in row['citations']:
                citation = cit['cite']
                if citation in citations.keys():
                    cit_sc_metadata[citations[citation]] = i
        return cit_sc_metadata

    cit_sc = get_all_citations(metadata, unique_citations(sc_labels, 'usCite'))
    cit_sct = get_all_citations(metadata, unique_citations(sc_labels, 'sctCite'))
    cit_led = get_all_citations(metadata, unique_citations(sc_labels, 'ledCite'))
    cit_lexis = get_all_citations(metadata, unique_citations(sc_labels, 'lexisCite'))

    # Merge, giving priority to the lexisCite, etc
    caselaw_row_sc = {}
    for i in range(len(sc_labels)):
        if i in cit_lexis:
            caselaw_row_sc[cit_lexis[i]] = i
        elif i in cit_led:
            caselaw_row_sc[cit_led[i]] = i
        elif i in cit_sct:
            caselaw_row_sc[cit_sct[i]] = i
        elif i in cit_sc:
            caselaw_row_sc[cit_sc[i]] = i

    # ------------------------------------------
    # Download and join them up
    dataset = []
    for metadata_id, sc_id in tqdm(caselaw_row_sc.items()):
        caselaw_ip_call = requests.get(
            'https://api.case.law/v1/cases/' + str(metadata[metadata_id]['id']) + '/?full_case=true',
            headers={'Authorization': f'Token {caselaw_api_token}'})

        caselaw = json.loads(caselaw_ip_call.text)
        sc_db = sc_labels.iloc[sc_id].to_dict()
        data_row = {'caselaw': caselaw, 'sc_db': sc_db}
        dataset.append(data_row)

    # Save as a jsonl file
    with open(args.save_file, "w") as jsonl_file:
        for item in dataset:
            jsonl_file.write(json.dumps(item) + "\n")

    print('Saved to', args.save_file)