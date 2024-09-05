import os
import re

import json
import zipfile
import urllib.request
import concurrent.futures

import pyreadstat
import pandas as pd
from tqdm import tqdm


def get_scdb(tmp_dir='tmp/'):
    # make the tmp directory if it doesn't exist
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    scdb_file = 'SCDB_2023_01_caseCentered_Citation.csv'
    scdb_file_location = f'http://scdb.wustl.edu/_brickFiles/2023_01/{scdb_file}.zip'

    # download the file
    zip_file = f'{tmp_dir}{scdb_file}.zip'
    urllib.request.urlretrieve(scdb_file_location, zip_file)

    # unzip the file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    scdb = pd.read_csv(f'{tmp_dir}{scdb_file}', encoding='Windows-1252')

    # delete the files
    os.remove(zip_file)
    os.remove(f'{tmp_dir}{scdb_file}')

    return scdb


def get_songer(tmp_dir='tmp/'):
    # make the tmp directory if it doesn't exist
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # download the file
    songer_data_url = 'http://nebula.wsimg.com/1530fa26a7f31fcdf01a775a7d196b2a?AccessKeyId=96203964AD4677DE3481&disposition=0&alloworigin=1'
    zip_file = f'{tmp_dir}songer.zip'
    urllib.request.urlretrieve(songer_data_url, zip_file)

    # extract the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    songer_file = f'{tmp_dir}cta96_stata.dta'
    songer, _ = pyreadstat.read_dta(songer_file)

    # delete the files
    os.remove(zip_file)
    os.remove(songer_file)

    return songer


def browser_download(url):
    assert url.endswith('.json')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        out = response.read()
    return json.loads(out)


def get_volumes_in_url(url):
    # works for https://static.case.law/{reporter}/
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        out = response.read()
    volumes = re.findall(r'>[0-9]+<', str(out))
    volumes = [int(v[1:-1]) for v in volumes]
    return volumes


def cit_to_volume(citation):
    if type(citation) != float:
        return int(citation.split(' ')[0])
    else:
        return None
    

def get_files_from_metadata(scdb_citations, caselaw_dir, verbose=False, skip={}):
    """
    scdb_citations: pd.DataFrame with columns ['citation', 'docket']
    caselaw_dir: str, the URL to the caselaw project directory
    verbose: bool, print out the citations that are not found
    skip: set, set of indices to skip
    return: dict of scdb index -> file URL on the caselaw project
    """
    volumes = get_volumes_in_url(caselaw_dir)

    case_volumes = {i: cit_to_volume(row['citation']) for i, row in scdb_citations.iterrows()}
    case_volumes = {k: v for k, v in case_volumes.items() if v in volumes and k not in skip}
    case_volumes = {k: v for k, v in sorted(case_volumes.items(), key=lambda item: item[1])}

    current_volume = None
    matched_files = {}
    for i, volume in tqdm(case_volumes.items()):
        if current_volume != volume:
            metadata = browser_download(f"{caselaw_dir}{volume}/CasesMetadata.json")

            # metadata (list of dicts) into a dict of citation -> [(docket number, file name)]
            citations_dict = {}
            for row in metadata:
                for cit in row['citations']:
                    citation = cit['cite'].strip()
                    if citation not in citations_dict:
                        citations_dict[citation] = []

                    docket_number = row['docket_number']
                    docket_number = docket_number.replace('–', '-')
                    docket_number = re.sub(r'[^\d -]', '', docket_number).strip()

                    citations_dict[citation].append((docket_number, row['file_name']))
            
            if verbose:
                print('Volume citations:', citations_dict.keys())

            current_volume = volume

        # find the case in the metadata
        citation = scdb_citations.iloc[i]['citation'].strip()
        if citation not in citations_dict:
            if verbose:
                print('Missing citation:', citation)
            continue

        # check if any docket numbers match
        docket = scdb_citations.iloc[i]['docket']
        if type(docket) == float:
            continue
        docket = docket.strip().replace('–', '-')
        docket = re.sub(r'[^\d -]', '', docket).strip()
        for docket_number, file_name in citations_dict[citation]:
            if docket in docket_number:
                matched_files[i] = (f"{caselaw_dir}{volume}/cases/{file_name}.json")
                break
        
        # if no docket number matches, print the citation and the docket numbers
        if verbose and (i not in matched_files):
            print(f"Row {i}, {citation}: {docket} not found in {citations_dict[citation]}")

    return matched_files


def download_in_parallel(matched_files):
    files = {}
    print('Downloading files...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(browser_download, file_url): i for i, file_url in matched_files.items()}
        # Collect the results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            i = futures[future]
            files[i] = future.result()

    # sort by the index
    files = {k: v for k, v in sorted(files.items(), key=lambda item: item[0])}
    return files


def match_files_with_db(files, database, database_key):
    dataset = []
    for i, caselaw_case in files.items():
        scdb_data = database.iloc[i].to_dict()
        data_row = {'caselaw': caselaw_case, database_key: scdb_data}
        dataset.append(data_row)
    return dataset


def save_jsonl(save_file, dataset):
    with open(save_file, "w") as jsonl_file:
        for item in dataset:
            jsonl_file.write(json.dumps(item) + "\n")
    print('Saved to', save_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sc', action='store_true')
    parser.add_argument('--songer', action='store_true')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--tmp_dir', type=str, default='tmp/')
    args = parser.parse_args()

    if len(args.save_dir) > 0 and (not os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)
    
    # Download the Supreme Court Database
    if args.sc:
        scdb = get_scdb(args.tmp_dir)

        # Match the cases in the SCDB database with those in the Caselaw Project
        sct_files = get_files_from_metadata(
            scdb[['sctCite', 'docket']].rename(columns={'sctCite': 'citation'}),
            caselaw_dir='https://static.case.law/s-ct/', 
        )
        led_files = get_files_from_metadata(
            scdb[['ledCite', 'docket']].rename(columns={'ledCite': 'citation'}),
            caselaw_dir='https://static.case.law/l-ed-2d/',
            skip=set(sct_files.keys())
        )
        us_files = get_files_from_metadata(
            scdb[['usCite', 'docket']].rename(columns={'usCite': 'citation'}),
            caselaw_dir='https://static.case.law/us/',
            skip=set(sct_files.keys()) | set(sct_files.keys())
        )

        matched_files = {**sct_files, **led_files, **us_files}
        print(f"Matched {len(matched_files)/len(scdb)*100:.1f}% of SCDB cases ({len(matched_files)}/{len(scdb)})")

        files = download_in_parallel(matched_files)
        joint_database = match_files_with_db(files, scdb, 'sc_db')
        save_jsonl(args.save_dir + 'caselaw_sc.jsonl', joint_database)

    if args.songer:
        songer = get_songer(args.tmp_dir)

        f_docnum = lambda x: str(int(x[3:])) if x.isdigit() and len(x) == 8 else float('nan')
        f_citation = lambda x: f"{int(x['vol'])} F.2d {int(x['beginpg'])}"

        songer['docket'] = songer['docnum'].apply(f_docnum)
        songer['citation'] = songer.apply(f_citation, axis=1)

        matched_files = get_files_from_metadata(
                songer[['citation', 'docket']],
                caselaw_dir="https://static.case.law/f2d/",
        )
        print(f"Matched {len(matched_files)/len(songer)*100:.1f}% of Songer cases ({len(matched_files)}/{len(songer)})")

        files = download_in_parallel(matched_files)
        joint_database = match_files_with_db(files, songer, 'songer')
        save_jsonl(args.save_dir + 'caselaw_songer.jsonl', joint_database)

    print('Done')
