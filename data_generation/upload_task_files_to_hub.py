import os
from huggingface_hub import HfApi


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True)
    parser.add_argument('--repo_id', type=str, default='ricdomolm/lawma-task-files')
    args = parser.parse_args()

    base_dir = 'tasks/'
    assert os.path.exists(base_dir), f"Base directory {base_dir} doesn't exist"

    # compress the base directory
    print("Compressing the base directory...")
    filename = 'tasks.tar.gz'
    os.system(f"tar -czf {filename} {base_dir}")

    api = HfApi()
    print("Uploading the compressed file to the hub...")
    api.upload_file(
        path_or_fileobj=filename,
        token=args.token,
        path_in_repo=filename,
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    print(f"Uploaded {base_dir} to {args.repo_id}")

    # remove the compressed file
    os.remove(f"{filename}")
    
    print("Done.")