import json
import random

class TaskLoader:
    def __init__(self, opinions, task_file, split='train', max_samples=None, seed=42):
        """
        Takes in a task `.json` file ...
        and returns an iterator over 'opinion', 'instruction', 'question', 'choices', 'answer'
        """
        self.opinions = opinions

        with open(task_file, 'r') as f:
            task = json.load(f)

        self.task = task['task']
        self.question = self.task['question']
        self.instruction = self.task['instruction']
        self.fill_in = self.task.get('fill_in', [])  # list of keys
        self.examples = task['examples'][split]

        if 'answer_choices' in self.task and self.task['answer_choices'] is not None:
            self.choices = self.task['answer_choices']
        else:
            self.choices = []

        # Shuffle the dataset
        random.seed(seed)
        random.shuffle(self.examples)

        if max_samples is not None:
            self.examples = self.examples[:max_samples]

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        match_index = lambda c, c_dict: [i for i, k in enumerate(c_dict) if int(k) == c][0]

        for example in self.examples:
            choices = self.get_choices(example)
            choices_list = list(choices.values()) if choices else []

            answer = example['target']
            if type(answer) != list:
                answer = [answer]
            if choices:
                answer = [match_index(a, choices) for a in answer]

            yield {
                'opinion': self.opinions[str(example['input'])],
                'instruction': self.get_instruction(example),
                'question': self.get_question(example),
                'choices': choices_list,
                'answer': answer,
            }

    def fill(self, string, example):
        fills = {}
        for key in self.fill_in:
            if '{' + key + '}' in string:
                fills[key] = example[key]
        return string.format(**fills)
    
    def get_question(self, example):
        question = self.question
        if 'key' in example and type(self.question) == dict:
            question = self.question[example['key']]
        return self.fill(question, example)
    
    def get_instruction(self, example):
        instruction = self.instruction
        if 'key' in example and type(self.instruction) == dict:
            instruction = self.instruction[example['key']]
        return self.fill(instruction, example)
    
    def get_choices(self, example):
        if not self.choices:
            return []
        if 'choices' in example:
            return {c: self.choices[str(c)] for c in example['choices']}
        if 'key' in example and type(list(self.choices.values())[0]) == dict:
            return self.choices[example['key']]
        return self.choices
    

if __name__ == "__main__":
    import os 
    import datasets
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='../datasets/')
    parser.add_argument('--task_dir', type=str, default='../tasks/')
    parser.add_argument('--push_to_hub', action='store_true')
    
    args = parser.parse_args()
    task_dir = args.task_dir
    save_dir = args.save_dir

    print('Loading opinions...')
    files = sorted(os.listdir(task_dir))
    opinion_files = [f for f in files if f.endswith('opinions.json')]
    opinions = {}
    for opinion_file in opinion_files:
        with open(f"{task_dir}/{opinion_file}", 'r') as f:
            opinions.update(json.load(f))

    print('Saving datasets...')
    task_files = [f for f in files if f.endswith('.json') and f not in opinion_files]
    os.makedirs(save_dir, exist_ok=True)
    for task in task_files:
        task_name = task[:-5]
        os.makedirs(f"{save_dir}{task_name}", exist_ok=True)
        for split in ['train', 'val', 'test']:
            loader = TaskLoader(
                task_file=f"{task_dir}{task}",
                opinions=opinions,
                split=split,
            )

            dset = datasets.Dataset.from_list(list(loader))
            save_name = f"{save_dir}{task_name}/{split}-00000-of-00001.parquet"
            dset.to_parquet(save_name)

    if args.push_to_hub:  # you need to be authenticated to push to the hub
        import git
        print('Cloning the repository...')
        repo = git.Repo.clone_from("https://huggingface.co/datasets/ricdomolm/lawma-tasks", 'lawma-tasks') 
        print('Copying files...')
        os.system(f"cp -r {save_dir}* lawma-tasks/")
        print('Adding files...')
        repo.index.add('*')
        print('Committing...')
        repo.index.commit("Update")
        print('Pushing...')
        origin = repo.remote(name='origin')
        origin.push()
        os.system("rm -rf lawma-tasks")
