import csv
import os

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class PatriPrompts(Dataset):
    def __init__(self, path):
        super().__init__()
        self.prompts = []
        with os.fdopen(os.open(
                path, os.O_RDONLY), "r", encoding='utf8') as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                prompt = line[0]
                self.prompts.append(prompt)

    def __getitem__(self, index):
        return self.prompts[index]

    def __len__(self):
        return len(self.prompts)


def make_dataloader(path, batch_size):
    dataset = PatriPrompts(path)

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=4,
    )
    return dataloader


if __name__ == '__main__':
    path = '/workspaces/kapok/flux/prompts/PartiPrompts.tsv'
    dataset = PatriPrompts(path)

    for p in dataset:
        print(p)