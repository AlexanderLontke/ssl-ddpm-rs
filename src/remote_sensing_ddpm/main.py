import json
import torch
from remote_sensing_ddpm.evaluation.baselines.ddpm_cd import (
    create_model,
    create_CD_model,
    create_classification_model,
)


class NoneDict(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


if __name__ == "__main__":
    with open(
        "/Users/alexanderlontke/Documents/Uni/St. Gallen/SS_23/master_thesis/remote-sensing-ddpm/config/baselines/ddpm-cd/classification.json",
        "rb",
    ) as config_file:
        opt = NoneDict(json.load(config_file))
    model = create_classification_model(opt)
    print(model.print_network())

    model.classification_net(torch.rand((256, 256, 3)))
