import sys
import logging

from train import main as train_main
from audio_aug.augment import get_augment_schemes


if __name__ == '__main__':
    experiment_name = "spec"
    base_args = sys.argv[1:]  # Remove script name
    template = "audio_aug/specaugment_scheme.yml"
    runs = get_augment_schemes(gammas=[0.5, 0.75, 0.875, 1],
                               num_runs=1,
                               template_file=template,
                               name_prefix=experiment_name)

    for run_num, augmentor in enumerate(runs):
        run_args = base_args.copy()
        run_args.append('-m')
        run_args.append(augmentor.config['run_name'])
        logging.info(f"Starting Run: {augmentor.config['run_name']} with gamma={augmentor.config['params']['gamma']}")
        train_main(augmentor)
