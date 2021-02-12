import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from isegm.utils.exp import load_config_file


def parse_args():
    parser = argparse.ArgumentParser()

    group_pkl_path = parser.add_mutually_exclusive_group(required=True)
    group_pkl_path.add_argument('--folder', type=str, default=None,
                                help='Path to folder with .pickle files.')
    group_pkl_path.add_argument('--files', nargs='+', default=None,
                                help='List of paths to .pickle files separated by space.')
    group_pkl_path.add_argument('--model-dirs', nargs='+', default=None,
                                help="List of paths to model directories with 'plots' folder "
                                     "containing .pickle files separated by space.")
    group_pkl_path.add_argument('--exp-models', nargs='+', default=None,
                                help='List of experiments paths suffixes (relative to cfg.EXPS_PATH/evaluation_logs). '
                                     'For each experiment, the checkpoint prefix must be specified '
                                     'by using the ":" delimiter at the end.')

    parser.add_argument('--mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                           'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        default=None, nargs='*', help='')
    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,COCO_MVal,SBD',
                        help='List of datasets for plotting the iou analysis'
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, COCO_MVal, SBD')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--n-clicks', type=int, default=-1,
                        help='Maximum number of clicks to plot.')
    parser.add_argument('--plots-path', type=str, default='',
                        help='The path to the evaluation logs. '
                             'Default path: cfg.EXPS_PATH/evaluation_logs/iou_analysis.')

    args = parser.parse_args()

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    args.datasets = args.datasets.split(',')
    if args.plots_path == '':
        args.plots_path = cfg.EXPS_PATH / 'evaluation_logs/iou_analysis'
    else:
        args.plots_path = Path(args.plots_path)
    print(args.plots_path)
    args.plots_path.mkdir(parents=True, exist_ok=True)

    return args, cfg


def main():
    args, cfg = parse_args()

    files_list = get_files_list(args, cfg)

    # Dict of dicts with mapping dataset_name -> model_name -> results
    aggregated_plot_data = defaultdict(dict)
    for file in files_list:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        data['all_ious'] = [x[:args.n_clicks] for x in data['all_ious']]
        aggregated_plot_data[data['dataset_name']][data['model_name']] = np.array(data['all_ious']).mean(0)

    for dataset_name, dataset_results in aggregated_plot_data.items():
        plt.figure(figsize=(12, 7))

        max_clicks = 0
        for model_name, model_results in dataset_results.items():
            if args.n_clicks != -1:
                model_results = model_results[:args.n_clicks]

            n_clicks = len(model_results)
            max_clicks = max(max_clicks, n_clicks)

            miou_str = ' '.join([f'mIoU@{click_id}={model_results[click_id-1]:.2%};'
                                 for click_id in [1, 3, 5, 10, 20] if click_id <= len(model_results)])
            print(f'{model_name} on {dataset_name}:\n{miou_str}\n')

            plt.plot(1 + np.arange(n_clicks), model_results, linewidth=2, label=model_name)

        plt.title(f'mIoU after every click for {dataset_name}', fontsize='x-large')
        plt.grid()
        plt.legend(loc=4, fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.xticks(1 + np.arange(max_clicks), fontsize='x-large')

        fig_path = get_target_file_path(args.plots_path, dataset_name)
        plt.savefig(str(fig_path))


def get_target_file_path(plots_path, dataset_name):
    previous_plots = sorted(plots_path.glob(f'{dataset_name}_*.png'))
    if len(previous_plots) == 0:
        index = 0
    else:
        index = int(previous_plots[-1].stem.split('_')[-1]) + 1

    return str(plots_path / f'{dataset_name}_{index:03d}.png')


def get_files_list(args, cfg):
    if args.folder is not None:
        files_list = Path(args.folder).glob('*.pickle')
    elif args.files is not None:
        files_list = args.files
    elif args.model_dirs is not None:
        files_list = []
        for folder in args.model_dirs:
            folder = Path(folder) / 'plots'
            files_list.extend(folder.glob('*.pickle'))
    elif args.exp_models is not None:
        files_list = []
        for rel_exp_path in args.exp_models:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')
            exp_path_prefix = cfg.EXPS_PATH / 'evaluation_logs' / rel_exp_path
            candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
            assert len(candidates) == 1, "Invalid experiment path."
            exp_path = candidates[0]
            files_list.extend(sorted((exp_path / 'plots').glob(checkpoint_prefix + '*.pickle')))

    if args.mode is not None:
        files_list = [file for file in files_list
                      if any(mode in file.stem for mode in args.mode)]
    files_list = [file for file in files_list
                  if any(dataset in file.stem for dataset in args.datasets)]

    return files_list


if __name__ == '__main__':
    main()
