import os
conf = {
    "WORK_PATH": os.path.join(os.path.dirname(os.path.abspath(__file__)),"work"),
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': os.path.join(os.path.dirname(os.path.abspath(__file__)),"OUMVLP"),
        'resolution': '64',
        'dataset': 'OUMVLP',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data

        #insert half number of views.
        'pid_num': 1000,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 1000,
        'margin': 0.2,
        'num_workers': 0,
        'frame_num': 30,
        'model_name': 'geiset',
    },
}
