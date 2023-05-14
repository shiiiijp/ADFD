from argparse import ArgumentParser


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default="pretrained_models/sam_ffhq_aging.pt", type=str,
                                 help='Path to pSp model checkpoint')
        self.parser.add_argument('--data_path', type=str, default='gt_images',
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--couple_outputs', action='store_true',
                                 help='Whether to also save inputs + outputs side-by-side')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at 1024x1024')
        self.parser.add_argument('--n_images', type=int, default=None,
                                 help='Number of images to output. If None, run on all data')

        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        # arguments for aging
        self.parser.add_argument('--target_age', type=str, default=None,
                                 help='Target age for inference. Can be comma-separated list for multiple ages.')

        # arguments for ADFD
        self.parser.add_argument('--n_outputs_to_generate', type=int, default=5,
                                 help='Number of outputs to generate per input image.')
        self.parser.add_argument('--div_opt', default='adam', type=str,
                                 help='Optimizer')
        self.parser.add_argument('--div_lr', default=0.01, type=float,
                                 help='Learning rate for optimization')
        self.parser.add_argument('--patience', default=7, type=int,
                                 help='Parameter for EarlyStopping. Defualt value is 7.')
        self.parser.add_argument('--es_delta', default=0.0001, type=float,
                                 help='Parameter for EarlyStopping. Defualt value is 0.0001.')
        self.parser.add_argument('--lpips_lambda', default=0, type=float,
                                 help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0, type=float,
                                 help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=0, type=float,
                                 help='L2 loss multiplier factor')
        self.parser.add_argument('--aging_lambda', default=0, type=float,
                                 help='Aging loss multiplier factor')

        self.parser.add_argument('--lpips_lambda_crop', default=0, type=float,
                                 help='LPIPS loss multiplier factor for inner image region')
        self.parser.add_argument('--l2_lambda_crop', default=0, type=float,
                                 help='L2 loss multiplier factor for inner image region')

        self.parser.add_argument('--lpips_lambda_aging', default=0, type=float,
                                 help='LPIPS loss multiplier factor for aging')
        self.parser.add_argument('--l2_lambda_aging', default=0, type=float,
                                 help='L2 loss multiplier factor for aging')

        self.parser.add_argument('--use_weighted_id_loss', action="store_true",
                                 help="Whether to weight id loss based on change in age (more change -> less weight)")

        # arguments for guided optimization
        self.parser.add_argument('--age_samples_n', default=4, type=int,
                                 help='Number of sample latents to extrapolate with. Default value is 4.')
        self.parser.add_argument('--age_interval', default=3, type=int,
                                 help='Interval of sample latents. Default value is 3.')
        self.parser.add_argument('--max_steps', default=100, type=int,
                                 help='Number of iteration')
        self.parser.add_argument('--tol_beta', default=0.5, type=float,
                                 help='Tolerance parameter. Default value is 0.5.')

    def parse(self):
        opts, unknown = self.parser.parse_known_args()
        return opts
