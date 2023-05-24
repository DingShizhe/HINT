# encoding: utf-8
import sys
from pathlib import Path
import configargparse

from src.DeepRegression import Model
from src import train, test, test_Mathit, save_mathit_testdataset_as_file


def main():

    assert sys.argv[1] == "--config_path", "The first argument must be .yml relative config path"
    config_path = Path(__file__).absolute().parent / sys.argv[2]
    sys.argv = sys.argv[0:1] + sys.argv[3:]

    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[str(config_path),],
        description="Hyper-parameters.",
    )

    # mode
    parser.add_argument(
        "-m", "--mode", type=str, default="train", help="model: train or test or plot"
    )

    # problem dimension
    parser.add_argument(
        "--prob_dim", default=2, type=int, help="dimension of the problem"
    )

    # args for plot in point-based methods
    parser.add_argument("--plot", action="store_true", help="use profiler")

    # args for training
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="which gpu: 0 for cpu, 1 for gpu 0, 2 for gpu 1, ...",
    )
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--lr", default="1.0", type=float)
    parser.add_argument("--lr_decay", default="-1.0", type=float)
    parser.add_argument(
        "--resume_from_checkpoint", type=str, help="resume from checkpoint"
    )
    parser.add_argument(
        "--num_workers", default=2, type=int, help="num_workers in DataLoader"
    )
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument(
        "--use_16bit", type=bool, default=False, help="use 16bit precision"
    )
    parser.add_argument("--profiler", action="store_true", help="use profiler")

    # args for validation
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1,
        help="how often within one training epoch to check the validation set",
    )

    # args for validation
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="how many epochs to check the validation set",
    )

    # args for testing
    parser.add_argument(
        "-v", "--test_check_num", default="0", type=str, help="checkpoint for test"
    )
    parser.add_argument("--test_args", action="store_true", help="print args")


    # DSZ define these
    parser.add_argument(
        "--dataset_type", type=str, default="Mathit", help="dataset: Mathit or TFR"
    )

    parser.add_argument(
        "--mathit_data_cfg", type=str, default=None, help="mathit_data_cfg: dataset generation config"
    )

    parser.add_argument(
        "--model_arch_cfg", type=str, default=None, help="model_arch_cfg: model architecture config"
    )

    parser.add_argument(
        "--cfg_dim_input", type=int, default=-1, help="cfg_dim_input: input point dimension"
    )

    parser.add_argument(
        "--cfg_dim_output", type=int, default=-1, help="cfg_dim_output: output point dimension"
    )


    import math
    parser.add_argument(
        "--SAMPLE_TP", type=float, default=math.nan, help=""
    )

    parser.add_argument(
        "--wandb_project", type=str, default="", help="wandb project name"
    )

    parser.add_argument(
        "--checkpoint_dir", type=str, default="none", help="wandb project name"
    )

    parser.add_argument(
        "--observed_as_target", type=bool, default=False, help="observed points as target points"
    )

    parser.add_argument(
        "--align_observed", type=bool, default=False, help="align features of observed points"
    )

    parser.add_argument(
        "--align_loss_factor", type=float, default=0.1, help="align features loss factor"
    )

    parser.add_argument(
        "--o_loss_coff", type=float, default=0.5, help="observed loss factor"
    )

    parser.add_argument(
        "--gradient_clip", type=float, default=0.1, help="gradient_clip"
    )

    parser.add_argument(
        "--K_0_inv", type=int, default=2, help="K0 = n / K_0_inv"
    )

    parser.add_argument(
        "--K_min_inv", type=int, default=8, help="K_min = n / K_0_inv"
    )

    parser.add_argument(
        "--K_min", type=int, default=-1, help="default K_min"
    )


    parser.add_argument(
        "--n_blocks", type=int, default=4, help="n_blocks"
    )



    # args from Model
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # running
    assert hparams.mode in ["train", "test", "test_Mathit", "save_mathit_testdataset_as_file"]
    if hparams.test_args:
        print(hparams)
    else:
        getattr(eval(hparams.mode), "main")(hparams)


if __name__ == "__main__":
    main()

