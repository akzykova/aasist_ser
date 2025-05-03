"""
Main script that trains, validates, and evaluates
various models including AASIST.
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list, Dataset_Custom)
from evaluation import calculate_tDCF_EER, compute_eer
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


from models.AASIST_GFILM import AASISTGFILM
from models.AASIST_Concat import AASISTConcat
from models.AASIST_FILM import AASISTFILM

from tqdm import tqdm

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        print("Start evaluation...")
        evaluate_per_emotion(model, device, config['emo_bonafide'], config['emo_spoof'])

        # produce_evaluation_file(eval_loader, model, device,
        #                         eval_score_path, eval_trial_path)
        # calculate_tDCF_EER(cm_scores_file=eval_score_path,
        #                    output_file=model_tag / "t-DCF_EER.txt")
        # print("DONE.")
        # eval_eer, eval_tdcf = calculate_tDCF_EER(
        #     cm_scores_file=eval_score_path,
        #     output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        # print(eval_eer, eval_tdcf)
        sys.exit(0)

    optimizer = torch.optim.Adam(
        [
            {'params': model.film.parameters()},
            {'params': model.classifier.parameters()}
        ],
        lr=optim_config["base_lr"],
        betas=tuple(optim_config["betas"]),
        weight_decay=optim_config["weight_decay"],
        amsgrad=optim_config["amsgrad"]
    )


    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device, config)
        print(f"DONE. \n Loss: {running_loss:.5f}")

        evaluate_per_emotion(model, device, config['emo_bonafide'], config['emo_spoof'])

        # produce_evaluation_file(dev_loader, model, device,
        #                         metric_path/"dev_score.txt", dev_trial_path)
        # dev_eer, dev_tdcf = calculate_tDCF_EER(
        #     cm_scores_file=metric_path/"dev_score.txt",
        #     output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
        #     printout=False)
        
        # print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
        #     running_loss, dev_eer, dev_tdcf))
        
        
        # best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        # if best_dev_eer >= dev_eer:
        #     print("best model find at epoch", epoch)
        #     best_dev_eer = dev_eer

        model_state = model.state_dict()

        torch.save(
            model_state,
            model_save_path / f"epoch_{epoch}_full_model.pth"
        )
        print(f"Saved model weights and optimizer state to {model_save_path}/epoch_{epoch}_full_model.pth")


    print('End of training')

    # print("Start evaluation...")
    # produce_evaluation_file(eval_loader, model, device,
    #                         eval_score_path, eval_trial_path)
    # calculate_tDCF_EER(cm_scores_file=eval_score_path,
    #                     output_file=model_tag / "t-DCF_EER.txt")
    # print("DONE.")
    # eval_eer, eval_tdcf = calculate_tDCF_EER(
    #     cm_scores_file=eval_score_path,
    #     output_file=model_tag/"loaded_model_t-DCF_EER.txt")
    # print(eval_eer, eval_tdcf)


def evaluate_per_emotion(model, device, esd_dir, zonos_dir):
    model.eval()

    emotions = ["angry_flac", "happy_flac", "neutral_flac", "sad_flac", "surprised_flac"]
    for emotion in emotions:
        print(f'Validating emotion: {emotion}')
        
        esd_scores = run_inference_on_folder(model, device, Path(esd_dir) / emotion)
        
        zonos_scores = run_inference_on_folder(model, device, Path(zonos_dir) / emotion)
        
        eer, _ = compute_eer(esd_scores, zonos_scores)
        
        print(f"EER for {emotion}: {eer * 100}")

def run_inference_on_folder(model, device, folder_path):
    audio_files = [f.stem for f in folder_path.glob("*.flac")]
    test_dataset = Dataset_Custom(list_IDs=audio_files, base_dir=folder_path)

    gen = torch.Generator()
    gen.manual_seed(42)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=24,
        shuffle=False,
        generator=gen
    )

    results = []
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        results.extend(batch_score.tolist())
    return np.array(results)

def get_model(model_config: Dict, device: torch.device):
    
    model = AASISTWithEmotion(
        aasist_config=model_config["aasist_config"],
        ser_config=model_config["ser_config"]
    ).to(device)
    
    if "model_path" in model_config:
        print(f"\nLoading weights from {model_config['model_path']}")
        try:
            state_dict = torch.load(model_config["model_path"], map_location=device)
            model.load_state_dict(state_dict, strict=False)
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {unexpected_keys}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    
    def count_params(module):
        return sum(p.numel() for p in module.parameters())
    
    total_params = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel summary:")
    print(f"Total: {total_params:,} params (Trainable: {trainable_params:,})")
    
    return model



def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []

    pbar = tqdm(data_loader, desc="Evaluating", leave=False)

    for batch_x, utt_id in pbar:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    ii = 0
    num_total = 0.0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(trn_loader, desc="Training", leave=False)

    for batch_x, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()

        optim.step()
        
        pbar.set_postfix({
            'loss': batch_loss.item()
        })

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())