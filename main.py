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
                        Dataset_ASVspoof2019_devNeval, genSpoof_list, DatasetCustom)
from evaluation import calculate_tDCF_EER, compute_eer
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)

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
        evaluate_per_emotion(model, device, config['emo_dataset'])

        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        eval_eer, _ = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        print(eval_eer)
        sys.exit(0)

    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer = create_optimizer(model.parameters(), optim_config)

    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device, config)
        print(f"DONE. \n Loss: {running_loss:.5f}")

        evaluate_per_emotion(model, device, config['emo_dataset'])

        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
            running_loss, dev_eer, dev_tdcf))

        model_state = model.state_dict()

        torch.save(
            model_state,
            model_save_path / f"epoch_{epoch}_full_model.pth"
        )
        print(f"Saved model weights to {model_save_path}/epoch_{epoch}_full_model.pth")


    print('End of training')

    print("Start final evaluation...")
    produce_evaluation_file(eval_loader, model, device,
                            eval_score_path, eval_trial_path)
    calculate_tDCF_EER(cm_scores_file=eval_score_path,
                        output_file=model_tag / "t-DCF_EER.txt")
    print("DONE.")
    eval_eer, _ = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        output_file=model_tag/"loaded_model_t-DCF_EER.txt")
    print(eval_eer)

def evaluate_per_emotion(model, device, dataset_dir):
    model.eval()
    emotions = ["angry", "happy", "neutral", "sad", "surprised"]

    for emotion in emotions:
        print(f'Validating emotion: {emotion}')
        synth_scores = []
        bonafide_scores = []

        for speaker_id in range(11, 21):
            bona_path = Path(dataset_dir) / 'ESD' / f"00{speaker_id}" / emotion
            esd_scores = run_inference_on_folder(model, device, bona_path)
            bonafide_scores.extend(esd_scores)

            for dataset in ['Zonos', 'CosyVoice', 'EmoSpeech']:
                spoof_path = Path(dataset_dir) / dataset / f"00{speaker_id}"/ emotion
                scores = run_inference_on_folder(model, device, spoof_path)
                synth_scores.extend(scores)

        synth_scores = np.array(synth_scores)
        bonafide_scores = np.array(bonafide_scores)

        eer, _ = compute_eer(bonafide_scores, synth_scores)
        print(f"EER for {emotion}: {eer * 100:.2f}%")


def run_inference_on_folder(model, device, folder_path):
    test_dataset = DatasetCustom(audio_dir=Path(folder_path))

    gen = torch.Generator()
    gen.manual_seed(42)
    
    num_workers = min(4, os.cpu_count() // 2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=24,
        shuffle=False,
        persistent_workers=num_workers > 0,
        generator=gen
    )

    results = []
    for batch_x, batch_emo, _, filenames in test_loader:
        try:
            batch_x = batch_x.float().to(device)
            batch_emo = batch_emo.float().to(device)
            
            _, batch_out = model(batch_x, batch_emo)
            scores = batch_out[:, 1].data.cpu().numpy().ravel()
            results.extend(zip(filenames, map(float, scores)))            
        except Exception as e:
            print(f"\nError processing batch: {str(e)}")
            results.extend((f, None) for f in filenames)
    return results

def get_model(model_config: Dict, device: torch.device):
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    if model_config.get("model_path"):
        model.load_state_dict(torch.load(model_config["model_path"], map_location=device))
        print("Weights are downloaded from ", model_config["model_path"])

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

    for batch_x, batch_emo, utt_id in pbar:
        batch_x = batch_x.to(device)
        batch_emo = batch_emo.to(device)

        with torch.no_grad():
            _, batch_out = model(batch_x, batch_emo)
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

    for batch_x, batch_emo, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_emo = batch_emo.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, batch_emo)
        
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