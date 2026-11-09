#!/usr/bin/env python3
import datetime

from pathlib import Path
from collections import defaultdict
import fire
import math
from typing import Dict, List, Any, Union, Sequence, Tuple
import logging

import torch
import numpy as np
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events)
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    CosineAnnealingScheduler,
    global_step_from_engine,
)
import ignite
import ignite.distributed as idist
from ignite.metrics import RunningAverage, Loss, EpochMetric
from ignite.utils import convert_tensor
from torch.cuda.amp import GradScaler, autocast
from ignite.utils import manual_seed, setup_logger

from datasets import dataset
from model import UniCueNet as models
from loss import criterion as losses
import utils
import visualize
import wandb

import os
torch.backends.cudnn.benchmark = False

def wandb_init(config_parameters, output_dir):
    wandb.init(
            project="unicine",
            name=config_parameters.get('cue', 'all_sep'),
            config=config_parameters,
            dir=output_dir
        )

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_num_threads(1)
# Ensure no distributed port is forced via environment
os.environ.pop('MASTER_PORT', None)

def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")

def transfer_to_device(batch):
    DEVICE = idist.device()
    return (x.to(DEVICE, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


def __setup(config: Path,
            default_args=utils.DEFAULT_ARGS,
            **override_kwargs) -> Dict[str, Any]:
    config_parameters = utils.parse_config_or_kwargs(
        config, default_args=default_args, **override_kwargs)
    
    return config_parameters


def log_basic_info(logger, config_parameters):
    logger.info(f"Train {config_parameters['model']} on UniCue")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
    # explicitly import cudnn as
    # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config_parameters.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")

def train(local_rank, config_parameters):
    """Trains a given model specified in the config file or passed as the --model parameter.
    All options in the config file can be overwritten as needed by passing --PARAM
    Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'

    :param config: yaml config file
    :param **kwargs: parameters to overwrite yaml config
    """
    rank = idist.get_rank()
    manual_seed(config_parameters["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name='vox2_ddp')
    metrics_tracker = visualize.MetricsTracker()
    

    outputpath = config_parameters['outputpath']
    if rank == 0:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
        folder_name = f"{config_parameters['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        output_dir = Path(outputpath) / folder_name
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        config_parameters["outputpath"] = output_dir.as_posix()
        log_file = output_dir / 'train.log'
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
        logger.info(f"Output dir: {config_parameters['outputpath']}")
        log_basic_info(logger, config_parameters)

        if "cuda" in device.type:
            config_parameters["cuda device name"] = torch.cuda.get_device_name(local_rank)

        wandb_init(config_parameters, output_dir)
  
    train_df = config_parameters['train_scp']
    cv_df = config_parameters['eval_scp']
    # train_dur = config_parameters['train_dur']
    # cv_dur = config_parameters['eval_dur']

    train_ds = dataset.MixtureDataset(
        train_df,   
        'train',
        sample_rate=16000,
        max_length_sec=6.0,
        # start_index=12700
    )
    cv_ds = dataset.MixtureDataset(
        cv_df, 
        'test',
        sample_rate=16000,
        max_length_sec=6.0
    )
    nproc = idist.get_nproc_per_node() 
    train_loader = idist.auto_dataloader(train_ds, batch_size=nproc, num_workers=config_parameters['num_workers'] * nproc, shuffle=False, drop_last=True, collate_fn = dataset.dummy_collate_fn)
    cv_loader = idist.auto_dataloader(cv_ds, batch_size=nproc, num_workers=config_parameters['num_workers'] * nproc, shuffle=False, drop_last=True, collate_fn = dataset.dummy_collate_fn)
    config_parameters['num_iters_per_epoch'] = len(train_loader)

    model = getattr(models, config_parameters['model'])()
    model = idist.auto_model(model, sync_bn = True)

    optimizer = getattr(
        torch.optim,
        config_parameters['optimizer'],
    )(model.parameters(), **config_parameters['optimizer_args'])
    optimizer = idist.auto_optim(optimizer)


    loss_func = getattr(losses, config_parameters['loss'])().to(idist.device())

    epochs = config_parameters['epochs']
    decay_steps = epochs * config_parameters['num_iters_per_epoch']

    trainer = create_trainer(model, optimizer, loss_func, train_loader.sampler, config_parameters, logger)
    # Attach metrics tracker to trainer so train_step can record iteration metrics
    setattr(trainer, "_metrics_tracker", metrics_tracker)
    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = {
            "si-snr loss": Loss(loss_func)
            }

    evaluator = create_evaluator(model, metrics=metrics, config_parameters=config_parameters)

    def run_validation(engine):
        epoch = trainer.state.epoch
        eval_num = int(config_parameters.get("eval_num", 1000))
        batch_size = cv_loader.batch_size or 1
        max_batches = max(1, math.ceil(eval_num / batch_size))
        epoch_length = min(len(cv_loader), max_batches)
        state = evaluator.run(cv_loader, epoch_length=epoch_length)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Val", state.metrics)
        if idist.get_rank() == 0:
            val_loss_value = float(state.metrics.get("si-snr loss", 0.0))
            si_snr_value = -val_loss_value
            metrics_tracker.log_val_epoch(epoch, val_loss_value, si_snr_value)

            wandb.log({
                "val/loss": val_loss_value,
                "val/si_snr": si_snr_value,
                "val/eval_time": state.times["COMPLETED"],
                "epoch": epoch
            })
    
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda _: logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=config_parameters["validate_every"]) | Events.COMPLETED, run_validation)

    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config_parameters),
        filename_prefix="best",
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_si_snr",
        score_function=Checkpoint.get_default_score_fn("si-snr loss", -1.0),
    )

    # lr scheduler
    half_lr_rate_handler = utils.half_lr_rate(optimizer = optimizer, param_name = 'lr', score_function = Checkpoint.get_default_score_fn("si-snr loss", -1.0), patience = 3)
    evaluator.add_event_handler(Events.COMPLETED, half_lr_rate_handler) 

    evaluator.add_event_handler(Events.COMPLETED(lambda *_: trainer.state.epoch > 5), best_model_handler)

    earlystop_handler = EarlyStopping(
            patience=config_parameters.get('early_stop', 20),
            score_function=Checkpoint.get_default_score_fn('si-snr loss', -1.0),
            trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)
        
    try:
        trainer.run(train_loader, max_epochs=config_parameters["epochs"], epoch_length=config_parameters["epoch_length"])
    except Exception as e:
        logger.exception("")
        raise e
    finally:
        if idist.get_rank() == 0:
            # save plots & csv
            output_dir = Path(config_parameters["outputpath"]) if isinstance(config_parameters["outputpath"], str) else config_parameters["outputpath"]
            metrics_tracker.save_csv(output_dir)
            metrics_tracker.plot(output_dir)

            wandb.finish()


def create_trainer(model, optimizer, criterion, train_sampler, config_parameters, logger):
    with_amp = config_parameters["with_amp"]
    scaler = GradScaler(enabled=with_amp)
    
    def train_step(engine, batch):
        model.train()
        with autocast(enabled=with_amp):
            mixture, source, spk1_face, spk2_face, spk1_body, spk2_body, pre, text, main = batch
            # print("mixture shape:", mixture.shape)
            if(mixture.shape[1] > 80000):
                mixture = mixture[:, :80000]
                source = source[:, :80000]

                # 80000 samples 对应 5 秒视频，即 5 * 25 = 125 帧
                if spk1_body is not None:
                    spk1_body = spk1_body[:5 * 25, :]
                    spk2_body = spk2_body[:5 * 25, :]
                if spk1_face is not None:
                    spk1_face = spk1_face[:5 * 25, :]
                    spk2_face = spk2_face[:5 * 25, :]
            # print("source shape:", source.shape)
            # print("text:", text)
            # mixture, source = mixture.squeeze(0), source.squeeze(0)
            # print("After squeeze, mixture shape:", mixture.shape)
            # print("After squeeze, source shape:", source.shape)
            mixture, source, spk1_face, spk2_face, spk1_body, spk2_body, pre, main = transfer_to_device([mixture, source, spk1_face, spk2_face, spk1_body, spk2_body, pre, main])
            # pred_wav = model(mixture, condition)
            if spk1_face is not None:
                # spk1_av = spk1_body * 0.5 + spk1_face * 0.5
                # spk2_av = spk2_body * 0.5 + spk2_face * 0.5
                # av = torch.cat((spk1_av.unsqueeze(0), spk2_av.unsqueeze(0)), dim=0)
                # spk1_av = spk1_body
                # spk2_av = spk2_body
                spk1_av = torch.cat((spk1_body, spk1_face), dim=1)
                spk2_av = torch.cat((spk2_body, spk2_face), dim=1)
                av = torch.cat((spk1_av.unsqueeze(0), spk2_av.unsqueeze(0)), dim=0)
            else:
                spk1_av = None
                spk2_av = None
                av = None
            # print("--------spk1_av---------: ", spk1_av.shape)
            # print("--------spk2_av---------: ", spk2_av.shape)
            # print("----------av_shape", av.shape)
            pred_wav = model(mixture, av, pre, text, main)
            # print("pred_wav shape:", pred_wav.shape)
            # print("source shape:", source.shape)
            loss = criterion(source, pred_wav)
            total_loss = loss

            if not torch.isfinite(total_loss):
                logger.warning(f"Detected non-finite loss (NaN/Inf) at epoch={engine.state.epoch} iter={getattr(engine, '_global_iter', engine.state.iteration)}. Skipping batch.")
                # 记录到 wandb（仅主进程）以便排查
                if idist.get_rank() == 0:
                    try:
                        global_iter = getattr(engine, '_global_iter', None)
                        if global_iter is None:
                            global_iter = engine.state.iteration + (engine.state.epoch - 1) * config_parameters['num_iters_per_epoch'] if engine.state.epoch is not None else engine.state.iteration
                        wandb.log({"train/loss": float('nan'), "train/skipped_batch": 1}, step=global_iter)
                    except Exception:
                        pass
                # 返回一个包含 NaN 的输出，且不执行 backward/step
                return {"si-snr loss": float('nan')}
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(optimizer)
        scaler.update()

        output = {"si-snr loss": loss.item()}
        # Log iteration loss on rank 0 for plotting
        if idist.get_rank() == 0:
            # engine.state.iteration is global within epoch; use global iteration counter
            global_iter = engine.state.iteration + (engine.state.epoch - 1) * config_parameters['num_iters_per_epoch'] if engine.state.epoch is not None else engine.state.iteration
            # Access outer scope metrics_tracker via engine (attach later)
            mt = getattr(engine, "_metrics_tracker", None)
            if mt is not None:
                mt.log_train_iter(global_iter, float(loss.item()))

        if not hasattr(engine, "_global_iter"):
            engine._global_iter = 0
        engine._global_iter += 1
        global_iter = engine._global_iter

        # 使用 global_iter log
        mt = getattr(engine, "_metrics_tracker", None)
        if mt is not None:
            mt.log_train_iter(global_iter, float(loss.item()))
            
        if global_iter % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/iteration": global_iter
            }, step=global_iter)

        return output
    trainer = Engine(train_step)
    trainer.logger = logger
    # Placeholder for attaching metrics tracker from outer scope
    setattr(trainer, "_metrics_tracker", None)

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer}
    metric_names = [
        "si-snr loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        save_handler=get_save_handler(config_parameters),
        output_names=metric_names,
        with_pbars=True,
        clear_cuda_cache=False,
        log_every_iters=1
    )

    resume_from = config_parameters.get("resume_from", None)
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_evaluator(model, metrics, config_parameters, tag='val'):
    with_amp = config_parameters['with_amp']

    @torch.no_grad()
    def evaluate_step(engine, batch):
        model.eval()
        mixture, source, spk1_face, spk2_face, spk1_body, spk2_body, pre, text, main = batch
        if(mixture.shape[1] > 80000):
            mixture = mixture[:, :80000]
            source = source[:, :80000]

            # 80000 samples 对应 5 秒视频，即 5 * 25 = 125 帧
            if spk1_body is not None:
                spk1_body = spk1_body[:5 * 25, :]
                spk2_body = spk2_body[:5 * 25, :]
            if spk1_face is not None:
                spk1_face = spk1_face[:5 * 25, :]
                spk2_face = spk2_face[:5 * 25, :]
        # mixture, source = mixture.squeeze(0), source.squeeze(0)
        mixture, source, spk1_face, spk2_face, spk1_body, spk2_body, pre, main = transfer_to_device([mixture, source, spk1_face, spk2_face, spk1_body, spk2_body, pre, main])
        if spk1_face is not None:
            # spk1_av = spk1_body * 0.5 + spk1_face * 0.5
            # spk2_av = spk2_body * 0.5 + spk2_face * 0.5
            # av = torch.cat((spk1_av.unsqueeze(0), spk2_av.unsqueeze(0)), dim=0)
            # spk1_av = spk1_body
            # spk2_av = spk2_body
            spk1_av = torch.cat((spk1_body, spk1_face), dim=1)
            spk2_av = torch.cat((spk2_body, spk2_face), dim=1)
            av = torch.cat((spk1_av.unsqueeze(0), spk2_av.unsqueeze(0)), dim=0)
        else:
            spk1_av = None
            spk2_av = None
            av = None
        with autocast(enabled=with_amp):
            pred_wav = model(mixture, av, pre, text, main)
        return source, pred_wav
    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    if idist.get_rank() == 0:
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)
            
    return evaluator

def run(config, **kwargs):
    setup_args = __setup(config, **kwargs)
    config_parameters = setup_args
    config_parameters.setdefault("eval_num", 1000)
    # Force single-process execution without specifying a port
    # Use 'gloo' backend for widest compatibility (CPU/CUDA, single process)
    config_parameters['backend'] = 'gloo'
    config_parameters['nproc_per_node'] = 1
    config_parameters['num_workers'] = 0
    with idist.Parallel(backend = config_parameters['backend'], nproc_per_node=1) as parallel:
        parallel.run(train, config_parameters)
   
def get_save_handler(config):
    return DiskSaver(config["outputpath"], require_empty=False)

if __name__ == "__main__":
    fire.Fire({"run": run})
