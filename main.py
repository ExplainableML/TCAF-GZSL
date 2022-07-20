import sys
import torch
from torch import optim
from torch.utils import data
#from ptflops import get_model_complexity_info
from src.args import args_main
from get_evaluation import get_evaluation
#from src.args_sem_pcyc import Options
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, ContrastiveDataset, VGGSoundDataset, UCFDataset
from src.dataset import DefaultCollator
from src.metrics import DetailedLosses, MeanClassAccuracy, PercentOverlappingClasses, TargetDifficulty
from src.model_sequence import Multimodal_Sequence_Transformer
from src.sampler import SamplerFactory
from src.train import train
from src.loss import L2Loss
from src.utils import fix_seeds, setup_experiment, get_git_revision_hash, print_model_size
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils_improvements import get_model_params


def run():
    args, eval_args = args_main()

    run_mode = args.run
    best_epoch = None
    if run_mode == 'stage-1' or run_mode == 'all':
        args.retrain_all = False
        args.save_checkpoints = False
        path_stage_1, best_epoch = main(args)
        eval_args.load_path_stage_A = path_stage_1

    if run_mode == 'stage-2' or run_mode == 'all':
        # set stage-2 args if not yet set
        args.retrain_all = True
        args.save_checkpoints = True

        if best_epoch:
            # train stage-2 only for required epochs
            args.epochs = best_epoch + 1

        path_stage_2, _ = main(args)
        eval_args.load_path_stage_B = path_stage_2

    if run_mode == 'eval' or run_mode == 'all':
        assert eval_args.load_path_stage_A != None
        assert eval_args.load_path_stage_B != None
        get_evaluation(eval_args)


def main(args):
    # args, eval_args = args_main()

    if args.input_size is not None:
        args.input_size_audio = args.input_size
        args.input_size_video = args.input_size
    fix_seeds(args.seed)
    logger, log_dir, writer, train_stats, val_stats = setup_experiment(args, "epoch", "loss", "hm")

    logger.info("Git commit hash: " + get_git_revision_hash())

    if args.dataset_name == "AudioSetZSL":
        train_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="seen",
        )

        val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode="seen",
        )

        train_val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode="seen",
        )

        val_all_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode="all",
        )

    elif args.dataset_name == "VGGSound":
        if args.retrain_all==False:
            train_dataset = VGGSoundDataset(
                args=args,
                dataset_split="train",
                zero_shot_mode="train",
            )

        if args.retrain_all==True:
            train_val_dataset = VGGSoundDataset(
                args=args,
                dataset_split="train_val",
                zero_shot_mode=None,
            )

        val_all_dataset = VGGSoundDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        if args.retrain_all==False:
            train_dataset = UCFDataset(
                args=args,
                dataset_split="train",
                zero_shot_mode="train",
            )
        if args.retrain_all==True:
            train_val_dataset = UCFDataset(
                args=args,
                dataset_split="train_val",
                zero_shot_mode=None,
            )

        val_all_dataset = UCFDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        if args.retrain_all==False:
            train_dataset = ActivityNetDataset(
                args=args,
                dataset_split="train",
                zero_shot_mode="train",
            )

        if args.retrain_all==True:
            train_val_dataset = ActivityNetDataset(
                args=args,
                dataset_split="train_val",
                zero_shot_mode=None,
            )
        val_all_dataset = ActivityNetDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    if args.retrain_all==False:
        contrastive_train_dataset = ContrastiveDataset(train_dataset)
    if args.retrain_all==True:
        contrastive_train_val_dataset = ContrastiveDataset(train_val_dataset)
    contrastive_val_all_dataset = ContrastiveDataset(val_all_dataset)

    if args.retrain_all==False:
        train_sampler = SamplerFactory(logger).get(
            class_idxs=list(contrastive_train_dataset.target_to_indices.values()),
            batch_size=args.bs,
            n_batches=args.n_batches,
            alpha=1,
            kind='random'
        )

    if args.retrain_all==True:
        train_val_sampler = SamplerFactory(logger).get(
            class_idxs=list(contrastive_train_val_dataset.target_to_indices.values()),
            batch_size=args.bs,
            n_batches=args.n_batches,
            alpha=1,
            kind='random'
        )

    val_all_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_all_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )
    if args.selavi==False:
        collator_train = DefaultCollator(mode=args.batch_seqlen_train, max_len=args.batch_seqlen_train_maxlen, trim=args.batch_seqlen_train_trim,)
        collator_test = DefaultCollator(mode=args.batch_seqlen_test, max_len=args.batch_seqlen_test_maxlen, trim=args.batch_seqlen_test_trim)
    elif args.selavi==True:
        collator_train = DefaultCollator(mode=args.batch_seqlen_train, max_len=args.batch_seqlen_train_maxlen,trim=args.batch_seqlen_train_trim,rate_video=1, rate_audio=1)
        collator_test = DefaultCollator(mode=args.batch_seqlen_test, max_len=args.batch_seqlen_test_maxlen,trim=args.batch_seqlen_test_trim,rate_video=1, rate_audio=1)

    if args.retrain_all==False:
        train_loader = data.DataLoader(
            dataset=contrastive_train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collator_train,
            num_workers=4
        )

    final_test_loader=data.DataLoader(
        dataset=contrastive_val_all_dataset,
        collate_fn=collator_test,
        batch_size=args.bs,
        num_workers=4
    )

    if args.retrain_all==True:
        train_val_loader = data.DataLoader(
            dataset=contrastive_train_val_dataset,
            batch_sampler=train_val_sampler,
            collate_fn=collator_train,
            num_workers=4
        )

    val_all_loader = data.DataLoader(
        dataset=contrastive_val_all_dataset,
        batch_sampler=val_all_sampler,
        collate_fn=collator_test,
        num_workers=4
    )

    model_params = get_model_params(
        args.lr, args.reg_loss, args.embedding_dropout,
        args.decoder_dropout, args.additional_dropout, args.embeddings_hidden_size, args.decoder_hidden_size,
        args.embeddings_batch_norm, args.rec_loss, args.cross_entropy_loss,
        args.transformer_use_embedding_net, args.transformer_dim, args.transformer_depth, args.transformer_heads,
        args.transformer_dim_head, args.transformer_mlp_dim, args.transformer_dropout,
        args.transformer_embedding_dim, args.transformer_embedding_time_len, args.transformer_embedding_dropout,
        args.transformer_embedding_time_embed_type, args.transformer_embedding_fourier_scale, args.transformer_embedding_embed_augment_position,
        args.lr_scheduler, args.optimizer, args.use_self_attention, args.use_cross_attention, args.transformer_average_features,
        args.audio_only, args.video_only, args.transformer_use_class_token, args.transformer_embedding_modality,
    )
    if args.new_model_sequence==True:
        model = Multimodal_Sequence_Transformer(model_params, input_size_audio=args.input_size_audio, input_size_video=args.input_size_video)
    else:
        raise AttributeError("No correct model name.")
    print_model_size(model, logger)
    model.to(args.device)

    distance_fn = getattr(sys.modules[__name__], args.distance_fn)()
    metrics = [
        MeanClassAccuracy(model=model, dataset=(val_all_dataset, final_test_loader), device=args.device, distance_fn=distance_fn,
                          new_model_sequence=args.new_model_sequence,
                          args=args)
                ]


    logger.info(model)
    logger.info(None)
    logger.info(None)
    logger.info(None)
    logger.info([metric.__class__.__name__ for metric in metrics])

    # optimizer not used for model_sequence

    optimizer = None
    lr_scheduler = None

    best_loss, best_score, best_epoch = train(
        train_loader=train_val_loader if args.retrain_all else train_loader,
        val_loader=val_all_loader,
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=None,
        epochs=args.epochs,
        device=args.device,
        writer=writer,
        metrics=metrics,
        train_stats=train_stats,
        val_stats=val_stats,
        log_dir=log_dir,
        new_model_sequence=args.new_model_sequence,
        args=args
    )

    logger.info(f"FINISHED. Run is stored at {log_dir}")

    return log_dir, best_epoch


if __name__ == '__main__':
    run()
