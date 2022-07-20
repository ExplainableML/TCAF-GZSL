import logging
from tqdm import tqdm
import torch

from src.metrics import MeanClassAccuracy
from src.utils import check_best_loss, check_best_score, save_best_model


def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, epochs, device, writer, metrics,
          train_stats, val_stats, log_dir, new_model_sequence,args):
    best_loss = None
    best_score = None
    best_epoch = None

    for epoch in range(epochs):

        train_loss = train_step(train_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics,
                                train_stats,  args)
        val_loss, val_hm = val_step(val_loader, model, criterion, epoch, epochs, writer, device, metrics, val_stats,args)


        best_loss = check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir)
        best_score, best_epoch = check_best_score(epoch, best_score, best_epoch, val_hm, model, optimizer, log_dir)

        if args.save_checkpoints:
            # save_best_model(epoch, val_loss, model, optimizer, log_dir / "checkpoints", metric="loss", checkpoint=True)
            save_best_model(epoch, val_hm, model, optimizer, log_dir / "checkpoints", metric="score", checkpoint=True)



        model.optimize_scheduler(val_hm)
    return best_loss, best_score, best_epoch

def add_loss_details(current_loss_details, batch_loss_details):
    for key, value in current_loss_details.items():
        if key not in batch_loss_details:
            batch_loss_details[key]=value
        else:
            batch_loss_details[key]+=value
    return batch_loss_details

def add_logs_tensorboard(batch_loss_details, writer, batch_idx, step, which_stage):


    writer.add_scalar(f"Loss/total_loss_"+which_stage, batch_loss_details['Loss/total_loss']/(batch_idx), step)
    writer.add_scalar(f"Loss/loss_reg_"+which_stage, batch_loss_details['Loss/loss_reg']/(batch_idx), step)
    writer.add_scalar(f"Loss/loss_cmd_rec_"+which_stage, batch_loss_details['Loss/loss_cmd_rec']/(batch_idx), step)
    writer.add_scalar(f"Loss/cross_entropy_"+which_stage, batch_loss_details['Loss/cross_entropy']/(batch_idx), step)


def train_step(data_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics, stats,  args):
    logger = logging.getLogger()
    model.train()

    for metric in metrics:
        metric.reset()
    embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
    batch_loss = 0
    batch_loss_details={}
    for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
        p = data["positive"]
        q = data["negative"]

        x_p_a = p["audio"].to(device)
        x_p_v = p["video"].to(device)
        x_p_t = p["text"].to(device)
        x_p_num = target["positive"].to(device)


        masks={}
        masks['positive']={'audio':p['audio_mask'], 'video':p['video_mask']}

        timesteps = {}
        timesteps['positive'] = {'audio': p['timestep']['audio'], 'video': p['timestep']['video']}

        inputs = (
            x_p_a, x_p_v, x_p_num, x_p_t, masks['positive'], timesteps['positive']

        )

        if args.z_score_inputs:
            inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])


        if args.cross_entropy_loss==True:
            for i in range(inputs[2].shape[0]):
                inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
        if args.cross_entropy_loss==True:
            loss, loss_details = model.optimize_params(*inputs, embedding_crossentropy=embeddings, optimize=True)
        else:
            loss, loss_details = model.optimize_params(*inputs, embedding_crossentropy=None, optimize=True)
        batch_loss_details=add_loss_details(loss_details, batch_loss_details)
        audio_emb, video_emb, emb_cls=model.get_embeddings(inputs[0], inputs[1], inputs[3], inputs[4], inputs[5])
        outputs=torch.stack([video_emb, emb_cls], dim=0)

        batch_loss += loss.item()

        p_target = target["positive"].to(device)
        q_target = target["negative"].to(device)

        # url
        p_url = p["url"]
        q_url = q["url"]

        # stats
        iteration = len(data_loader) * epoch + batch_idx
        if iteration % len(data_loader) == 0:
            for metric in metrics:
                if isinstance(metric, MeanClassAccuracy):
                    continue
                metric(outputs, (p_target, q_target), (loss, loss_details))
                for key, value in metric.value().items():
                    if "recall" in key:
                        continue
                    writer.add_scalar(
                        f"train_{key}", value, iteration
                    )

    batch_loss /= (batch_idx + 1)
    stats.update((epoch, batch_loss, None))


    add_logs_tensorboard(batch_loss_details, writer, (batch_idx + 1) ,len(data_loader) * (epoch + 1),"train")


    logger.info(
        f"TRAIN\t"
        f"Epoch: {epoch}/{epochs}\t"
        f"Iteration: {iteration}\t"
        f"Loss: {batch_loss:.4f}\t"
    )
    return batch_loss



def val_step(data_loader, model, criterion, epoch, epochs, writer, device, metrics, stats, args=None):

    logger = logging.getLogger()
    model.eval()

    for metric in metrics:
        metric.reset()
    embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
    with torch.no_grad():
        batch_loss = 0
        hm_score = 0
        seen_score=0
        unseen_score=0
        batch_loss_details={}
        for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
            p = data["positive"]
            q = data["negative"]

            x_p_a = p["audio"].to(device)
            x_p_v = p["video"].to(device)
            x_p_t = p["text"].to(device)
            x_p_num = target["positive"].to(device)

            masks = {}
            masks['positive'] = {'audio': p['audio_mask'], 'video': p['video_mask']}

            timesteps={}
            timesteps['positive']={'audio':p['timestep']['audio'], 'video':p['timestep']['video']}

            inputs = (
                x_p_a, x_p_v, x_p_num, x_p_t, masks['positive'], timesteps['positive']
            )

            if args.z_score_inputs:
                inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])

            if args.cross_entropy_loss == True:
                for i in range(inputs[2].shape[0]):
                    inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
            if args.cross_entropy_loss == True:
                loss, loss_details = model.optimize_params(*inputs, embedding_crossentropy=embeddings, optimize=False)
            else:
                loss, loss_details = model.optimize_params(*inputs, embedding_crossentropy=None, optimize=False)
            batch_loss_details = add_loss_details(loss_details, batch_loss_details)
            audio_emb, video_emb, emb_cls = model.get_embeddings(inputs[0], inputs[1], inputs[3], inputs[4], inputs[5])
            outputs = (video_emb, emb_cls)


            batch_loss += loss.item()

            p_target = target["positive"].to(device)
            q_target = target["negative"].to(device)

            # stats
            iteration = len(data_loader) * epoch + batch_idx
            if iteration % len(data_loader) == 0:
                for metric in metrics:
                    metric(outputs, (p_target, q_target), (loss, loss_details))
                    """
                    logger.info(
                        f"{metric.name()}: {metric.value()}"
                    )
                    """
                    for key, value in metric.value().items():
                        if "recall" in key:
                            continue
                        if "both_hm" in key:
                            hm_score = value
                            writer.add_scalar(
                                f"metric_val/{key}", value, iteration
                            )
                        if "both_zsl" in key:
                            zsl_score=value
                            writer.add_scalar(
                                f"metric_val/{key}", value, iteration
                            )
                        if "both_seen" in key:
                            seen_score=value
                            writer.add_scalar(
                                f"metric_val/{key}", value, iteration
                            )
                        if "both_unseen" in key:
                            unseen_score=value
                            writer.add_scalar(
                                f"metric_val/{key}", value, iteration
                            )

        batch_loss /= (batch_idx + 1)
        stats.update((epoch, batch_loss, hm_score))
        add_logs_tensorboard(batch_loss_details, writer, (batch_idx + 1), len(data_loader) * (epoch + 1),"val")
        logger.info(
            f"VALID\t"
            f"Epoch: {epoch}/{epochs}\t"
            f"Iteration: {iteration}\t"
            f"Loss: {batch_loss:.4f}\t"
            f"ZSL score: {zsl_score:.4f}\t"
            f"Seen score: {seen_score:.4f}\t"
            f"Unseen score:{unseen_score:.4f}\t"
            f"HM: {hm_score:.4f}"
        )
    return batch_loss, hm_score
