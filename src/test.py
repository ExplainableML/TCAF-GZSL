import logging

from src.utils import evaluate_dataset_baseline


def test(eval_name, val_dataset, test_dataset, model_A, model_B, device, distance_fn,test_stats, eval_dir,
        args, new_model_sequence=False,save_performances=False):
    logger = logging.getLogger()
    model_A.eval()
    model_B.eval()

    test_evaluation = _get_test_performance(val_dataset=val_dataset, test_dataset=test_dataset, model_A=model_A,
                                            model_B=model_B, device=device, distance_fn=distance_fn,
                                            args=args,
                                            new_model_sequence=new_model_sequence,
                                            save_performances=save_performances)

    if args.dataset_name == "AudioSetZSL":
        output_string = fr"""
                   Seen performance={100 * test_evaluation["both"]["seen"]:.2f}, Unseen performance={100 * test_evaluation["both"]["unseen"]:.2f}, GZSL performance={100 * test_evaluation["both"]["hm"]:.2f}, ZSL performance={100 * test_evaluation["both"]["zsl"]:.2f} 
                   """
    elif args.dataset_name == "VGGSound" or args.dataset_name == "UCF" or args.dataset_name == "ActivityNet":
        output_string = fr"""
                    Seen performance={100 * test_evaluation["both"]["seen"]:.2f}, Unseen performance={100 * test_evaluation["both"]["unseen"]:.2f}, GZSL performance={100 * test_evaluation["both"]["hm"]:.2f}, ZSL performance={100 * test_evaluation["both"]["zsl"]:.2f} 
                    """
    else:
        raise NotImplementedError()

    logger.info(output_string)

    return test_evaluation


def _get_test_performance(val_dataset, test_dataset, model_A, model_B, device, distance_fn,
                           args, new_model_sequence,
                          save_performances=False):
    logger = logging.getLogger()

    val_evaluation = evaluate_dataset_baseline(val_dataset, model_A, device, distance_fn,
                                               args=args,
                                               new_model_sequence=new_model_sequence,
                                              )

    best_beta_combined = 1. / 3 * (
            val_evaluation['audio']['beta'] + val_evaluation['video']['beta'] + val_evaluation['both']['beta']+1e-10)
    logger.info(
        f"Validation betas:\tAudio={val_evaluation['audio']['beta']}\tVideo={val_evaluation['video']['beta']}\tBoth={val_evaluation['both']['beta']}")
    logger.info(f"Best beta combined: {best_beta_combined}")


    test_evaluation = evaluate_dataset_baseline(test_dataset, model_B, device, distance_fn, best_beta=best_beta_combined,
                                                 args=args,
                                                 new_model_sequence=new_model_sequence,
                                                 save_performances=save_performances)

    return test_evaluation
