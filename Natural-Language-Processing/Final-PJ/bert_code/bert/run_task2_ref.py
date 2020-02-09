import random
import numpy as np
import argparse
import shutil
from tqdm import trange
import torch
from tqdm import tqdm

from transformers import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils_task2_ref import *
from model.bertmodel_ref import Task2Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, tokenizer, test=False):
    wrong_samples = []
    results = {}
    eval_dataset = load_examples_and_cache_features(data_dir=args.data_dir, data_name=args.data_name,
                                                    tokenizer=tokenizer, max_seq_length=args.max_seq_length, max_ref_length=args.max_ref_length,
                                                    dev=not test, test=test,trp_num = args.trp_num)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            # inputs = {"input_ids": batch[0],
            #           "attention_mask": batch[1],
            #           "token_type_ids": batch[2],
            #           "labels": batch[3],
            #           "guids": batch[4]}
            inputs = {
                "sent_input_ids": batch[0],
                "sent_attention_masks": batch[1],
                "sent_token_type_ids": batch[2],
                "refer_input_ids": batch[3],
                "refer_attention_masks": batch[4],
                "refer_token_type_ids": batch[5],
                "labels": batch[6],
                "guids": batch[7]
            }
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            if args.n_gpu > 1:
                loss = loss.mean()
            eval_loss += loss
        nb_eval_steps += 1

        pred = logits.detach().cpu().numpy()
        pred = np.argmax(pred, axis=1)
        out_label_id = inputs["labels"].detach().cpu().numpy()
        if preds is None:
            preds = pred
            out_label_ids = out_label_id
        else:
            preds = np.append(preds, pred, axis=0)
            out_label_ids = np.append(out_label_ids, out_label_id, axis=0)


        compare = pred == out_label_id
        for i in range(len(compare)):
            if not compare[i]:
                wrong_samples.append(inputs["guids"].detach().cpu().numpy()[i])

    eval_loss = eval_loss / nb_eval_steps
    acc = simple_accuracy(preds, out_label_ids)
    result = {"eval_acc": acc, "eval_loss": eval_loss}
    results.update(result)

    return results, wrong_samples




def train(args, train_dataset, model, tokenizer):
    # Overwrite the content of the output directory,
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        os.mkdir(args.output_dir)
    else:
        os.mkdir(args.output_dir)

    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer=optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    # Multi-gpu setting
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Instanteneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (parallel & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss, best_step = 0.0, 9999999999.0, 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # inputs = {"input_ids":      batch[0],
            #           "attention_mask": batch[1],
            #           "token_type_ids": batch[2],
            #           "labels":         batch[3],
            #           "guids":          batch[4]}
            #inputs = {
            #    "sent_input_ids":       batch[0],
            #   "sent_attention_masks": batch[1],
            #    "sent_token_type_ids":  batch[2],
             #   "trp_input_ids":        batch[3],
             #   "trp_attention_masks":  batch[4],
             #   "trp_token_type_ids":   batch[5],
             #   "labels":               batch[6],
             #   "guids":                batch[7]
            #}
            inputs = {
                "sent_input_ids": batch[0],
                "sent_attention_masks": batch[1],
                "sent_token_type_ids": batch[2],
                "refer_input_ids": batch[3],
                "refer_attention_masks": batch[4],
                "refer_token_type_ids": batch[5],
                "labels": batch[6],
                "guids": batch[7]
            }
           # print(inputs)
           # exit(886)

            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if ((step+1) % args.gradient_accumulation_steps) == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # Save model checkpoints if it has lower loss
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Eval on the dev set
                    results, _ = evaluate(args, model, tokenizer) # dev set evaluation
                    logger.info("***** Step %s: evaluate dev set results", str(global_step))
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        logger.info("      %s = %s", key, value)
                    if results["eval_loss"] < best_dev_loss:
                        best_dev_acc = results["eval_acc"]
                        best_dev_loss = results["eval_loss"]
                        best_step = global_step
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("train_loss", (tr_loss-logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    outputs_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(outputs_dir):
                        os.makedirs(outputs_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(outputs_dir)
                    logger.info("Checkpoint {} saved at".format(global_step))
                    torch.save(args, os.path.join(outputs_dir, "training_args.bin"))

    logger.info("Min dev loss: %s, dev acc: %s, best global step: %s", str(best_dev_loss), str(best_dev_acc), str(best_step))
    tb_writer.close()
    return global_step, tr_loss / global_step, best_step





def main():
    parser = argparse.ArgumentParser()
    # File directory
    parser.add_argument("--data_dir", default="./data/task2/sent_triple/", type=str, help="Input datadir. Should contain the .tsv files for the task.")
    parser.add_argument("--data_name", default="ref", type=str, help="Prefix of tsv files. If default, then train.tsv et.")
    parser.add_argument("--output_dir", default="./model/fine_tuned/", type=str, help="Output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--wrong_file", default="./eval_result/task2.csv", type=str, help="Save the wrong predicted samples.")

    # Checkpoint saving setting
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoints every X update steps.")

    # Model parameters setting
    parser.add_argument("--num_train_epochs", default=4.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_seq_length", default=60, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_ref_length", default=30, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--trp_num", default=1, type=int, help="Maximum number of triples in one answer")

    parser.add_argument("--n_gpu", default=n_gpu, type=int, help="Num of gpus to use. Equal to the length of CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--per_gpu_train_batch_size", default=5, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=5, type=int, help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # Train or evaluate setting
    parser.add_argument("--do_test", default=True, help='Whether to run test on the test set')
    parser.add_argument("--eval_step", default=None, type=int, help="If don't run, only test the dev/test set on the specifix checkpoint model w.r.t eval_step.")
    parser.add_argument("--training", default=True, help="Attention batch")
    parser.add_argument("--trp_attn", default=True)

    args = parser.parse_args()

    # Set Up CUDA and GPU device
    device = torch.device("cuda")
    args.device = device
    logger.info("Training/evaluation parameters %s", args)

    set_seed(args)

    # Load pretrained Bert model and tokenizer
    config = BertConfig.from_pretrained("./model/pretrained_bert/bert-base-uncased/bert-base-uncased-config.json")
    tokenizer = BertTokenizer.from_pretrained("./model/pretrained_bert/bert-base-uncased/bert-base-uncased-vocab.txt", do_lower_case=True)
    model = Task2Model.from_pretrained("./model/pretrained_bert/bert-base-uncased/bert-base-uncased-pytorch_model.bin", config=config, args=args)
    model.to(device)

    # Start Traning and save the best practice. Eval the specific checkpoint only when eval_step is not None.
    if args.eval_step is None:
        train_dataset = load_examples_and_cache_features(data_dir=args.data_dir, data_name=args.data_name, tokenizer=tokenizer, max_seq_length=args.max_seq_length, max_ref_length=args.max_ref_length,trp_num = args.trp_num)
        global_step, tr_loss, best_step = train(args, train_dataset, model, tokenizer)
        logger.info(" Training global_step = %s", global_step)
        logger.info(" Training average loss = %s", tr_loss)
        logger.info(" Best global step = %s", best_step)
        args.eval_step = best_step
    else:
        pass

    # Evaluation on the test dataset with the best checkpoint from train or manually specified
    if args.do_test:
        checkpoint = os.path.join(args.output_dir, "checkpoint-{}".format(str(args.eval_step)))
        model_eval = Task2Model.from_pretrained(checkpoint, config=config, args=args)
        model_eval.to(args.device)
        test_results, wrong_samples = evaluate(args, model_eval, tokenizer, test=True)
        logger.info("Corresponding test loss: %s, test acc: %s, eval step: %s", str(test_results["eval_loss"]),
                    str(test_results["eval_acc"]), str(args.eval_step))

        logger.info("Save %d wrong samples into file: %s", len(wrong_samples), str(args.wrong_file))
        # record_wrong(wrong_samples, args)


if __name__ == '__main__':
    main()

