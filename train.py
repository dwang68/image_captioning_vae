import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import CVAE
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
from scipy.special import expit, logit
dataset_root_dir = "/home/dalinw/datasets/mscoco/mscoco1"

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from device import device

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    

    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    pad_idx = vocab.word2idx['<pad>']
    sos_idx = vocab.word2idx['<start>']
    eos_idx = vocab.word2idx['<end>']
    unk_idx = vocab.word2idx['<unk>']
    
    # Build data loader
    train_data_loader, valid_data_loader = get_loader(args.train_image_dir, args.val_image_dir,
                                                      args.train_caption_path, args.val_caption_path, vocab, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            # return float(1 / (1 + np.exp(-k * (step - x0))))
            return float(expit(k * (step - x0)))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    nll = torch.nn.NLLLoss(ignore_index=pad_idx)

    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        nll_loss = nll(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return nll_loss, KL_loss, KL_weight

    # Build the models
    model = CVAE(
        vocab_size=len(vocab),
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        max_sequence_length=args.max_sequence_length,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        unk_idx=unk_idx
    )
    model.to(device)
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the models
    total_step = len(train_data_loader)
    step_for_kl_annealing = 0
    best_valid_loss = float("inf")
    patience = 0

    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(train_data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions_src = captions[:, :captions.size()[1]-1]
            captions_tgt = captions[:, 1:]
            captions_src = captions_src.to(device)
            captions_tgt = captions_tgt.to(device)
            lengths = lengths - 1
            lengths = lengths.to(device)
            
            # Forward, backward and optimize
            logp, mean, logv, z = model(images, captions_src, lengths)

            #loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, captions_tgt, lengths, mean, logv, args.anneal_function,
                                                   step_for_kl_annealing, args.k, args.x0)

            loss = (NLL_loss + KL_weight * KL_loss) / args.batch_size


            # backward + optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_for_kl_annealing += 1


            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                outputs = model._sample(logp)
                outputs = outputs.cpu().numpy()

                # Convert word_ids to words
                sampled_caption = []
                ground_truth_caption = []
                for word_id in outputs[-1]:
                    word = vocab.idx2word[word_id]
                    sampled_caption.append(word)
                    if word == '<end>':
                        break

                captions_tgt = captions_tgt.cpu().numpy()
                for word_id in captions_tgt[-1]:
                    word = vocab.idx2word[word_id]
                    ground_truth_caption.append(word)
                    if word == '<end>':
                        break
                reconstructed = ' '.join(sampled_caption)
                ground_truth = ' '.join(ground_truth_caption)
                print("ground_truth: {0} \n reconstructed: {1}\n".format(ground_truth, reconstructed))

                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'model-{}-{}.ckpt'.format(epoch+1, i+1)))

        torch.save(model.state_dict(), os.path.join(args.model_path, 'model-{}-epoch.ckpt'.format(epoch + 1)))

        valid_loss = 0

        #check against validation set and early stop if the validation score is not improving within patience period
        for j, (images, captions, lengths) in enumerate(valid_data_loader):
            # Set mini-batch dataset
            images = images.to(device)
            captions_src = captions[:, :captions.size()[1] - 1]
            captions_tgt = captions[:, 1:]
            captions_src = captions_src.to(device)
            captions_tgt = captions_tgt.to(device)
            lengths = lengths - 1
            lengths = lengths.to(device)

            # Forward, backward and optimize
            logp, mean, logv, z = model(images, captions_src, lengths)

            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, captions_tgt, lengths, mean, logv, args.anneal_function,
                                                   step_for_kl_annealing, args.k, args.x0)

            valid_loss += (NLL_loss + KL_weight * KL_loss) / args.batch_size

            if j == 2:
                break
        print("validation loss for epoch {}: {}".format(epoch+1, valid_loss))
        print("patience is at {}".format(patience))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience = 0
        else:
            patience += 1

        if patience == 5:
            print("early stopping at epoch {}".format(epoch+1))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default='{0}/models_image_captioning_cvae2_emb256_hid256_nl2/'.format(dataset_root_dir),
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='{0}/vocab.pkl'.format(dataset_root_dir),
                        help='path for vocabulary wrapper')
    parser.add_argument('--train_image_dir', type=str, default='{0}/resized2014'.format(dataset_root_dir),
                        help='directory for training resized images')
    parser.add_argument('--val_image_dir', type=str, default='{0}/val2014'.format(dataset_root_dir),
                        help='directory for validation images')
    parser.add_argument('--train_caption_path', type=str, default='{0}/annotations/captions_train2014.json'.format(dataset_root_dir),
                        help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str,
                        default='{0}/annotations/captions_val2014.json'.format(dataset_root_dir),
                        help='path for validation annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--max_sequence_length', type=int, default=32)
    parser.add_argument('--embedding_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')

    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-bi', '--bidirectional', default=False, action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.1)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)  # k is steepness of the logistic function
    parser.add_argument('-x0', '--x0', type=int, default=2500)  # x0 is the mid-point, and for linear function, this is the denominator

    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    args = parser.parse_args()
    print(args)
    main(args)
