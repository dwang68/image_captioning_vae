import os
import torch
import argparse
import pickle
from torchvision import transforms
from PIL import Image
from build_vocab import Vocabulary

from model import CVAE
from device import device
from data_loader import get_loader
dataset_root_dir = "/home/dalinw/datasets/mscoco/mscoco1"

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224), Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    pad_idx = vocab.word2idx['<pad>']
    sos_idx = vocab.word2idx['<start>']
    eos_idx = vocab.word2idx['<end>']
    unk_idx = vocab.word2idx['<unk>']

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

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from {}".format(args.load_checkpoint))

    model.to(device)
    model.eval()

    # Build data loader
    train_data_loader, valid_data_loader = get_loader(args.train_image_dir, args.val_image_dir,
                                                      args.train_caption_path, args.val_caption_path, vocab,
                                                      args.batch_size,
                                                      shuffle=True, num_workers=args.num_workers)

    f1 = open('{}/results/generated_captions.txt'.format(dataset_root_dir), 'w')
    f2 = open('{}/results/ground_truth_captions.txt'.format(dataset_root_dir), 'w')
    for i, (images, captions, lengths) in enumerate(valid_data_loader):
        images = images.to(device)

        sampled_ids, z = model.inference(n=args.batch_size, c=images)

        sampled_ids_batches = sampled_ids.cpu().numpy()  # (batch_size, max_seq_length)
        captions = captions.cpu().numpy()

        # Convert word_ids to words
        for j, sampled_ids in enumerate(sampled_ids_batches):
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            generated_sentence = ' '.join(sampled_caption)
            generated_sentence = generated_sentence.rstrip()
            generated_sentence = generated_sentence.replace("\n", "")
            generated_sentence = "{0}\n".format(generated_sentence)
            if j == 0:
                print("RE: {}".format(generated_sentence))
            f1.write(generated_sentence)

        for g, ground_truth_ids in enumerate(captions):
            ground_truth_caption = []
            for word_id in ground_truth_ids:
                word = vocab.idx2word[word_id]
                ground_truth_caption.append(word)
                if word == '<end>':
                    break
            ground_truth_sentence = ' '.join(ground_truth_caption)
            ground_truth_sentence = ground_truth_sentence.rstrip()
            ground_truth_sentence = ground_truth_sentence.replace("\n", "")
            ground_truth_sentence = "{0}\n".format(ground_truth_sentence)
            if g == 0:
                print("GT: {}".format(ground_truth_sentence))
            f2.write(ground_truth_sentence)
        if i % 10 == 0:
            print("This is the {0}th batch".format(i))
    f1.close()
    f2.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str,
                        default='{}/models_image_captioning_cvae2_emb256_hid256_nl2/model-12-epoch.ckpt'.format(dataset_root_dir))
    parser.add_argument('--train_image_dir', type=str, default='{0}/resized2014'.format(dataset_root_dir),
                        help='directory for training resized images')
    parser.add_argument('--val_image_dir', type=str, default='{0}/val2014'.format(dataset_root_dir),
                        help='directory for validation images')
    parser.add_argument('--train_caption_path', type=str,
                        default='{0}/annotations/captions_train2014.json'.format(dataset_root_dir),
                        help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str,
                        default='{0}/annotations/captions_val2014.json'.format(dataset_root_dir),
                        help='path for validation annotation json file')
    parser.add_argument('--vocab_path', type=str, default='{}/vocab.pkl'.format(dataset_root_dir), help='path for vocabulary wrapper')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=32)
    parser.add_argument('-eb', '--embedding_size', type=int, default=256)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.1)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-nl', '--num_layers', type=int, default=2)
    parser.add_argument('-bi', '--bidirectional', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)





