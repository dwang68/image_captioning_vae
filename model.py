import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class CVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                 max_sequence_length, sos_idx, eos_idx, pad_idx, unk_idx, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.vocab_size = vocab_size
        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size, momentum=0.01)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size + embedding_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, images, captions, lengths):
        batch_size = images.size()[0]
        with torch.no_grad():
            image_features = self.resnet(images)
        image_features = image_features.reshape(image_features.size(0), -1)
        image_features = self.bn(self.linear(image_features))

        # ENCODER
        input_embeddings = self.embedding(captions)

        encoder_input_embeddings = torch.cat((image_features.unsqueeze(1), input_embeddings), 1)
        packed = pack_padded_sequence(encoder_input_embeddings, lengths+1, batch_first=True)

        _, hidden = self.encoder_rnn(packed)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        z = torch.cat((z, image_features), dim=-1)


        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        decoder_input_embeddings = input_embeddings
        # decoder input
        if self.word_dropout_rate > 0:
            prob = torch.rand(captions.size())
            prob[(captions.data - self.sos_idx) * (captions.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = captions.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            decoder_input_embeddings = self.embedding(decoder_input_sequence)

        decoder_input_embeddings = self.embedding_dropout(decoder_input_embeddings)
        packed = pack_padded_sequence(decoder_input_embeddings, lengths, batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        temp = self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2)))
        logp = nn.functional.log_softmax(temp, dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z


    def inference(self, n=4, z=None, c=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        with torch.no_grad():
            c_features = self.resnet(c)
        c_features = c_features.reshape(c_features.size(0), -1)
        c_features = self.bn(self.linear(c_features))
        z = torch.cat((z, c_features), dim=-1)
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()
        #logp = self.tensor(batch_size, self.max_sequence_length, self.vocab_size).fill_(self.pad_idx).long()

        t = 0

        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)# unsqueeze to batch_size x 1 ...

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)
            #logp = self._save_sample(logp, logits, sequence_running, t)

            # update global running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                if input_sequence.size() == torch.Size([]):
                    input_sequence = input_sequence.reshape(1)
                if running_seqs.size() == torch.Size([]):
                    running_seqs = running_seqs.reshape(1)
                try:
                    input_sequence = input_sequence[running_seqs]
                except IndexError as err:
                    print("Caught index error" + repr(err))
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to