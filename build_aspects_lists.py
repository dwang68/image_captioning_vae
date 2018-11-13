import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
from json_helper import deserialize_from_file
import spacy
nlp = spacy.load('en')


class Aspects(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.humans = []
        self.non_humans = []


def check_if_human(anns):
    humans_sentence_ids = []
    non_humans_sentence_ids = []
    human_set = {"human", "person", "people", "man", "woman", "men", "women", "childen", "child", "boy", "girl"}
    for sid, sentence in anns.items():
        doc = nlp(sentence['caption'])
        sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
        if len(sub_toks) > 0:

            for sub_tok in sub_toks:
                if sub_tok.text in human_set:
                    humans_sentence_ids.append(sid)
                    break
                else:
                    non_humans_sentence_ids.append(sid)
                    break
        else:
            sub_toks = [tok for tok in doc if (tok.pos_ == "NOUN")]
            for sub_tok in sub_toks:
                if sub_tok.text in human_set:
                    humans_sentence_ids.append(sid)
                    break
                else:
                    non_humans_sentence_ids.append(sid)
                    break
    return humans_sentence_ids, non_humans_sentence_ids


def build_aspects(json, threshold):
    coco = COCO(json)
    humans_sid, non_humans_sid = check_if_human(coco.anns)
    aspects = Aspects()
    aspects.humans = humans_sid
    aspects.non_humans = non_humans_sid

    aspects_path = args.aspects_path
    with open(aspects_path, 'wb') as f:
        pickle.dump(aspects, f)
    print("Total humans aspect size: {}".format(len(aspects.humans)))
    print("Total non_humans aspect size: '{}'".format(len(aspects.non_humans)))
    print("Saved Aspects to: '{}'".format(aspects_path))


def main(args):

    aspects = build_aspects(json=args.caption_path, threshold=args.threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions_train2014.json',
                        help='path for train annotation file')
    parser.add_argument('--aspects_path', type=str, default='./data/aspects.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)