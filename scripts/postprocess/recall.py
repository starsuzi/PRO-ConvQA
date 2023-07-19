import argparse
import json
import re
import unicodedata
from collections import defaultdict
from tqdm import tqdm
#from scripts.preprocess.simple_tokenizer import SimpleTokenizer



# TODO
# https://github.com/facebookresearch/DrQA/blob/master/drqa/tokenizers/simple_tokenizer.py#L18

#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Basic tokenizer that splits text into alpha-numeric tokens and
non-whitespace tokens.
"""

import copy
import regex
import logging

logger = logging.getLogger(__name__)


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """
    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()

class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def read_file(infile, handle_file, log=False, skip_first_line=False):
    if log:
        print('Opening "{}"...'.format(infile))
    data = None
    with open(infile) as f:
        if skip_first_line:
            f.readline()
        data = handle_file(f)
    if log:
        print('  Done.')
    return data


def read_jsonl(infile, log=False):
    handler = lambda f: [json.loads(line) for line in f.readlines()]
    return read_file(infile, handler, log=log)


def read_json(infile, log=False):
    handler = lambda f: json.load(f)
    return read_file(infile, handler, log=log)


def _normalize(text):
    return unicodedata.normalize('NFD', text)

###############################################################################
### HAS_ANSWER FUNCTIONS   ####################################################
###############################################################################
def has_answer_field(ctx, answers):
    return ctx['has_answer']


tokenizer = SimpleTokenizer(**{})
def string_match(ctx, answers):
    text = tokenizer.tokenize(ctx['text']).words(uncased=True)

    for single_answer in answers:
        single_answer = _normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)

        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True
    return False


def normalized_title(ctx, answers):
    for answer in answers:
        a = a.lower().strip()
        title = ctx['title'].lower().strip()
        if a == title[:len(a)]:
            return True
    return False


def regex(ctx, answers):
    text = ctx['text']
    for answer in answers:
        answer = _normalize(answer)
        if regex_match(text, answer):
            return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


###############################################################################
### CALCULATION FUNCTIONS   ###################################################
###############################################################################
def precision_fn(results, k_vals, has_answer):
    n_hits = {k: 0 for k in k_vals}
    mrrs = []
    precs = []
    PREC_K = 10
    MRR_K = 10

    for result in tqdm(results):
        ans = result['answers']
        ctxs = result['ctxs']
        found_k = len(ctxs) + 1
        found = False
        num_hit = 0
        for c_idx,ctx in enumerate(ctxs):
            if has_answer(ctx, ans):
                if not found:
                    found_k = c_idx # record first one
                found = True

                if c_idx < PREC_K: # P@k
                    num_hit += 1
                # break
        for k in k_vals:
            if found_k < k:
                n_hits[k] += 1

        if found_k >= MRR_K:
            mrrs.append(0)
        else:   
            mrrs.append(1/(found_k + 1))
        precs.append(num_hit/PREC_K)
    
    print('*'*50)
    for k in k_vals:
        if len(results) == 0:
            print('No results.')
        else:
            print('Top-{} = {:.2%}'.format(k, n_hits[k] / len(results)))

    print(f'Acc@{k_vals[0]} when Acc@{k_vals[-1]} = {n_hits[k_vals[0]]/n_hits[k_vals[-1]]*100:.2f}%')
    print(f'MRR@{MRR_K} = {sum(mrrs)/len(mrrs)*100:.2f}')
    print(f'P@{PREC_K} = {sum(precs)/len(precs)*100:.2f}')


def precision_fn_file(infile, n_docs, k_vals, has_answer, args):
    results = read_jsonl(infile) if args.jsonl else read_json(infile)

    # stats
    ctx_lens = [sum([len(pp['text'].split()) for pp in re['ctxs']])/len(re['ctxs']) for re in results]
    print(f'ctx token length: {sum(ctx_lens)/len(ctx_lens):.2f}')

    # unique titles
    title_lens = [len(set(pp['title'] for pp in re['ctxs'])) for re in results]
    print(f'unique titles: {sum(title_lens)/len(title_lens):.2f}')

    precision_fn(results, k_vals, has_answer)


# Top-20 and Top-100
def precision_per_bucket(results_file, longtail_file, n_docs, k_vals, longtail_tags, ans_fn):
    results = read_json(results_file)
    annotations = read_json(longtail_file)
    for tag in longtail_tags:
        bucket = [result for idx,result in enumerate(results) if tag == annotations[idx]['annotations']]
        print('==== Bucket={} ====='.format(tag))
        precision_fn(bucket, n_docs, k_vals, ans_fn)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', required=True, type=str, default=None,
                        help="Location of the results file to parse.")
    parser.add_argument('--n_docs', type=int, default=100,
                        help="Maximum number of docs retrieved.")
    parser.add_argument('--k_values', type=str, default='1,5,10,20,40,50,60,80,100',
                        help="Top-K values to print out")
    parser.add_argument('--ans_fn', type=str, default='has_answer',
                        help="How to check whether has the answer. title | has_answer")
    parser.add_argument('--jsonl', action='store_true', help='Set if results is a jsonl file.')

    # Longtail Entity Analysis
    parser.add_argument('--longtail', action='store_true',
                        help='whether or not to include longtail buckets')
    parser.add_argument('--longtail_file', required=False, type=str, default=None,
                        help='Mapping from question to longtail entity tags.')
    parser.add_argument('--longtail_tags', type=str, default='p10,p25,p50,p75,p90',
                        help='Tags for the longtail entities within longtail_file')

    args = parser.parse_args()
    ks = [int(k) for k in args.k_values.split(',')]
    if args.ans_fn == 'has_answer':
        ans_fn = has_answer_field
    elif args.ans_fn == 'title':
        ans_fn = normalized_title
    elif args.ans_fn == 'string':
        ans_fn = string_match
    elif args.ans_fn == 'regex':
        ans_fn = regex
    else:
        raise Exception('Answer function not recognized')
    
    if args.longtail:
        longtail_tags = args.longtail_tags.split(',')
        precision_per_bucket(args.results_file, args.longtail_file, 
            args.n_docs, ks, longtail_tags, ans_fn)
    else:
        precision_fn_file(args.results_file, args.n_docs, ks, ans_fn, args)
