import sys
import io
import numpy as np
import logging
import os

#PATH_TO_BGT = '/projects/tir5/users/jwieting/bi-vae/bilingual-1'
CODE_DIR = "/projects/tir5/users/jwieting/multi-single-crosslingual3"

# Set PATHs
PATH_TO_SENTEVAL = '/projects/tir3/users/jwieting/vae-expts2/analysis2/SentEval/'
PATH_TO_DATA = '/projects/tir3/users/jwieting/vae-expts2/analysis2/SentEval/data'
PATH_TO_MY_DATA = '/projects/tir3/users/jwieting/vae-expts2/analysis2/'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
sys.path.insert(0, CODE_DIR)
import senteval

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    pass

def batcher(params, batch):
    batch = [" ".join(s) for s in batch]
    new_batch = []
    for i in batch:
        p = i.lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        new_batch.append(p)
    vecs = params.embedder.embed(new_batch, params.encoder, lang="en", lang_emb=params.lang_emb)
    return vecs

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32, 'tenacity': 3, 'epoch_size': 1}
#params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam,lr=0.001', 'batch_size': 32,
#                                 'tenacity': 3, 'epoch_size': 1}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

from senteval.probing import PROBINGEval

class POSCountEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'en-total-punct.txt')
        # labels: 'O', 'C'
        PROBINGEval.__init__(self, 'POSCount', task_path, seed)

class POSFirstEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'en-first-punct.txt')
        # labels: 'O', 'C'                                                                                                                                                                                                                                          
        PROBINGEval.__init__(self, 'POSFirst', task_path, seed)

class ENLengthEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'en-length.txt')
        # labels: 'O', 'C'                                                                                                                                                                                                                       
        PROBINGEval.__init__(self, 'ENLength', task_path, seed)

if __name__ == "__main__":
    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber']
                      #'OddManOut', 'CoordinationInversion']

    #transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                  'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                  'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber']

    from embed import Embedder
    from fairseq import options
    import sentencepiece as spm
    from senteval import utils

    os.chdir(CODE_DIR)

    parser = options.get_generation_parser(interactive=True)
    options.add_embed_args(parser)
    args = options.parse_args_and_arch(parser)
    params = utils.dotdict(params_senteval)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.sentencepiece)

    embedder = Embedder(args)

    params.batch_size = 32
    params.sp = sp
    params.embedder = embedder
    params.encoder = args.eval_encoder
    params.lang_emb = args.lang_emb

    results = {}
    s = ENLengthEval(PATH_TO_MY_DATA)
    s.do_prepare(params, prepare)
    results.update(s.run(params, batcher))

    s = POSCountEval(PATH_TO_MY_DATA)
    s.do_prepare(params, prepare)
    results.update(s.run(params, batcher))

    s = POSFirstEval(PATH_TO_MY_DATA)
    s.do_prepare(params, prepare)
    results.update(s.run(params, batcher))

    gc_results = results

    se = senteval.engine.SE(params, batcher, prepare)

    results = se.eval(transfer_tasks)

    for i in gc_results:
        print(i, gc_results[i])

    for i in results:
        print(i, results[i])
