import sentencepiece as spm
import numpy as np
import time

#python embed_bucc.py bucc-data-bin/de:bucc-data-bin/fr:bucc-data-bin/ru:bucc-data-bin/zh --path checkpoints/mgtcompressed-ep-en-de,en-fr,en-ru,en-zh-66666666-10-0.5-1.0-1-1-mgtsepcompressed1024-6layer2-1-1/checkpoint1.pt  --sentencepiece bucc-data-bin/bucc.all.model --bucc-file ../LASER/tasks/bucc/embed/bucc2018.fr-en.train.txt.en --task multi_vae --lang-pairs en-de,en-fr,en-ru,en-zh  --add-lang-tokens 1 --add-encoder-tokens 1 --compressed 1 --output-file temp

def embed(params, batcher, sentences):
    results = []
    for ii in range(0, len(sentences), params.batch_size):
        batch1 = sentences[ii:ii + params.batch_size]
        results.extend(batcher(params, batch1))
    return np.vstack(results)

def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def batcher(params, batch):
    new_batch = []
    for i in batch:
        p = i.strip().lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        new_batch.append(p)
    vecs = params.embedder.embed(new_batch, params.encoder)
    return vecs

def evaluate(embedder, args):

    sp = spm.SentencePieceProcessor()
    sp.Load(args.sentencepiece)

    from argparse import Namespace

    new_args = Namespace(batch_size=32, sp=sp, embedder=embedder,
                     encoder=args.eval_encoder)

    fin = open(args.bucc_file, 'r', errors='surrogateescape')
    fout = open(args.output_file, mode='wb')
    n = 0
    t = time.time()
    for sentences in buffered_read(fin, 10000):
        embed(new_args, batcher, sentences).tofile(fout)
        n += len(sentences)
        if n % 10000 == 0:
            print('\r - Encoder: {:d} sentences'.format(n), end='')
    print('\r - Encoder: {:d} sentences'.format(n), end='')
    t = int(time.time() - t)
    if t < 1000:
        print(' in {:d}s'.format(t))
    else:
        print(' in {:d}m{:d}s'.format(t // 60, t % 60))
    fin.close()
    fout.close()

def add_bucc_args(parser):
    group = parser.add_argument_group('Embed')

    # fmt: off
    group.add_argument('--bucc-file', default=None,
                       help='beam size')
    group.add_argument('--output-file', default=None,
                       help='number of hypotheses to output')

if __name__ == '__main__':

    from embed import Embedder
    from fairseq import options

    parser = options.get_generation_parser(interactive=True)
    add_bucc_args(parser)
    args = options.parse_args_and_arch(parser)

    embedder = Embedder(args)

    evaluate(embedder, args)
