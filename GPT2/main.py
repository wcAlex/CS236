import sys
from datasets.neurips_dataset import NIPS2015Dataset
from datasets.encoder import get_codec
from model import GPT2, load_weight
from utils import *
import requests
import torch
import pickle as pkl
from tqdm import trange
import numpy as np
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def setup():
    args = parse_args()
    config = parse_config(args)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)

    paper_dataset = NIPS2015Dataset(data_folder='datasets')

    codec = get_codec()
    model = GPT2(config)
    if not os.path.exists('gpt2-pytorch_model.bin'):
        print("Downloading GPT-2 checkpoint...")
        url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin'
        r = requests.get(url, allow_redirects=True)
        open('gpt2-pytorch_model.bin', 'wb').write(r.content)

    model = load_weight(model, torch.load('gpt2-pytorch_model.bin', map_location=device))
    model = model.to(device)
    model.eval()
    if not os.path.exists('submit'):
        os.makedirs('submit')

    return codec, model, paper_dataset, config


def plot_log_p(filename, codec, model):
    from questions.likelihood import log_likelihood
    with open(os.path.join('datasets', filename + '.pkl'), 'rb') as f:
        lls = []
        data = pkl.load(f)
        for i in trange(len(data)):
            text = data[i]
            text = codec.encode(text).to(device)
            ## TODO: complete the code in the function `log_likelihood`
            lls.append(log_likelihood(model, text))
        lls = np.asarray(lls)

    with open(os.path.join('submit', filename + '_raw.pkl'), 'wb') as f:
        pkl.dump(lls, f, protocol=pkl.HIGHEST_PROTOCOL)

    plt.figure()
    plt.hist(lls)
    plt.xlabel('Log-likelihood')
    plt.xlim([-600, 0])
    plt.ylabel('Counts')
    plt.title(filename)
    plt.savefig(os.path.join('submit', filename + '.png'), bbox_inches='tight')
    plt.show()
    plt.close()
    print("# Figure written to %s.png." % filename)


def main():
    codec, model, paper_dataset, config = setup()

    # question 3)
    print("Question 3)")
    from questions.sample import sample
    paper_iter = iter(paper_dataset)
    with open(os.path.join('submit', 'samples.txt'), 'w', encoding='utf-8') as f:
        for i in range(5):
            ## Use paper abstracts as the starting text
            start_text = next(paper_iter)['abstract'][:100]
            start_text = codec.encode(start_text).to(device)
            ## TODO: Complete the code of the following function
            text = sample(model, start_text, config)
            ## Decode samples
            text = codec.decode(text.tolist()[0])
            f.write('=' * 50 + " SAMPLE_{} ".format(i) + '=' * 50 + '\n')
            f.write(text + '\n')
    print("# Samples written to samples.txt.")

    # question 4)
    print("Question 4)")
    plot_log_p('random', codec, model)
    plot_log_p('shakespeare', codec, model)
    plot_log_p('neurips', codec, model)

    # question 5)
    print("Question 5)")
    from questions.classifier import classification

    with open(os.path.join('datasets', 'snippets.pkl'), 'rb') as f:
        snippets = pkl.load(f)
    lbls = []
    for snippet in snippets:
        ## TODO: complete the code in function `classification`
        lbls.append(classification(model, codec.encode(snippet).to(device)))

    with open(os.path.join("submit", "classification.pkl"), 'wb') as f:
        pkl.dump(lbls, f, protocol=pkl.HIGHEST_PROTOCOL)

    return 0

if __name__ == '__main__':
    sys.exit(main())
