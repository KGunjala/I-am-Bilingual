import os
import math
import wget
import torch

import tarfile
import shutil
import codecs
import youtokentome



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



''' Helper function to download the training, validation, and test files for translation. '''

def download_data(data_folder):

    train_urls = ["http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
                  "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
                  "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"]

    
    # If it doesn't exist, create a folder to store the downloaded files
    
    if not os.path.isdir(os.path.join(data_folder, "tar files")):
        os.mkdir(os.path.join(data_folder, "tar files"))

        
    # If it doesn't exist, create a new folder to extract downloaded files
    # Previous extractions deleted to prevent tarfile module errors
    
    if os.path.isdir(os.path.join(data_folder, "extracted files")):
        shutil.rmtree(os.path.join(data_folder, "extracted files"))
        os.mkdir(os.path.join(data_folder, "extracted files"))

        
    # Download and extract training data
    
    for url in train_urls:
        filename = url.split("/")[-1]
        if not os.path.exists(os.path.join(data_folder, "tar files", filename)):
            # Downloading the file
            wget.download(url, os.path.join(data_folder, "tar files", filename))
        
        # Extracting the file
        tar = tarfile.open(os.path.join(data_folder, "tar files", filename))
        members = [m for m in tar.getmembers() if "de-en" in m.path]
        tar.extractall(os.path.join(data_folder, "extracted files"), members=members)

    # Download validation and testing data using sacreBLEU since we will be using this library to calculate BLEU scores
    print("\n")
    os.system("sacrebleu -t wmt13 -l en-de --echo src > '" + os.path.join(data_folder, "val.en") + "'")
    os.system("sacrebleu -t wmt13 -l en-de --echo ref > '" + os.path.join(data_folder, "val.de") + "'")
    print("\n")
    os.system("sacrebleu -t wmt14/full -l en-de --echo src > '" + os.path.join(data_folder, "test.en") + "'")
    os.system("sacrebleu -t wmt14/full -l en-de --echo ref > '" + os.path.join(data_folder, "test.de") + "'")

    # Move files if they were extracted into a subdirectory
    for dir in [d for d in os.listdir(os.path.join(data_folder, "extracted files")) if
                os.path.isdir(os.path.join(data_folder, "extracted files", d))]:
        for f in os.listdir(os.path.join(data_folder, "extracted files", dir)):
            shutil.move(os.path.join(data_folder, "extracted files", dir, f),
                        os.path.join(data_folder, "extracted files"))
        os.rmdir(os.path.join(data_folder, "extracted files", dir))



''' Helper function to filter and prepare the training data, and train a Byte-Pair Encoding (BPE) model. '''

def prepare_data(data_folder, euro_parl=True, common_crawl=True, news_commentary=True, min_length=3, max_length=100,
                 max_length_ratio=1.5, retain_case=True):
    """
    :param data_folder: the folder where the files were downloaded
    :param euro_parl: include the Europarl v7 dataset in the training data?
    :param common_crawl: include the Common Crawl dataset in the training data?
    :param news_commentary: include theNews Commentary v9 dataset in the training data?
    :param min_length: exclude sequence pairs where one or both are shorter than this minimum BPE length
    :param max_length: exclude sequence pairs where one or both are longer than this maximum BPE length
    :param max_length_ratio: exclude sequence pairs where one is much longer than the other
    :param retain_case: retain case? """
    
    # Read raw files and combine
    german = list()
    english = list()
    files = list()
    assert euro_parl or common_crawl or news_commentary, "Set at least one dataset to True!"
    if euro_parl:
        files.append("europarl-v7.de-en")
    if common_crawl:
        files.append("commoncrawl.de-en")
    if news_commentary:
        files.append("news-commentary-v9.de-en")
    print("\nReading extracted files and combining...")
    for file in files:
        with codecs.open(os.path.join(data_folder, "extracted files", file + ".de"), "r", encoding="utf-8") as f:
            if retain_case:
                german.extend(f.read().split("\n"))
            else:
                german.extend(f.read().lower().split("\n"))
        with codecs.open(os.path.join(data_folder, "extracted files", file + ".en"), "r", encoding="utf-8") as f:
            if retain_case:
                english.extend(f.read().split("\n"))
            else:
                english.extend(f.read().lower().split("\n"))
        assert len(english) == len(german)

    # Write to file so stuff can be freed from memory
    print("\nWriting to single files...")
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english))
    with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding="utf-8") as f:
        f.write("\n".join(german))
    with codecs.open(os.path.join(data_folder, "train.ende"), "w", encoding="utf-8") as f:
        f.write("\n".join(english + german))
    del english, german  # free some RAM

    # Perform BPE
    print("\nLearning BPE...")
    youtokentome.BPE.train(data=os.path.join(data_folder, "train.ende"), vocab_size=37000,
                           model=os.path.join(data_folder, "bpe.model"))

    # Load BPE model
    print("\nLoading BPE model...")
    bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

    # Re-read English, German
    print("\nRe-reading single files...")
    with codecs.open(os.path.join(data_folder, "train.en"), "r", encoding="utf-8") as f:
        english = f.read().split("\n")
    with codecs.open(os.path.join(data_folder, "train.de"), "r", encoding="utf-8") as f:
        german = f.read().split("\n")

    # Filter
    print("\nFiltering...")
    pairs = list()
    for en, de in zip(english, german):
        en_tok = bpe_model.encode(en, output_type=youtokentome.OutputType.ID)
        de_tok = bpe_model.encode(de, output_type=youtokentome.OutputType.ID)
        len_en_tok = len(en_tok)
        len_de_tok = len(de_tok)
        if min_length < len_en_tok < max_length and \
                min_length < len_de_tok < max_length and \
                1. / max_length_ratio <= len_de_tok / len_en_tok <= max_length_ratio:
            pairs.append((en, de))
        else:
            continue
    print("\nNote: %.2f per cent of en-de pairs were filtered out based on sub-word sequence length limits." % (100. * (
            len(english) - len(pairs)) / len(english)))

    # Rewrite files
    english, german = zip(*pairs)
    print("\nRe-writing filtered sentences to single files...")
    os.remove(os.path.join(data_folder, "train.en"))
    os.remove(os.path.join(data_folder, "train.de"))
    os.remove(os.path.join(data_folder, "train.ende"))
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english))
    with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding="utf-8") as f:
        f.write("\n".join(german))
    del english, german, bpe_model, pairs

    print("\n...DONE!\n")


    
''' Helper Function to compute positional encoding as defined in the paper (Attention is all you need) '''

def get_positional_encoding(d_model, max_length=100):
    
    """
    :param d_model: size of vectors throughout the transformer model
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """
    
    positional_encoding = torch.zeros((max_length, d_model))  # (max_length, d_model)
    for i in range(max_length):
        for j in range(d_model):
            if j % 2 == 0:
                positional_encoding[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else:
                positional_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

    positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)

    return positional_encoding



''' Helper function to set the Learning Rate schedule. This version below is twice the definition in the paper (Attention is all you need, as used in the official T2T repository. '''

def get_lr(step, d_model, warmup_steps):
    
    """
    :param step: training step number
    :param d_model: size of vectors throughout the transformer model
    :param warmup_steps: number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official T2T repo
    :return: updated learning rate
    """
    
    lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))

    return lr



''' Helper function to save Checkpoints. Each save overwrites previous save. '''

def save_checkpoint(epoch, model, optimizer, prefix=''):
    
    """
    :param epoch: epoch number (0-indexed)
    :param model: transformer model
    :param optimizer: optimized
    :param prefix: checkpoint filename prefix
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = prefix + 'transformer_checkpoint.pth.tar'
    torch.save(state, filename)



''' Helper function to scale learning rate by a specified factor. '''

def change_lr(optimizer, new_lr):
    
    """
    :param optimizer: optimizer whose learning rate must be changed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


        
        
''' A class to keep track of most recent, average, sum, and count of a metric. '''

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
