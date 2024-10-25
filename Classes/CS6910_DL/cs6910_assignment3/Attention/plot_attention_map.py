import os
import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from typing import *
import torch
import dataset
import model
import config
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_word(source: torch.Tensor,  # type: torch.Tensor
             pred: torch.Tensor,  # type: torch.Tensor
             corpus: dataset.Corpus,  # type: dataset.Corpus
             ) -> Tuple[List[str], List[str]]:
    """
    Convert the input and predicted word from the model to a list of strings.

    Args:
        source: The input word as a tensor.
        pred: The predicted word as a tensor.
        corpus: The corpus used for the input and output word representations.

    Returns:
        A tuple containing the list of input words as strings and the list of predicted words as strings.
    """
    source_words = []
    pred_words = []

    for word in source.cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(corpus.source_int2char[letter])

        temp_word = ''.join(temp_word)
        source_words.append(temp_word)

    for word in pred.cpu().numpy():
        temp_word = []
        for letter in word:
            if letter == 1:
                break
            temp_word.append(corpus.target_int2char[letter])

        temp_word = ''.join(temp_word)
        pred_words.append(temp_word)

    return source_words, pred_words


def get_attention_map(encoder: torch.nn.Module,  # type: torch.nn.Module
                      decoder: torch.nn.Module,  # type: torch.nn.Module
                      dataloader: torch.utils.data.DataLoader,  # type: torch.utils.data.DataLoader
                      ) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Gets the attention map for a single batch of data from the given dataloader.

    Args:
        encoder: The encoder model to use for encoding the input.
        decoder: The decoder model to use for generating the output.
        dataloader: The data loader to use for getting the input and output data.

    Returns:
        A tuple containing the list of input words as strings, the list of predicted words as strings,
        and the attention map as a numpy array.
    """
    c = dataset.Corpus(lang='ben', type='test')

    for data in dataloader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        encoder_outputs, encoder_hidden = encoder(inputs)
        decoder_outputs, decoder_hidden, attention_map = decoder(
            encoder_outputs, encoder_hidden)
        outputs = decoder_outputs.argmax(-1)

        source, pred = get_word(inputs, outputs, c)
        break

    return source, pred, attention_map.detach().cpu().numpy()


def plot_attention(input_word: List[str],
                   predicted_word: List[str],
                   attention_map: np.ndarray,
                   file_name: str = None) -> None:
    """
    Plots the attention map for a single batch of data.

    Args:
        input_word: The input word as a list of strings.
        predicted_word: The predicted word as a list of strings.
        attention_map: The attention map as a numpy array.
        file_name: The file name to save the plot to.
    """
    path = os.path.join(os.getcwd(), 'Attention_Heatmap', file_name)
    prop = fm.FontProperties(fname=os.path.join(
        os.getcwd(), 'Attention_Heatmap', 'Kalpurush.ttf'))

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention_map)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + list(input_word), fontdict=fontdict, rotation=0)
    ax.set_yticklabels([''] + list(predicted_word),
                       fontdict=fontdict, fontproperties=prop)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(path)
    with wandb.init(project="CS23E001_DL_3", name=file_name):
        wandb.log({'Attention Heatmap': wandb.Image(plt)})
        # plt.show()
    wandb.finish()


if __name__ == "__main__":
    input_lang, output_lang, train_loader, valid_loader, test_loader = dataset.get_dataloader()

    encoder1 = model.Encoder().to(device)
    decoder1 = model.AttentionDecoder().to(device)

    encoder1.load_state_dict(torch.load(
        './attention_best_model_encoder.pth'))
    decoder1.load_state_dict(torch.load(
        './attention_best_model_decoder.pth'))

    source_words, pred_words, attentions = get_attention_map(
        encoder1, decoder1, test_loader)

    for i in range(10):
        plot_attention(source_words[i], pred_words[i], attentions[i][1: len(
            pred_words[i]), 1: len(source_words[i])], f'heatmap_{i+1}')
