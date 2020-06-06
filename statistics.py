""""
Takes in a preprocessed CSV file and gives statistics
Writes the frequency distribution of words and bigrams to pickle files.
"""

from nltk import FreqDist
import pickle
import sys
import os

from utils import write_status
from collections import Counter
from typing import Tuple, Any, List, Dict


def analyze_tweet(tweet: str) -> Tuple[Any, List[Tuple[Any, Any]]]:
    """
    :param tweet:
    :return:

    """
    words: List[str] = tweet.split()
    bigrams = get_bigrams(words)
    return words, bigrams


def get_bigrams(tweet_words: List[str]) -> List[Tuple[str, str]]:
    """
    :param tweet_words:
    :return:
    """
    bigrams: List[Tuple[str, str]] = []
    num_words = len(tweet_words)
    for i in range(num_words - 1):
        bigrams.append((tweet_words[i], tweet_words[i + 1]))
    return bigrams


def get_bigram_freqdist(bigrams) -> Counter:
    """

    :param bigrams:
    :return:
    """
    freq_dict: Dict[Tuple[str, str], int] = {}

    for bigram in bigrams:

        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1

    counter = Counter(freq_dict)

    return counter


def main(file_name: str):

    num_tweets, num_pos_tweets, num_neg_tweets = 0, 0, 0
    all_words: List[str] = []
    all_bigrams: List[Tuple[str, str]] = []

    with open(file_name) as csv:

        lines = csv.readlines()
        num_tweets = len(lines)

        for i, line in enumerate(lines):

            t_id, if_pos, tweet = line.strip().split(',')
            if_pos = int(if_pos)

            if if_pos:
                num_pos_tweets += 1
            else:
                num_neg_tweets += 1

            words, bigrams = analyze_tweet(tweet)
            all_words.extend(words)
            all_bigrams.extend(bigrams)
            write_status(i + 1, num_tweets)

    unique_words = list(set(all_words))
    unique_words_file_name = 'unique.txt'

    with open(os.path.join('dataset', unique_words_file_name), 'w') as uwf:
        uwf.write('\n'.join(unique_words))
    sys.stdout.write('\nCalculating frequency distribution')
    sys.stdout.flush()

    # Unigrams
    freq_dist = FreqDist(all_words)
    pkl_file_name = 'freqdist.pkl'

    with open(os.path.join('dataset', pkl_file_name), 'wb') as pkl_file:
        pickle.dump(freq_dist, pkl_file)
    sys.stdout.write(f'Saved uni-frequency distribution to {pkl_file_name}')
    sys.stdout.flush()

    # Bigrams
    bigram_freq_dist = get_bigram_freqdist(all_bigrams)
    bi_pkl_file_name = 'freqdist-bi.pkl'

    with open(os.path.join('dataset', bi_pkl_file_name), 'wb') as pkl_file:
        pickle.dump(bigram_freq_dist, pkl_file)

    sys.stdout.write(f'Saved bi-frequency distribution to {bi_pkl_file_name}')
    sys.stdout.write('\n[Analysis Statistics]')
    sys.stdout.write(f'Tweets => Total: {num_tweets}, Positive: {num_pos_tweets}, Negative: {num_neg_tweets}')
    sys.stdout.flush()


main('dataset/data_set_train_preprocessed.csv')
