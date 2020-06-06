import pandas as pd
import re
import os

from const import *
from utils import *
from typing import List, Dict, NoReturn
from sklearn.model_selection import train_test_split
from collections import Counter

file_path = os.path.dirname(__file__)


class ManualDataSetHandler:
    """"
    Handler for manual loaded data set

    The manual loaded data set has the following structure:
        a .csv file with columns
         - target
         - ids
         - date
         - flag
         - user
         - text

    The main tasks of this handler are:
        - load data set

        - reformat structure to:
            • tweet_id (where the tweet_id is a unique integer identifying the tweet)
            • sentiment (sentiment is either 1 (positive) or 0 (negative))
            • tweet ( tweet is the tweet enclosed in "" )

        - split processed data set into training and testing
            • the test data set is a csv file of type tweet_id,tweet )

    """

    def __init__(self, data_set_raw):

        self.data_set_raw: str = data_set_raw
        assert self.data_set_raw.endswith('csv'), 'Only .csv file is supported'

        self.data_set_test, self.data_set_train = None, None
        self.make_format()

    def make_format(self):
        """"
        Organize structure of data set to a proper format
        """

        df = pd.read_csv(
            self.data_set_raw,
            encoding=DATA_SET_ENCODING,
            names=DATA_SET_COLUMNS
        )

        # decode ndarray(dtype=int64) -> int
        decode_map: Dict[int, int] = {0: 0, 2: 2, 4: 1}
        df.target = df.target.apply(lambda x: decode_map[int(x)])

        target_cnt = Counter(df.target)
        sys.stdout.write(f'Dataset labels distribution: {target_cnt.keys()} {target_cnt.values()}')

        # drop excess values and reformat columns order
        df.drop(columns=["date", "flag", "user"])
        df = df[['ids', 'target', 'text']]

        # we don't actually need neutral statements
        df.drop(df.loc[df['target'] == 2].index, inplace=True)

        df_train, df_test = train_test_split(df, test_size=1 - TRAIN_SIZE, random_state=42)
        sys.stdout.write(f'\nTRAIN size: {len(df_train)} \nTEST size: {len(df_test)}')
        sys.stdout.flush()

        df_test.drop(columns=['target'])

        self.data_set_test = df_test
        self.data_set_test.name = 'data_set_test'
        self.data_set_train = df_train
        self.data_set_train.name = 'data_set_train'

    @staticmethod
    def preprocess_word(word: str) -> str:
        """
        Remove punctuation from word

            :param str word: string word
            :return: word with punctuation

        """
        word: str = word.strip('\'"?!,.():;')

        # Convert more than 2 letter repetitions to 2 letter
        # loooool --> lol or fuuuuunny --> funny
        word = re.sub(r'(.)\1+', r'\1\1', word)
        # remove special symbols - & '
        word = re.sub(r'(-|\')', '', word)

        return word

    @staticmethod
    def is_valid_word(word: str) -> bool:
        """
        Validation for valid word.
        Under valid word we assume that word begins with  an alphabet
        :param str word: string word
        :return: bool True - word is valid else False
        """
        return re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None

    @staticmethod
    def handle_emojis(tweet: str) -> str:
        """
        Handling emojies in tweet

            :param tweet:
            :return:

        """
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet: str = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)

        return tweet

    @staticmethod
    def preprocess_tweet(tweet: str) -> str:
        """
        Tweet preprocessing

            param str tweet: string tweet
            :return: processed tweet

        """
        processed_tweet: List[str] = []

        # Convert to lower case
        tweet: str = tweet.lower()

        # Replaces URLs with the word URL
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)

        # Replace @handle with the word USER_MENTION
        tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)

        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)

        # Remove RT (retweet)
        tweet = re.sub(r'\brt\b', '', tweet)

        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', ' ', tweet)

        # Strip space, " and ' from tweet
        tweet = tweet.strip(' "\'')

        # Replace emojis with either EMO_POS or EMO_NEG
        tweet = ManualDataSetHandler.handle_emojis(tweet)

        # Replace multiple spaces with a single space
        tweet = re.sub(r'\s+', ' ', tweet)
        words: List[str] = tweet.split()

        for word in words:

            word = ManualDataSetHandler.preprocess_word(word)
            if ManualDataSetHandler.is_valid_word(word):
                processed_tweet.append(word)

        return ' '.join(processed_tweet)

    def preprocess_csv(self) -> NoReturn:
        """
        Preprocessing .csv file

            :return: file name that processed

        """
        for item in [self.data_set_train, self.data_set_test]:
            processed_file_name: str = item.name + '_preprocessed.csv'
            save_to_file = open(os.path.join('dataset', processed_file_name), 'w')

            lines: List[str] = item.values.tolist()
            total: int = len(lines)

            for i, line in enumerate(lines):

                line = " ,".join(map(str, line))
                tweet_id: str = line[:line.find(',')]
                line: str = line[1 + line.find(','):]
                positive: int = int(line[:line.find(',')])
                tweet: str = line
                processed_tweet: str = self.preprocess_tweet(tweet)

                save_to_file.write(f'{tweet_id},{positive},{processed_tweet}\n')
                write_status(i + 1, total)

        save_to_file.close()
        print(f'\nSaved processed tweets to: {processed_file_name}')


ManualDataSetHandler('dataset/dataset_manual_raw.csv').preprocess_csv()
#
# if __name__ == '__main__':
#     csv_file_name = sys.argv[1]
#     ManualDataSetHandler(csv_file_name).preprocess_csv()
