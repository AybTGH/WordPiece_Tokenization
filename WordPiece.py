from transformers import AutoTokenizer
from collections import defaultdict
import re

class wordpiece: 
    """
    A class for WordPiece tokenization using the Hugging Face Transformers library.

    Attributes:
        corpus (str): The input text corpus to be tokenized.
        vocab_size (int): The desired size of the token vocabulary.

    Methods:
    """
    
    def __init__(self, corpus, vocab_size=100) -> None:
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.word_freqs = self.pre_tokenize()
        self.splits_initial = self.split_list()
        self.vocab_initial = self.initial_vocab()
        self.vocab_final, self.splits_final= self.update_vocab(vocab_size)
        
    def normalize(self, corpus):
        """
        This function takes a text corpus as input and performs several text normalization steps.

        Args:
        corpus (str): The input text corpus to be normalized.

        Returns:
        str: The normalized text corpus after applying the following steps:
            1. Convert all text to lowercase.
            2. Remove any numbers from the text.
            3. Remove all punctuation except for words and spaces.
            4. Strip leading and trailing white spaces.
        """
        #loewrcase
        corpus = corpus.lower()
        # remove numbers
        corpus = re.sub(r'\d+','',corpus)
        # remove all punctuation except words and space
        corpus = re.sub(r'[^\w\s]','', corpus) 
        # remove white spaces
        corpus = corpus.strip()

        return corpus
    
    def pre_tokenize(self):
        """
        Pre-tokenizes the input corpus using the BERT tokenizer and computes word frequencies.

        Returns:
            defaultdict: A dictionary containing word frequencies.
        """
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        word_freqs = defaultdict(int)
        corpus = self.normalize(self.corpus)
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(corpus)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
            
        return word_freqs   

    def split_list(self):
        """
        Splits words in the corpus into subword pieces (subtokens) and stores them in a dictionary.

        Returns:
            dict: A dictionary mapping words to their subword splits.
        """
        splits = {
        word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
        for word in self.word_freqs.keys()
        }
        return splits


    def initial_vocab(self):
        """
        Creates the initial vocabulary including special tokens like [PAD], [UNK], [CLS], [SEP], and [MASK].

        Returns:
            list: A list containing the initial vocabulary tokens.
        """
        alphabet = []
        for word in self.word_freqs.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")
        alphabet.sort()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
        
        return vocab
    
    def compute_pair_scores(self, splits):
        """
        Computes scores for each pair of successive subword elements in each word.

        Args:
            splits (dict): A dictionary mapping words to their subword splits.

        Returns:
            dict: A dictionary mapping subword pairs to their computed scores.
        """
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    def merge_pair(self, a, b, L):
        """
        Merges subword pieces 'a' and 'b' in the splits dictionary for all words in the corpus.

        Args:
            a (str): The first subword piece.
            b (str): The second subword piece.
            L (dict): A dictionary mapping words to their subword splits.

        Returns:
            dict: The updated dictionary of word splits after merging.
        """
        for word in self.word_freqs:
            split = L[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            L[word] = split
        return L

    def update_vocab(self, vocab_size):
        """
        Iteratively updates the vocabulary until it reaches the desired size using pair score computation and merging.

        Args:
            vocab_size (int): The desired size of the token vocabulary.

        Returns:
            tuple: A tuple containing the final vocabulary list and splits dictionary.
        """
        splits_final = self.splits_initial
        vocab_final = self.vocab_initial
        while len(vocab_final) < vocab_size:
            scores = self.compute_pair_scores(splits_final)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            splits_final = self.merge_pair(*best_pair, splits_final)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            vocab_final.append(new_token)
        return vocab_final, splits_final
    
    def encode_word(self, word):
        """
        Encodes a word into a sequence of subtokens using the current vocabulary.

        Args:
            word (str): The input word to be encoded.

        Returns:
            list: A list of subtokens representing the encoded word.
        """
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab_final:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens
    
    def tokenize(self, text):
        """
        Tokenizes a given input text into a sequence of subtokens using the WordPiece tokenization scheme.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of subtokens representing the tokenized text.
        """
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]
        return sum(encoded_words, [])

if __name__ == "__main__":
    with open('Data\Harry_Potter1.txt') as file:
        corpus = file.readlines()
        tokenize = wordpiece(''.join(corpus),vocab_size = 200)
        text = "It took quite a while for them all to get off the platform. A wizened old guard was up by the ticket barrier, letting them go through the gate in twos and threes so they didn't attract attention by all bursting out of a solid wall at once and alarming the Muggles. "
        print(tokenize.tokenize(text))
    