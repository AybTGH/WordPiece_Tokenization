from transformers import AutoTokenizer
from collections import defaultdict


class wordpiece: 
    
    def __init__(self, corpus, vocab_size=100) -> None:
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.word_freqs = self.pre_tokenize()
        self.splits_initial = self.split_list()
        self.vocab_initial = self.initial_vocab()
        self.vocab_final, self.splits_final= self.update_vocab(vocab_size)
    
    def pre_tokenize(self):
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        word_freqs = defaultdict(int)
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(self.corpus.strip())
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
            
        return word_freqs   

    def split_list(self):
        splits = {
        word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
        for word in self.word_freqs.keys()
        }
        return splits


    def initial_vocab(self):
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
    

    #function that computes the scores for each pair of successive elements in each word
    def compute_pair_scores(self, splits):
        
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

#function that apply the apply the previous lerge in the splits dictionnary 
    def merge_pair(self, a, b, L):
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
        splits_final = self.splits_initial
        vocab_final = self.vocab_initial
        while len(self.vocab_initial) < vocab_size:
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
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]
        return sum(encoded_words, [])

if __name__ == "__main__":
    corpus ='''
        Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
        Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future.
        Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
        Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
        Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future 
        integration mercurial self script web. Return raspberrypi community test she stable.
        Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
    '''
    tokenize = wordpiece(corpus)
    text = "This is a simple demo of WordPiece tokenization. exception! "
    print(tokenize.tokenize(text))
    