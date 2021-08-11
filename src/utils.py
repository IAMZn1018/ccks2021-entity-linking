import json 
import pickle

def vocab_build(vocab_path, min_count=5):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    word2id = {}
    with open('./src/data/entity_type/sentence.csv', 'r', encoding='utf-8') as f:
        data = f.readlines()
    for sent_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

if __name__ == "__main__":
    vocab_build('./src/data/dictionary.pkl', 5)