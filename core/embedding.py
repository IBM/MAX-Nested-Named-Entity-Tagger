import argparse

######## Flair and ELMO and Bert Embedding ########
from flair.embeddings import FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings
from flair.data import Sentence, Token
from typing import List
from pytorch_pretrained_bert import BertTokenizer

def bert_embeddings(sentences, tokenized_contents, output_file=None):
    # Using bert_tokenizer for checking for sequence wordpeice tokens length > 512
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    if output_file:
        f = open(output_file, 'w')
    # init embedding
    # init multilingual BERT
    bert_embedding = TransformerWordEmbeddings('bert-large-uncased')
    long_sent = False
    for i, (sent, sent_tokens) in enumerate(zip(sentences,tokenized_contents)):
        print("Encoding the {}th input sentence for BERT embedding!".format(i))
        # getting the length of bert tokenized sentence after wordpeice tokenization
        if len(bert_tokenizer.tokenize(sent[0])) >= 510:
            long_sent = True
            truncated_tokens=sent_tokens[:len(sent_tokens)//2]
            sent_tokens=sent_tokens[len(sent_tokens)//2:]

        # Using our own tokens (our own tokenization)
        tokens: List[Token] = [Token(token) for token in sent_tokens]

        # create an empty sentence
        sentence = Sentence()

        # add tokens from our own tokenization
        sentence.tokens = tokens

        bert_embedding.embed(sentence)

        for j , (token,st) in enumerate(zip(sentence,sent_tokens)):
            if token.text != st
                raise ValueError("Invalid token text")
            if output_file:
                f.write(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
            else:
                print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')

        if long_sent:
            # tokenization for the rest of the sentence
            truncated_tokens: List[Token] = [Token(token) for token in truncated_tokens]
            # Create empty sentence
            truncated_sentence = Sentence()
            # add tokens from our own tokenization
            truncated_sentence.tokens = truncated_tokens
            bert_embedding.embed(truncated_sentence)
            for token in truncated_sentence:
                if output_file:
                    f.write(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
                    #print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
                else:
                    print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
            long_sent = False

        f.write('\n')

def flair_embeddings(sentences, tokenized_contents, output_file=None):
    if output_file:
        f = open(output_file, 'w')
    # init embedding
    flair_embedding_forward = FlairEmbeddings('news-forward')
    for i, (sent, sent_tokens) in enumerate(zip(sentences,tokenized_contents)):
        print("Encoding the {}th input sentence for Flair embedding!".format(i))
        # Getting the tokens from our own tokenized sentence!
        tokens: List[Token] = [Token(token) for token in sent_tokens]

        assert len(tokens)==len(sent_tokens)

        # Create new empty sentence
        sentence = Sentence()

        # add our own tokens
        sentence.tokens = tokens

        flair_embedding_forward.embed(sentence)

        for token in sentence:

            if output_file:
                f.write(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
                #print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
            else:
                print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
        f.write('\n')

def elmo_embeddings(sentences, tokenized_contents, output_file=None):
    if output_file:
        f = open(output_file, 'w')
    # init embedding
    # For English biomedical data you can use 'pubmed'
    embedding = ELMoEmbeddings('original')  # English	4096-hidden, 2 layers, 93.6M parameters
    for i, (sent, sent_tokens) in enumerate(zip(sentences,tokenized_contents)):
        print("Encoding the {}th input sentence for ELMO embedding!".format(i))
        # Getting the tokens from our own tokenized sentence!
        tokens: List[Token] = [Token(token) for token in sent_tokens]

        if len(tokens) != len(sent_tokens):
            raise ValueError("token length does not match sent_tokens length")

        # Create new empty sentence
        sentence = Sentence()

        # add our own tokens
        sentence.tokens = tokens

        embedding.embed(sentence)

        for token in sentence:

            if output_file:
                f.write(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
                #print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
            else:
                print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
        f.write('\n')

def elmo_embeddings_flair_lib_tokenization(sentences, output_file=None):
    if output_file:
        f = open(output_file, 'w')
    # init embedding
    # For English biomedical data you can use 'pubmed'
    embedding = ELMoEmbeddings('original')  # English	4096-hidden, 2 layers, 93.6M parameters
    for i, sent in enumerate(sentences):
        print("Encoding the {}th input sentence!".format(i))
        # create a sentence
        sentence = Sentence(sent[0])  # [sent] --> sent

        # embed words in sentence
        embedding.embed(sentence)
        for token in sentence:
            if output_file:

                f.write(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
                #print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
            else:
                print(token.text + " " + " ".join([str(num) for num in token.embedding.tolist()]) + '\n')
        f.write('\n')

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default=None, type=str, help="Input data path.")
    parser.add_argument("--output_file", default=None, type=str, help="Output data path")
    parser.add_argument("--input_sentence", default=None, type=str, help="Input sentence.")
    parser.add_argument("--contextual_embedding", default="BERT", type=str, nargs='*',
                        help="BERT|ELMO|FLAIR contextual embeddings.")

    args = parser.parse_args()
    if args.input_file:
        contents = []
        tokenized_contents = []
        sentence = ''
        tokenized_sentence = []
        # Recreating the sentences to get the contextualized embedding as opposed to isolated word embeddings!
        # considering the format of the CONLL files (sentences/articles are separated by one empty line!)
        f = open(args.input_file, 'r')
        line = f.readline()
        n_lines = 0
        while line:
            n_lines += 1
            if line == '\n':  # next sentence
                contents.append([sentence])
                tokenized_contents.append(tokenized_sentence)
                sentence = ''
                tokenized_sentence = []

            else:
                token_per_line = line.split('\t')[0] # token in the conll
                sentence += ' ' + token_per_line
                tokenized_sentence.append(token_per_line)
            line = f.readline()

    else:
        contents = [args.input_sentence]

    #print(contents)
    #print('n_lines={}'.format(n_lines))
    assert len(contents) == len(tokenized_contents)
    #flat_list = [item for sublist in tokenized_contents for item in sublist]
    #print(len(flat_list))
    #print('len(tokenized_contents)={}'.format(len(tokenized_contents)))
    if 'BERT' in args.contextual_embedding:
        if args.output_file:

            bert_embeddings(contents, tokenized_contents, args.output_file+'_bert_large_embeddings.txt')
        else:
            bert_embeddings(contents, tokenized_contents)

    if 'ELMO' in args.contextual_embedding:
        if args.output_file:
            elmo_embeddings(contents, tokenized_contents, args.output_file+'_elmo_embeddings.txt')
        else:
            elmo_embeddings(contents, tokenized_contents)

    # For the FLAIR it seems the tokenization is aligned with CONLL tokenization!
    if 'FLAIR' in args.contextual_embedding:
        if args.output_file:
            flair_embeddings(contents, tokenized_contents, args.output_file+'_flair_embeddings.txt')
        else:
            flair_embeddings(contents, tokenized_contents)
