import pandas as pd
import string


path = '' #enter here data path if csv file
data = pd.read_csv(path)
#specifying to remove punctuation
exclude = set(string.punctuation)

class Main():
    def process(df):
        # lowercase all the character
        data['source']=data.source.apply(lambda x: x.lower())
        data['target']=data.target.apply(lambda x: x.lower())

        # # Remove all special character
        data['source']=data.source.apply(lambda x: "".join(ch for ch in x if x not in exclude))
        data['target']=data.target.apply(lambda x:"".join(ch for ch in x if x not in exclude))
        data['source']=data.source.apply(lambda x: re.sub("—","",x))
        data['target']=data.target.apply(lambda x: re.sub("—","",x))

        # Remove extra spaces
        data['source']=data.source.apply(lambda x: x.strip())

        return df

    data = process(data)
    data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    data.replace('', np.nan, inplace=True)

    # Add start and end tokens to target sequences
    data['target'] = data.target.apply(lambda x : 'START_ '+ x + ' _END')

    def vocab(data):
    # Vocabulary of English storing all the words in a set and same for marathi vocab
        all_source_words=set()
        for eng in data.source:
            for word in eng.split():
                if word not in all_source_words:
                    all_source_words.add(word)

        # Vocabulary of marathi
        all_target_words=set()
        for mar in data.target:
            for word in mar.split():
                if word not in all_target_words:
                    all_target_words.add(word)

        max_source_length = (max([len(l) for l in data.source]))
        max_target_size = (max([len(l) for l in data.target]))
        input_words = sorted(list(all_source_words))
        target_words = sorted(list(all_target_words))
        #storing the vocab size for encoder and decoder
        num_of_encoder_tokens = len(all_source_words)
        num_of_decoder_tokens = len(all_target_words)
        num_of_decoder_tokens += 1

        return num_of_encoder_tokens,num_of_decoder_tokens,input_words,target_words

    num_of_encoder_tokens,num_of_decoder_tokens,input_words,target_words = vocab(data)

    def dictionary():

        # dictionary to index each english character - key is index and value is english character
        eng_index_to_char_dict = {}

        # dictionary to get english character given its index - key is english character and value is index
        eng_char_to_index_dict = {}

        for key, value in enumerate(input_words):
            eng_index_to_char_dict[key] = value
            eng_char_to_index_dict[value] = key

        #similary for target i.e marathi words
        target_index_to_char_dict = {}
        target_char_to_index_dict = {}
        for key,value in enumerate(target_words):
            target_index_to_char_dict[key] = value
            target_char_to_index_dict[value] = key

    eng_index_to_char_dict,eng_char_to_index_dict,target_index_to_char_dict,target_index_to_char_dict = dictionary(input_words,target_words)

if __name__ = '__main__':
    fun = Main()
    
