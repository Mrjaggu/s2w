import pickle
from keras.layers import Input, LSTM, Embedding, Dense,Dropout,TimeDistributed
from keras.models import Model
from sklearn.model_selection import train_test_split
from model_file import build

class DumbModel:
    def __init__(self,vocab_size=10000,num_of_encoder_tokens,num_of_decoder_tokens):
        self.vocab_size = vocab_size
        self.clf=None
        self.num_of_encoder_tokens = num_of_encoder_tokens
        self.num_of_decoder_tokens = num_of_decoder_tokens


    def generate_batch(X = X_train, y = y_train, batch_size = 128):
        while True:
            for j in range(0, len(X), batch_size):

                #encoder input
                encoder_input_data = np.zeros((batch_size, max_source_length),dtype='float32')
                #decoder input
                decoder_input_data = np.zeros((batch_size, max_target_size),dtype='float32')

                #target
                decoder_target_data = np.zeros((batch_size, max_target_size, num_of_decoder_tokens),dtype='float32')

                for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                    for t, word in enumerate(input_text.split()):
                        encoder_input_data[i, t] = eng_char_to_index_dict[word] # encoder input seq

                    for t, word in enumerate(target_text.split()):
                        if t<len(target_text.split())-1:
                            decoder_input_data[i, t] = target_char_to_index_dict[word] # decoder input seq
                        if t>0:
                            # decoder target sequence (one hot encoded)
                            # does not include the START_ token
                            # Offset by one timestep since it is one time stamp ahead
                            decoder_target_data[i, t - 1, target_char_to_index_dict[word]] = 1

                yield([encoder_input_data, decoder_input_data], decoder_target_data)


    def train(self,X_train,y_train):

        model = build(num_of_encoder_tokens,num_of_decoder_tokens)

        X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.2)
        train_samples = len(X_train)
        val_samples = len(X_test)
        batch_size = 50
        epochs = 50
        
        model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size  ),
                    steps_per_epoch = train_samples//batch_size,
                    epochs=epochs,
                    callbacks=[es],
                    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                    validation_steps = val_samples//batch_size,)

        pass

    def inference(self):
        # Inference model
        # Encoder
        encoder_inputs = Input(shape=(None,))
        enc_emb =  Embedding(num_of_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
        encoder_lstm = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(num_of_decoder_tokens, latent_dim, mask_zero = True)
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                             initial_state=encoder_states)
        decoder_dense = TimeDistributed(Dense(num_of_decoder_tokens, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        #storing encoder input and internal states so as to give to decoder part
        encoder_model = Model(encoder_inputs, encoder_states)
        #specifying hidden and cell state for decoder part as vector process it will get output predicted and again we add to decoder states
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

        # Final decoder model
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs2] + decoder_states2)

        return encoder_model,decoder_model
    def decode_sequence(self,input_seq):
        # Encode the input as state vectors
        encoder_model,decoder_model= inference()
        states_value = encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))

        target_seq[0, 0] = mar_char_to_index_dict['START_']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = mar_index_to_char_dict[sampled_token_index]
            if (sampled_char == '_END'):
                break;
            decoded_sentence += ' '+sampled_char

            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence


    def pre_process(self):
        sentence = sentence.lower()
        sentance = re.sub("'","",sentence).strip()
        # sentence = re.sub(" +", " ", sentence)
        # remove_digits = str.maketrans('','',digits)
        # sentence=sentence.translate(remove_digits)
        sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in exclude)
        encoder_input_data = np.zeros((1, 35),dtype='float32')
        for t, word in enumerate(sentence.split()):
              encoder_input_data[0, t] = eng_char_to_index_dict[word]

        return encoder_input_data

    def predict(self,x):
        sent = pre_processing(x)
        predicted_output = decode_sequence(sent)
        return predicted_output

    def serialize(self,fname):
        with open(fname,'wb') as f:
            pickle.dump(self.clf,f)

    @staticmethod
    def deserialize(fname):
        model = DumbModel()
        with open(fname,'rb') as f:
            model.clf=pickle.load(f)

            return model
