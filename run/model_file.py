from keras.layers import Input, LSTM, Embedding, Dense,Dropout,TimeDistributed
from keras.models import Model

def build(num_of_encoder_tokens,num_of_decoder_tokens):
    #specifying our embedding output vector size
    latent_dim = 256

    # Encoder
    encoder_inputs = Input(shape=(None,))# here defining encoder input
    enc_emb =  Embedding(num_of_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs) # here defining embedding layer with number of input , units=256 i.e output embedded
    encoder_lstm = LSTM(latent_dim, return_sequences=True,dropout=0.4,recurrent_dropout=0.4) # here defining lstm layer
    encoder_output1= encoder_lstm(enc_emb)
    # #encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output2= encoder_lstm2(encoder_output1)
    # #encoder lstm 3
    encoder_lstm3=LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_outputs, state_h3, state_c3= encoder_lstm3(encoder_output2)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h3, state_c3]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,)) #here defining our decoder input shape
    dec_emb_layer = Embedding(num_of_decoder_tokens, latent_dim, mask_zero = True) # here defining embedding for decoder part
    dec_emb = dec_emb_layer(decoder_inputs) # passing decoder input to embedding
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) # here defining lstm layer with units=256 , return seq = True, return state=True
    decoder_outputs, _, _ = decoder_lstm(dec_emb,initial_state=encoder_states) # here passing embeddding output to lstm layer which give decoder output,hidden state, cell state
    decoder_dense = TimeDistributed(Dense(num_of_decoder_tokens, activation='softmax')) #here defining TimeDistributed layer with number of decoder
    decoder_outputs = decoder_dense(decoder_outputs) # passing decoder output to timedistributed layer

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model
