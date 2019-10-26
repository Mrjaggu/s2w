import pickle
import numpy as np
# import sys
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
import warnings
warnings.filterwarnings("ignore")

# load model_
model2 = load_model("./model_protein_sequence.h5")
labelencoder = joblib.load('./labelencoder.pkl')
char2index_dict=joblib.load('./char2index_dict.pkl')

def predict2(text):
    text=text.lower()
    final=[]
    seq1=[]
    for s in (text):
      x=char2index_dict[s]
      seq1.append(str(x))
    final.append(seq1)
    final_sequence = sequence.pad_sequences(final, maxlen=100,padding='post')
    nb_classes = 24
    targets = np.array(final_sequence)
    one_hot_train = np.eye(nb_classes)[targets]
    # print("Step 1 cleared")
    return (one_hot_train)

x = input("Please enter your phrase: ")
result = predict2(x)
res= model2.predict(result)
pred = labelencoder.inverse_transform([np.argmax(res)])
# print("Step 1 cleared")
print("Result",pred[0])
