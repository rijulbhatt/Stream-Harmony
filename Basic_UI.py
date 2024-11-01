import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout,Bidirectional,Dense,Embedding
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision,Recall,CategoricalAccuracy
import streamlit as st

df = pd.read_csv('data//train.csv')

x = df['comment_text'] # all the text data(the input values)
y = df[df.columns[2:]].values # labels for each text data as an array of integers(the target values)
MAX_FEATURES = 200000 # No. of unique words to be stored in the model.

vectorizer  = TextVectorization(max_tokens = MAX_FEATURES,
                                output_sequence_length = 1800,
                                output_mode = 'int')  
# max_tokens - Total number of tokens/words to store 
# output_sequance_length - Max length of one sentance(no. of words)
# output_mode = int - the words are mapped to integers.

vectorizer.adapt(x.values)

vectorized_text = vectorizer(x.values)

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text,y)) # create a tf.data.datset object where input is vectorized text and output is multilabels (supervised learning)
dataset = dataset.cache()  # catches the dataset in memory which saves time as we dont have to load data  from the disk after each epoch 
dataset = dataset.shuffle(160000)  # shuflle th e dataset so that NN does not earn the order of input which may result in overfitting
dataset = dataset.batch(16) # create batches of the datset helps in parallel processing 16 i sthe batch size
dataset = dataset.prefetch(8) # tells how many batches have to be prepared before the end of execution of current batch

train = dataset.take(int(len(dataset)*0.7))  # get the first 70% of the data as train data take takes in the number of batches to extract
val = dataset.skip(int(len(dataset)*0.7)).take(int(len(dataset)*0.2)) # skip the fisrt 70% as it is training data, of the 30% first 20% is taken as validation data
test = dataset.skip(int(len(dataset)*0.9)).take(int(len(dataset)*0.1)) # the data after 90% is left for testing

train_generator = train.as_numpy_iterator()


model = load_model('biltsm.h5')

st.title('Toxicity Comment Classification')
st.write("Enter a comment below to get a toxicity report:")

# Create a text input field
comment = st.text_area("Comment")

if 'prediction' not in st.session_state:
    st.session_state.prediction = []

if st.button("Classify"):
    if comment:
        try:
            # Vectorize the text
            vector_text = vectorizer([comment])
            if vector_text is None:
                st.error("Vectorization failed. The text might not be processed correctly.")
                st.stop()
            
            # Predict toxicity
            predictions = model.predict(vector_text)
            if predictions is None:
                st.error("Prediction failed. The model might not be processing the input correctly.")
                st.stop()

            # Update session state with new results
            st.session_state.prediction = predictions[0]
            
            # Display the results
            st.write("Toxicity Report:")
            categories = df.columns[2:]  # Get the categories from the DataFrame
            if len(st.session_state.prediction) == len(categories):
                for i, category in enumerate(categories):
                    st.write(f"{category}: {st.session_state.prediction[i]>0.5}")
            else:
                st.error("Prediction results do not match the expected number of categories.")
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
    else:
        st.write("Please enter a comment.")
