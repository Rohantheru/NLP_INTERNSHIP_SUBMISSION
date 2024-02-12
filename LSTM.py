#import the necessary libraries

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

#Loading the training data
df = pd.read_csv(r"train.csv")

#The missing values are dropped incase there are any.
data = df.dropna(subset=['label'],inplace=True)

#Tweets are converted to string format
df["Tweets"] = df["Tweets"].astype(str)
Tweets  = df["Tweets"].values
Labels = df["label"].values

#tokenizer is loaded
token = Tokenizer()

token.fit_on_texts(Tweets)
sequence = token.texts_to_sequences(Tweets)

max_len = max([len(i) for i in sequence])
sequence = pad_sequences(sequence, maxlen=max_len)

#Converting the Labels to numeric
labeler = {"Positive":0, "Negative":1, "Neutral": 2}
label = [labeler[label] for label in Labels]

label = to_categorical(label)

#Train test split is done
X_train, X_test, y_train , y_test = train_test_split(sequence, label , test_size=0.3)

embedding_dim = 100

vocab_size = len(token.word_index) + 1 
num_classes = 3

#The layers are defined
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(units=64, dropout=0.2))
model.add(Dense(units=num_classes, activation='softmax'))

#Activation function is applied
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=12, batch_size=64, validation_data=(X_test, y_test))



loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy with train data: {:.2f}%".format(accuracy * 100))

#For testing with the given test.csv

df = pd.read_csv(r"test.csv")
data = df.dropna(subset=['label'],inplace=True)
df["Tweets"] = df["Tweets"].astype(str)
Tweets  = df["Tweets"].values
Labels = df["label"].values
token = Tokenizer()

token.fit_on_texts(Tweets)
sequence = token.texts_to_sequences(Tweets)

max_len = max([len(i) for i in sequence])
sequence = pad_sequences(sequence, maxlen=max_len)

labeler = {"Positive":0, "Negative":1, "Neutral": 2}
label = [labeler[label] for label in Labels]

label = to_categorical(label)



loss, accuracy = model.evaluate(sequence, label)
print("Accuracy with test data: {:.2f}%".format(accuracy * 100))

