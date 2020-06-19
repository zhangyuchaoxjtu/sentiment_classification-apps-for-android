import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import collections
import pandas as pd
import numpy as np
import os
import codecs
import tensorflow.keras.backend as k
import matplotlib.pyplot as plt



'''
pos_list=[]
# pos=codecs.open('C:/Users/john/Desktop/情感分析/aclImdb_v1/aclImdb/train/pos_all.txt','r')
with open('/home/Desktop/Apps_for_Android.txt','r',encoding='utf8')as f:
    line=f.readlines()
    pos_list.extend(line)
neg_list=[]
with open('/home/Desktop/Apps_for_Android.txt','r',encoding='utf8')as f:
    line=f.readlines()
    neg_list.extend(line)
#创建标签
'''
content = []
with open('/home/yczhang/Desktop/test2.txt','r',encoding='utf8')as f:
    line=f.readlines()
    content.extend(line)
        
        
label=[1 for i in range(int(len(content)/2))]
label.extend([0 for i in range(int(len(content)/2))])

'''
#评论内容整合
content=pos_list.extend(neg_list)
content=pos_list
'''

seq=[]
seqtence=[]
stop_words=set(stopwords.words('english'))
for con in content:
    words=nltk.word_tokenize(con)
    line=[]
    for word in words:
        if word.isalpha() and word not in stop_words:
            line.append(word)
    seq.append(line)
    seqtence.extend(line)
    
#from here do you know why bug apperas
tokenizer = Tokenizer()
tokenizer.fit_on_texts(content)
one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')    
# 获取词索引
word_index=tokenizer.word_index #without above codes ,this sentence is a bug
temp_sequences=tokenizer.texts_to_sequences(seq) #here
# 此处设置每个句子最长不超过 1000
final_sequences=sequence.pad_sequences(temp_sequences,maxlen=1000) #sequence here means the import package,so we use seqtence above to instead

# 转换为numpy类型
label=np.array(label)
# 随机打乱数据
indices=np.random.permutation(len(final_sequences))
X=final_sequences[indices]
y=label[indices]
# 划分测试集和训练集
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)

def f1_score(y_true , y_pred):
    return 2*((metrics.Precision(y_true , y_pred)*metrics.Recall(y_true , y_pred))/(metrics.Precision(y_true , y_pred) + metrics.Recall(y_true , y_pred))) + k.epsilo()


# 网络构建
model=Sequential()
model.add(Embedding(89483,256,input_length=1000))
model.add(LSTM(128,dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
trained_model = model.fit(Xtrain,ytrain,batch_size=50,epochs=10,validation_data=(Xtest,ytest))



loss = trained_model.history['loss']
val_loss = trained_model.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
