import pandas as pd
import numpy
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = ""

from pr3d.de import ConditionalGaussianMM

condition_labels = ['queue_length_h0'] #,'longer_delay_prob_h1','queue_length_h0','queue_length_h1']
y_label = 'end2end_delay'

df_train = pd.read_parquet('dataset.parquet')
df_train = df_train[
    [
        y_label,
        *condition_labels
    ]
]
#df_train = df_train[
#    df_train.queue_length_h0 >= 0
#]
#df_train = df_train[
#    df_train.queue_length_h1 >= 0
#]
# dataset pre process
df_train = df_train[[y_label, *condition_labels]]
df_train["y_input"] = df_train[y_label]
df_train = df_train.drop(columns=[y_label])

# print(df_train)
# print("Hello world!")

model = ConditionalGaussianMM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)

batch_size = 1000

X = df_train[condition_labels]
Y = df_train.y_input

steps_per_epoch = len(df_train) // batch_size

model.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005,
    ),
    loss=model.loss,
)

model.training_model.fit(
    x=[X, Y],
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=1000,
    verbose=1,
)

# x=np.array([[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4]])
# y=np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170])
x1=np.array([[5],[5]])
y=np.arange(200)
p1=np.zeros(y.size)
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x1, yr)
    p1[i]=prob[0]
    p1[i+1]=prob[1]
print(p1[100])
x2=np.array([[10],[10]])
p2=np.zeros(y.size)
for i in range(0,y.size-1,2):
    yr2=np.array([y[i],y[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x2, yr2)
    p2[i]=prob[0]
    p2[i+1]=prob[1]
print(p2[100])
# prob, logprob, pred_cdf = model.prob_batch(x, y)
# plt.figure()
# plt.plot(y,p, 'go')
# print(prob)
# print(logprob)

plt.plot(y, p1)
plt.plot(y, p2)
plt.xlabel('delay')
plt.ylabel('probability')
plt.legend(['queue_length_h0=5', 'queue_length_h0=10'], loc='upper left')
plt.show()
