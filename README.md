# README FILE FOR ASSIGNMENT 0
## Error-metrics vs. Model size and Time taken vs. Model size

For the model size, the networks are implemented at a minimum of 4 different sizes and minimum size of each network must be at least 3 layers. I have named the networks: model1(3 layers), model2(4 layers), model3(5 layers), model4(6 layers), and model5(7 layers). The number of image is set to be 1500 to trains all 5 models.

network.py and data.py files are writen separately and are called in main.py

*All the networks were trained and the time and error values were taken. Therefore, I used the records and graph them manually.
### Code
```python
x_train,y_train = subset(nclasses=100, nimgs=1500, x_train=x_train , y_train=y_train)

#model1
model1=model1cnn()
model1.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy','top_k_categorical_accuracy'])
checkpoint = ModelCheckpoint(filepath='./weights.best.hdf5', monitor='val_acc',
                            verbose=1, save_best_only=True, mode='max')

tensorboard= TensorBoard(log_dir='./logs3', histogram_freq=0,
                         write_graph=True, write_images=True)

callbacks_list=[EarlyStopping(min_delta=0.001, patience=3), tensorboard, checkpoint]

start= time.time()

model1.fit(x_train/255.0, to_categorical(y_train),
          batch_size=128,
          shuffle=True,
          epochs=10,
          validation_data=(x_test/255.0, to_categorical(y_test)),
          callbacks=callbacks_list)

scores= model1.evaluate(x_test/255.0, to_categorical(y_test))
end=time.time()

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
print(model1.summary())
print(end-start)
print('Top5: %.3f' % scores[2])
time_model1=end-start


#plt.figure(figsize=(15,5))
#plt.plot(num_layers, (time_model1))
error1=1-scores[1]
print('time: %.3f' % time_model1)
print('error: %.3f' % error1)





#model2
model2=model2cnn()
model2.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy','top_k_categorical_accuracy'])
checkpoint = ModelCheckpoint(filepath='./weights.best.hdf5', monitor='val_acc',
                            verbose=1, save_best_only=True, mode='max')

tensorboard= TensorBoard(log_dir='./logs3', histogram_freq=0,
                         write_graph=True, write_images=True)

callbacks_list=[EarlyStopping(min_delta=0.001, patience=3), tensorboard, checkpoint]

start= time.time()

model2.fit(x_train/255.0, to_categorical(y_train),
          batch_size=128,
          shuffle=True,
          epochs=10,
          validation_data=(x_test/255.0, to_categorical(y_test)),
          callbacks=callbacks_list)

scores= model2.evaluate(x_test/255.0, to_categorical(y_test))
end=time.time()

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
print(model2.summary())
print(end-start)
print('Top5: %.3f' % scores[2])
time_model2=end-start
error2= 1- scores[1]
#plt.plot(num_layers, (time_model2), 'r-')

error2=1-scores[1]
print('time: %.3f' % time_model2)
print('error: %.3f' % error2)




#model3
model3=model3cnn()
model3.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy','top_k_categorical_accuracy'])
checkpoint = ModelCheckpoint(filepath='./weights.best.hdf5', monitor='val_acc',
                            verbose=1, save_best_only=True, mode='max')

tensorboard= TensorBoard(log_dir='./logs3', histogram_freq=0,
                         write_graph=True, write_images=True)

callbacks_list=[EarlyStopping(min_delta=0.001, patience=3), tensorboard, checkpoint]

start= time.time()

model3.fit(x_train/255.0, to_categorical(y_train),
          batch_size=128,
          shuffle=True,
          epochs=10,
          validation_data=(x_test/255.0, to_categorical(y_test)),
          callbacks=callbacks_list)

scores= model3.evaluate(x_test/255.0, to_categorical(y_test))
end=time.time()

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
print(model3.summary())
print(end-start)
print('Top5: %.3f' % scores[2])

time_model3=end-start
#plt.plot(num_layers, (time_model3), 'b-')
error3=1-scores[1]
print('time: %.3f' % time_model3)
print('error: %.3f' % error3)






#model4
model4=model4cnn()
model4.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy','top_k_categorical_accuracy'])
checkpoint = ModelCheckpoint(filepath='./weights.best.hdf5', monitor='val_acc',
                            verbose=1, save_best_only=True, mode='max')

tensorboard= TensorBoard(log_dir='./logs3', histogram_freq=0,
                         write_graph=True, write_images=True)

callbacks_list=[EarlyStopping(min_delta=0.001, patience=3), tensorboard, checkpoint]

start= time.time()

model4.fit(x_train/255.0, to_categorical(y_train),
          batch_size=128,
          shuffle=True,
          epochs=10,
          validation_data=(x_test/255.0, to_categorical(y_test)),
          callbacks=callbacks_list)

scores= model4.evaluate(x_test/255.0, to_categorical(y_test))
end=time.time()

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
print(model4.summary())
print(end-start)
time_model4=end-start
print('Top5: %.3f' % scores[2])

#plt.plot(num_layers, (time_model4), 'g-')
error4=1-scores[1]
print('time: %.3f' % time_model4)
print('error: %.3f' % error4)




#model5
model5=model5cnn()
model5.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy','top_k_categorical_accuracy'])
checkpoint = ModelCheckpoint(filepath='./weights.best.hdf5', monitor='val_acc',
                            verbose=1, save_best_only=True, mode='max')

tensorboard= TensorBoard(log_dir='./logs3', histogram_freq=0,
                         write_graph=True, write_images=True)

callbacks_list=[EarlyStopping(min_delta=0.001, patience=3), tensorboard, checkpoint]

start= time.time()

model5.fit(x_train/255.0, to_categorical(y_train),
          batch_size=128,
          shuffle=True,
          epochs=10,
          validation_data=(x_test/255.0, to_categorical(y_test)),
          callbacks=callbacks_list)

scores= model5.evaluate(x_test/255.0, to_categorical(y_test))
end=time.time()

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
print(model5.summary())
print(end-start)
print('Top5: %.3f' % scores[2])

time_model5=end-start
error5=1-scores[1]

print('time: %.3f' % time_model5)
print('error: %.3f' % error5)



from matplotlib import pyplot as plt
num_layers=[3,4,5,6,7]
#time_all=[time_model1,time_model2,time_model3,time_model4,time_model5]
time_all=[144,199,203,390,645]

plt.figure(figsize=(15,5))
plt.plot(num_layers, time_all)
plt.title('Time taken vs. Model size')
plt.ylabel('Time taken (s)', fontsize=12)
plt.xlabel('Model size', fontsize=12)
plt.show()



error_arr=[0.887,0.909,0.930,0.930,0.946]
#error_arr=[error1,error2,error3,error4,error5]
plt.figure(figsize=(15,5))
plt.plot(num_layers, error_arr)
plt.title('Error-metrics vs. Model size')
plt.ylabel('Error-metrics', fontsize=12)
plt.xlabel('Model size', fontsize=12)
plt.show()
```

##Error-metrics vs. Dataset size and Time taken vs. Dataset size
The number of images was initially set as 1500. For every time we train the network, we have to increase the number of images by 1500 (e.g 1500, 3000, 4500, 6000, 7500)

### Code
```python

#Dataset of different sizes
x_train,y_train = subset(nclasses=100, nimgs=1500, x_train=x_train , y_train=y_train)
model1=model1cnn()
model1.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy', 'top_k_categorical_accuracy'])
checkpoint = ModelCheckpoint(filepath='./weights.best.hdf5', monitor='val_acc',
                            verbose=1, save_best_only=True, mode='max')

tensorboard= TensorBoard(log_dir='./logs3', histogram_freq=0,
                         write_graph=True, write_images=True)

callbacks_list=[EarlyStopping(min_delta=0.001, patience=3), tensorboard, checkpoint]

start= time.time()

model1.fit(x_train/255.0, to_categorical(y_train),
          batch_size=128,
          shuffle=True,
          epochs=10,
          validation_data=(x_test/255.0, to_categorical(y_test)),
          callbacks=callbacks_list)

scores= model1.evaluate(x_test/255.0, to_categorical(y_test))
end=time.time()

time=end-start
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
print('Top5: %.3f' % scores[2])
print('time_modelb = %.3f'% time)
print(model1.summary())
print(end-start)

print('time_modelb = %.3f'% time)


error1b=scores[0]
print('error: %.3f' % error1b)

dataset_size=[1500,1515,1530,1545,1560]
num_layers=[3,4,5,6,7]
time_all_for_b=[157,186,209,273,328]
from matplotlib import pyplot as plt
plt.figure(figsize=(15,5))
plt.plot(dataset_size, time_all_for_b)
plt.title('Time taken vs. Dataset size')
plt.ylabel('Time taken (s)', fontsize=12)
plt.xlabel('Dataset size', fontsize=12)
plt.show()


error_arr_for_b=[4.409, 4.219, 4.057, 4.116, 3.985]

plt.figure(figsize=(15,5))
plt.plot(dataset_size,error_arr_for_b )
plt.title('Error-metrics vs. Dataset size')
plt.ylabel('Error-metrics', fontsize=12)
plt.xlabel('Dataset size', fontsize=12)
plt.show()
```

## Error-metrics vs. no. of iterations
Here is the code for the largest dataset size and corresponding largest network size. After training this network, we can see the tensorboard
### Code
```python

# largest dataset size and network size
x_train,y_train = subset(nclasses=100, nimgs=7500, x_train=x_train , y_train=y_train)
model5=model5cnn()
model5.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy','top_k_categorical_accuracy'])
checkpoint = ModelCheckpoint(filepath='./weights.best.hdf5', monitor='val_acc',
                            verbose=1, save_best_only=True, mode='max')

tensorboard= TensorBoard(log_dir='./logs3', histogram_freq=0,
                         write_graph=True, write_images=True)

callbacks_list=[EarlyStopping(min_delta=0.001, patience=3), tensorboard, checkpoint]

start= time.time()

model5.fit(x_train/255.0, to_categorical(y_train),
          batch_size=128,
          shuffle=True,
          epochs=10,
          validation_data=(x_test/255.0, to_categorical(y_test)),
          callbacks=callbacks_list)

scores= model5.evaluate(x_test/255.0, to_categorical(y_test))
end=time.time()

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
print(model5.summary())
print(end-start)
print('Top5: %.3f' % scores[2])

time=end-start
print('time = %.3f' % time)
error=1-scores[1]
print('error= %.3f' % error)
```


I record the values from tensorboard manually and graph them all together in one graph, and here is the code
### Code
```python
num_epoch=[0,1,2,3,4,5,6,7,8]
acc=[0.0268,0.06147,0.1043,0.1487,0.1929,0.2532,0.3187,0.3957,0.5133]
loss=[4.448,4.165,3.901,3.651,3.388,3.08,2.726,2.323,0]
top_k_categorical_accuracy=[0,0.21,0.3095,0.3831,0.4608,0.4418,0.6248,0.7064,0.7993]
val_acc=[0.0404,0.0777,0.109,0.128,0.1403,0.1452,0.1546,0.1471,0.1505]
val_loss=[4.275,4.098,3.893,3.844,3.824,3.817,3.877,4.209,4.854]
val_top_k_categorical_accuracy=[0,0.2452,0.3111,0.342,0.368,0.359,0.381,0.3666,0.368]

plt.figure(figsize=(15,5))
plt.plot(num_epoch, acc)
plt.plot(num_epoch, loss)
plt.plot(num_epoch, top_k_categorical_accuracy)
plt.plot(num_epoch, val_acc)
plt.plot(num_epoch, val_loss)
plt.plot(num_epoch, val_top_k_categorical_accuracy)

plt.title('Error-metrics vs.number of iterations')
plt.ylabel('Error-metrics', fontsize=12)
plt.xlabel('Epochs', fontsize=12)
plt.legend(['acc','loss','top_k_categorical_accuracy','val_acc','val_loss','val_top_k_categorical_accuracy'], loc='upper left')
plt.show()
```
