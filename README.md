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
