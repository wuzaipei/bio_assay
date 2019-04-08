# coding:utf-8
from keras.optimizers import Adam

from liveModel import LivenessNet
from faceDataHandle import getTrainData


longs = 32
wide = 32

X,Y,label = getTrainData("./data/trainingData",longs,wide)

x_train,y_train,x_test,y_test = X[:300,:,:,:],Y[:300,:],X[300:,:,:,:],Y[300:,:]

n_class = len(label)


INIT_LR = 1e-4
batchSize = 8
EPOCHS = 50

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

lmodel = LivenessNet.build(longs,wide,3,n_class )
lmodel.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

lmodel.fit(x_train,y_train,batch_size=batchSize,epochs=EPOCHS)

score = lmodel.evaluate(x_test,y_test)

lmodel.save("./saveModel/realFakeDiscern.h5")
print(score)