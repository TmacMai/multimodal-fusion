import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from util_function import  *
#from keras.layers import Input, LSTM, Dense, TimeDistributed, Masking, Dropout, Bidirectional,RepeatVector,Add,Activation,Concatenate
#from keras.layers import Convolution2D, MaxPooling2D, Flatten, Multiply, ZeroPadding2D,Reshape, BatchNormalization, Multiply
from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import Lambda
#import theano.tensor as T
import tensorflow
#import tensorflow.tensor as T
import pickle
import sys
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
np.random.seed(1337) # for reproducibility
#from nested_lstm import NestedLSTMCell
#from data_prep import batch_iter, createOneHotMosei2way, get_raw_data
#from nested_lstm import NestedLSTM
from sklearn.metrics import f1_score


unimodal_activations={}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def createOneHot(train_label,  test_label):

    	print('train_label:',train_label)
        maxlen = int(max(train_label.max(), test_label.max()))
	
	train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen+1))   #[shape[0], shape[1], maxlen+1] batch size,length,classes
	test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen+1))
	
	for i in xrange(train_label.shape[0]):
		for j in xrange(train_label.shape[1]):
			train[i,j,train_label[i,j]]=1

	for i in xrange(test_label.shape[0]):
		for j in xrange(test_label.shape[1]):
			test[i,j,test_label[i,j]]=1

	return train,  test

def createVal(train_data, train_mask, train_label, valid_portion=None):

	n_samples = train_data.shape[0]
	sidx = np.arange(n_samples)
	n_train = int(np.round(n_samples * (1. - valid_portion)))

	val_data = np.asarray([train_data[s] for s in sidx[n_train:]])
	val_mask = np.asarray([train_mask[s] for s in sidx[n_train:]])
	val_label = np.asarray([train_label[s] for s in sidx[n_train:]])

	train_data = np.asarray([train_data[s] for s in sidx[:n_train]])
	train_mask = np.asarray([train_mask[s] for s in sidx[:n_train]])
	train_label = np.asarray([train_label[s] for s in sidx[:n_train]])

	return train_data, train_mask, train_label, val_data, val_mask, val_label


def calc_test_result(result, test_label, test_mask):

	true_label=[]
	predicted_label=[]
     #   print('test_label:',test_label)
	for i in xrange(result.shape[0]):
		for j in xrange(result.shape[1]):
			if test_mask[i,j]==1:
				true_label.append(np.argmax(test_label[i,j] ))
				predicted_label.append(np.argmax(result[i,j] ))
		
	print "Confusion Matrix :"
	print confusion_matrix(true_label, predicted_label)
	print "Classification Report :"
	print classification_report(true_label, predicted_label,digits=4)
	print "Accuracy ", accuracy_score(true_label, predicted_label)


def segmentation(text, audio, video, size, stride):
        print('text',text.shape,'audio',audio.shape,'video',video.shape)
        s = stride; length = text.shape[2]
        local = int((length-size)/s) + 1
        if (length-size)%s != 0 :
           k = (length-size)%s
           pad = size - k
           text = np.concatenate((text,np.zeros([text.shape[0],text.shape[1],pad])),axis = 2)
           audio = np.concatenate((audio,np.zeros([text.shape[0],text.shape[1],pad])),axis = 2)
           video = np.concatenate((video,np.zeros([text.shape[0],text.shape[1],pad])),axis = 2)
           local +=1
        input1 =  np.zeros([text.shape[0],text.shape[1],local,3*size])
        fusion = np.zeros([text.shape[0],text.shape[1],local,(size+1)**3])

        for i in range(local):
           # text1 = np.concatenate(np.expand_dims(np.array(text[:,:,i]),axis=2),np.ones([62,63,1]))
            text1 = text[:,:,s*i:s*i+size]
            text2 = text1
          #  text1 = text1[:,:,np.newaxis]
       #     print('text1',text1.shape)
            text1 = np.concatenate((text1,np.ones([text.shape[0],text.shape[1],1])),axis = 2)
            text1 = text1[:,:,:,np.newaxis]

            audio1 = audio[:,:,s*i:s*i+size] 
            audio2 = audio1
         #   audio1 = audio1[:,:,np.newaxis]
            audio1 = np.concatenate((audio1,np.ones([text.shape[0],text.shape[1],1])),axis = 2)
            audio1 = audio1[:,:,np.newaxis,:]

            video1 = video[:,:,s*i:s*i+size]  
            video2 =video1
            video1 = np.concatenate((video1,np.ones([text.shape[0],text.shape[1],1])),axis = 2)
            video1 = video1[:,:,np.newaxis,:]

            ta = np.matmul(text1,audio1)
  
            ta = np.reshape(ta,[text.shape[0],text.shape[1],(size+1)**2,1])
            tav = np.matmul(ta,video1)
        #    print('tav',K.int_shape(tav))
            tav = np.reshape(tav,[text.shape[0],text.shape[1],(size+1)**3])
            fusion[:,:,i,:] = tav
            input1[:,:,i,0:size] = text2
            input1[:,:,i,size:size*2] = video2
            input1[:,:,i,size*2:size*3] = audio2
        return fusion, input1, local


def multimodal(unimodal_activations, args):

	#Fusion (appending) of features
        #[62 63 50] [62 63 150]
	train_data = np.concatenate((unimodal_activations['text_train'], unimodal_activations['audio_train'], unimodal_activations['video_train']), axis=2)
	test_data = np.concatenate((unimodal_activations['text_test'], unimodal_activations['audio_test'], unimodal_activations['video_test']), axis=2)
	train_mask=unimodal_activations['train_mask']
	test_mask=unimodal_activations['test_mask']
	train_label=unimodal_activations['train_label']
	test_label=unimodal_activations['test_label']
      #  concat = Lambda(lambda x: K.concatenate([x[0],x[1]],axis=-1))
        padd = np.ones([62,63,1])
        
        text = unimodal_activations['text_train']
        audio = unimodal_activations['audio_train']
        video = unimodal_activations['video_train']
        fusion, input1, local_number1 = segmentation(text, audio, video, args.segmentation_size, args.segmentation_stride)

        text = unimodal_activations['text_test']
        audio = unimodal_activations['audio_test']
        video = unimodal_activations['video_test']
        fusion2, input2, local_number2 = segmentation(text, audio, video, args.segmentation_size, args.segmentation_stride)



	input_data = Input(shape=(fusion.shape[1],fusion.shape[2],fusion.shape[3]))  #???

        lstm3 = TimeDistributed(ABS_LSTM4(units=3, intra_attention=True, inter_attention=True))(input_data)  # or ABS_LSTM5
        lstm3 = TimeDistributed(Activation('tanh'))(lstm3)  #tanh
        lstm3 = TimeDistributed(Dropout(0.6))(lstm3)   #0.6


        fla = TimeDistributed(Flatten())(lstm3)
 #       fla = TimeDistributed(Activation('tanh'))(fla)

        uni = TimeDistributed(Dense(50,activation='relu'))(fla)   ####50
	uni = Dropout(0.5)(uni)
	output = TimeDistributed(Dense(2,activation='softmax'))(uni) 
	model = Model(input_data, output)
#	model.compile(optimizer='Adagrad', loss='categorical_crossentropy', sample_weight_mode='temporal')
        model.compile(optimizer='RMSprop', loss='cosine_proximity', sample_weight_mode='temporal')
        model.summary()
	early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	model.fit(fusion, train_label,
	                epochs=args.epoch,
	                batch_size=args.batch_size,
	                sample_weight=train_mask,
	                shuffle=True, 
	                callbacks=[early_stopping],
	                validation_split=0.2)
	                
	result = model.predict(fusion2)
	calc_test_result(result, test_label, test_mask)



def unimodal(mode, data, classes):
    print(('starting unimodal ', mode))

    # with open('./mosei/text_glove_average.pickle', 'rb') as handle:
    if data == 'mosei' or data == 'mosi':
        with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
            u = pickle.Unpickler(handle)
          #  u.encoding = 'latin1'
            # (train_data, train_label, test_data, test_label, maxlen, train_length, test_length) = u.load()
            if data == 'mosei':
                (train_data, train_label, _, _, test_data, test_label, _, train_length, _, test_length, _, _,
                 _) = u.load()
                if classes == '2':
                    train_label, test_label = createOneHotMosei2way(train_label, test_label)
            elif data == 'mosi':
                (train_data, train_label, test_data, test_label, maxlen, train_length, test_length) = u.load()

            train_label = train_label.astype('int')
            test_label = test_label.astype('int')

            train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
            for i in range(len(train_length)):
                train_mask[i, :train_length[i]] = 1.0

            test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
            for i in range(len(test_length)):
                test_mask[i, :test_length[i]] = 1.0
    elif data == 'iemocap':
        train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_raw_data(
            data, classes)
        if mode == 'text':
            train_data = text_train
            test_data = text_test
        elif mode == 'audio':
            train_data = audio_train
            test_data = audio_test
        elif mode == 'video':
            train_data = video_train
            test_data = video_test

    # train_label, test_label = createOneHotMosei3way(train_label, test_label)

    print('train_mask', train_mask.shape)

    # print(train_mask_bool)
    seqlen_train = np.sum(train_mask, axis=-1)
    print('seqlen_train', seqlen_train.shape)
    seqlen_test = np.sum(test_mask, axis=-1)
    print('seqlen_test', seqlen_test.shape)

    '''
    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in xrange(len(train_length)):
	train_mask[i,:train_length[i]]=1.0     #[1 1 1 1 0 0 0 ]

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in xrange(len(test_length)):
	test_mask[i,:test_length[i]]=1.0

    train_label, test_label = createOneHot(train_label, test_label)
    '''
    train_label, test_label = createOneHot(train_label, test_label)

    input_data = Input(shape=(train_data.shape[1],train_data.shape[2]))   #[none, 63,100]
    print('input_data size',input_data.shape)
    masked = Masking(mask_value =0)(input_data)     #masked
    lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.6))(masked)
    #      lstm = MEMU(300)(input_data)
    #[none none 600]
    #    print('lstm size',lstm.shape[0],lstm.shape[1],lstm.shape[2])
    inter = Dropout(0.9)(lstm)
    inter1 = TimeDistributed(Dense(50,activation='tanh'))(inter)   #100
    inter = Dropout(0.9)(inter1)
    output = TimeDistributed(Dense(2,activation='softmax'))(inter)

    model = Model(input_data, output)
    aux = Model(input_data, inter1)    #?????
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', sample_weight_mode='temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
##
    print(train_data.shape)
    print(train_label.shape)
    print(train_mask.shape)
    model.fit(train_data, train_label,
                epochs=200,
                batch_size=10,
                sample_weight=train_mask,
                shuffle=True, 
                callbacks=[early_stopping],
                validation_split=0.2)
                

    model.save('./models/mosi_'+mode+'.h5') 

    train_activations = aux.predict(train_data)
    test_activations = aux.predict(test_data)

    unimodal_activations[mode+'_train']=train_activations
    unimodal_activations[mode+'_test']=test_activations

    unimodal_activations['train_mask']=train_mask
    unimodal_activations['test_mask']= test_mask
    unimodal_activations['train_label']=train_label
    unimodal_activations['test_label']=test_label

       

if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimodal", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--fusion", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_raw", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--data", type=str, default='mosi')
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--segmentation_size", type=int, default=2)
    parser.add_argument("--segmentation_stride", type=int, default=2)
    args, _ = parser.parse_known_args(argv)



    batch_size = args.batch_size
    epochs = args.epoch
    emotions = args.classes
    assert args.data in ['mosi', 'mosei', 'iemocap']

    if args.unimodal:
        print("Training unimodals first")
        modality = ['text', 'audio', 'video']
        for mode in modality:
            unimodal(mode, args.data, args.classes)

        print("Saving unimodal activations")
        with open('unimodal_{0}_{1}way.pickle'.format(args.data, args.classes), 'wb') as handle:
            #pickle.dump(unimodal_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)
          pickle.dump(unimodal_activations, handle, protocol=2)
    with open('unimodal_{0}_{1}way.pickle'.format(args.data, args.classes), 'rb') as handle:
         #u = pickle._Unpickler(handle)
         u = pickle.load(handle)
      #  u.encoding = 'latin1'

    multimodal(u, args)


