from keras.layers import *
from keras.activations import *
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU
from keras import backend as K
from keras.layers import Input, LSTM, Dense, Dropout
import numpy as np
from keras import regularizers








class ABS_LSTM4(Layer):

    def __init__(self, units, intra_attention=True, inter_attention=True, **kwargs):
        self.units = units 
        self.intra_attention = intra_attention
        self.inter_attention = inter_attention
        super(ABS_LSTM4, self).__init__(**kwargs)

    def build(self, input_shape): 
        

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units *4 ),
                                      initializer='glorot_normal',
                                      trainable=True)

        self.recurrent_kernel = self.add_weight(
                                shape=(self.units, self.units * 4),
                                name='recurrent_kernel',
                                initializer='glorot_normal',
                                trainable = True
                                                )    
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.inter_attention:

                 self.attention_h = self.add_weight(name='attention_h',
                                      shape=(1*input_shape[-1]+1*self.units, self.units ),  ########self.units
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
                 self.attention_h2 = self.add_weight(name='attention_h2',
                                      shape=(self.units, 1 ),
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
                 self.attention_c = self.add_weight(name='attention_c',
                                      shape=(1*input_shape[-1]+1*self.units, 5),
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
                 self.attention_c2 = self.add_weight(name='attention_c2',
                                      shape=(self.units, 1 ),
                                      initializer='glorot_normal',
                                      trainable=True)
                  

                 self.transform = self.add_weight(name='transform',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
                 '''
                 self.biase_c = self.add_weight(name='b_c',
                                      shape=(5,),
                                      initializer='glorot_normal',
                                      trainable=True)
                 self.biase_h = self.add_weight(name='b_h',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
                 
        if self.intra_attention:
             
             self.attention1 = self.add_weight(name='attention',
                                      shape=(1*self.units+input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.biase1 = self.add_weight(name='b',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             
             self.attention2 = self.add_weight(name='attention2',
                                      shape=(self.units, 1),     ####1
                                      initializer='glorot_normal',
                                      trainable=True)
             '''
             self.attention2_2 = self.add_weight(name='attention2_2',
                                      shape=(self.units, 1),
                                      initializer='glorot_normal',
                                      trainable=True)
             '''
             
             self.biase2 = self.add_weight(name='b2',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             
    def step_do(self, step_in, states):

        x_i =( K.dot(step_in, self.kernel_i) )
        x_f =( K.dot(step_in, self.kernel_f) ) 
        x_c =( K.dot(step_in, self.kernel_c) )
        x_o = ( K.dot(step_in, self.kernel_o) )
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        h2 = states[2]
        c_tm2 = states[3]
        h_next = h2
        c_next = c_tm2
        h_tm3 = states[4]
        c_tm3 = states[5]
     
        if self.inter_attention:
          #  step_in2 = K.relu(K.dot(step_in,self.transform)+self.biase3)
            a = K.concatenate([h_tm1,step_in],axis=1); a1 = K.tanh(K.dot(a,self.attention_h));a1 = K.sqrt(K.sum(a1**2,axis=1))#a1 = K.squeeze((K.dot(a1,self.attention_h2)),1)# 
            a = K.concatenate([h2,step_in],axis=1);    a2 = K.tanh(K.dot(a,self.attention_h));a2 = K.sqrt(K.sum(a2**2,axis=1))#a2 = K.squeeze((K.dot(a2,self.attention_h2)),1)# 
            a = K.concatenate([h_tm3,step_in],axis=1); a3 = K.tanh(K.dot(a,self.attention_h));a3 = K.sqrt(K.sum(a3**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_h2)),1)# 
            a = K.concatenate([K.expand_dims(a1,1),K.expand_dims(a2,1)],axis=1); a = K.concatenate([a,K.expand_dims(a3,1)],axis=1)
            a = K.softmax(a); a1 = K.expand_dims(a[:,0],1); a2 = K.expand_dims(a[:,1],1); a3 = K.expand_dims(a[:,2],1); 
            h2 =  K.relu(a1*h_tm1 + a2*h2 + a3*h_tm3)

            a = K.concatenate([c_tm1,step_in],axis=1); a1 = K.tanh(K.dot(a,self.attention_c));a1 = K.sqrt(K.sum(a1**2,axis=1))#a1 = K.squeeze((K.dot(a1,self.attention_c2)),1)# 
            a = K.concatenate([c_tm2,step_in],axis=1); a2 = K.tanh(K.dot(a,self.attention_c));a2 = K.sqrt(K.sum(a2**2,axis=1))#a2 = K.squeeze((K.dot(a2,self.attention_c2)),1)# 
            a = K.concatenate([c_tm3,step_in],axis=1); a3 = K.tanh(K.dot(a,self.attention_c));a3 = K.sqrt(K.sum(a3**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_c2)),1)# 
            a = K.concatenate([K.expand_dims(a1,1),K.expand_dims(a2,1)],axis=1); a = K.concatenate([a,K.expand_dims(a3,1)],axis=1)
            a = K.softmax(a); a1 = K.expand_dims(a[:,0],1); a2 = K.expand_dims(a[:,1],1); a3 = K.expand_dims(a[:,2],1); 
            c2 =  K.relu(a1*c_tm1 + a2*c_tm2 + a3*c_tm3)


        else:
            h2 =  K.relu(h_tm1 + h2 + h_tm3)/3 
            c2 =  K.relu(c_tm1 + c_tm2 + c_tm3)/3


        i = K.hard_sigmoid(x_i + K.dot(h2,self.recurrent_kernel_i))
        f = K.hard_sigmoid(x_f + K.dot(h2,self.recurrent_kernel_f))
        o = K.hard_sigmoid(x_o + K.dot(h2,self.recurrent_kernel_o))
        m = K.hard_sigmoid(x_c + K.dot(h2,self.recurrent_kernel_c))

        c = (f * c2 + i * m )

        h = (o * K.tanh( c ))

        return h, [h,c,h_tm1,c_tm1,h_next,c_next]   



    def call(self, inputs):
        s = K.zeros((K.shape(inputs)[0],self.units))
        init_states = [s,s,s,s,s,s]
        outputs = K.rnn(self.step_do, inputs, init_states)[1]
        
        if self.intra_attention:
      #     self.attention1_1 = self.attention1[:self.units,:]
      #     self.attention1_2 = self.attention1[self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs[:,i,:]

                h_atten = K.concatenate([h,step_in],axis=1)     
                h_atten=K.hard_sigmoid(K.dot(h_atten,self.attention1) + 1*self.biase1)     
                h_atten=(K.dot(h_atten,self.attention2))
                h_atten = K.elu(1*h_atten*h+0*self.biase2)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs = Reshape((inputs.shape[1],self.units))(output_atten)       

        
        init_states2 = [s,s,s,s,s,s]
        input2 = K.reverse(inputs,axes=1)
        outputs2 = K.rnn(self.step_do, input2, init_states2)[1]
        
        if self.intra_attention:
     #      self.attention1_1 = self.attention1[:self.units,:]
    #       self.attention1_2 = self.attention1[self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs2[:,i,:]

                h_atten = K.concatenate([h,step_in],axis=1)   
                h_atten=K.hard_sigmoid(K.dot(h_atten,self.attention1) + 1*self.biase1)     
                h_atten=(K.dot(h_atten,self.attention2))
                h_atten = K.elu(1*h_atten*h+0*self.biase2)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs2 = Reshape((inputs.shape[1],self.units))(output_atten)   
        

        outputs2 = K.reverse(outputs2,axes=1)
        outputs = (K.concatenate([outputs,outputs2]))


        '''
        if self.intra_attention:
           self.attention1_1 = self.attention1[:2*self.units,:]
           self.attention1_2 = self.attention1[2*self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs[:,i,:]

                h_atten=K.relu(K.dot(h,self.attention1_1) + 0*self.biase1)     ##################0
                h_atten=(K.dot(h_atten,self.attention2))

                h_b=K.relu(K.dot(step_in,self.attention1_2)+0*self.biase2)     ##################1
                h_b=(K.dot(h_b,self.attention2_2))

                h_atten = K.tanh(h_atten*h + h_b)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs = Reshape((inputs.shape[1],2*self.units))(output_atten)   
           '''
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.units*2)  


class ABS_LSTM5(Layer):

    def __init__(self, units, intra_attention=True, inter_attention=True, **kwargs):
        self.units = units 
        self.intra_attention = intra_attention
        self.inter_attention = inter_attention
        super(ABS_LSTM5, self).__init__(**kwargs)

    def build(self, input_shape): 
        

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units *4 ),
                                      initializer='glorot_normal',
                                      trainable=True)

        self.recurrent_kernel = self.add_weight(
                                shape=(self.units, self.units * 4),
                                name='recurrent_kernel',
                                initializer='glorot_normal',
                                trainable = True
                                                )    
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.inter_attention:

                 self.attention_h = self.add_weight(name='attention_h',
                                      shape=(1*input_shape[-1]+1*self.units, self.units ),
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
                 self.attention_h2 = self.add_weight(name='attention_h2',
                                      shape=(self.units, 1 ),
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
                 self.attention_c = self.add_weight(name='attention_c',
                                      shape=(1*input_shape[-1]+1*self.units, self.units ),
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
                 self.attention_c2 = self.add_weight(name='attention_c2',
                                      shape=(self.units, 1 ),
                                      initializer='glorot_normal',
                                      trainable=True)
                  

                 self.transform = self.add_weight(name='transform',
                                      shape=(input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
                 self.biase3 = self.add_weight(name='b3',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
                 '''
        if self.intra_attention:
             
             self.attention1 = self.add_weight(name='attention',
                                      shape=(2*self.units+input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
             self.biase1 = self.add_weight(name='b',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.attention2 = self.add_weight(name='attention2',
                                      shape=(self.units, 1),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.attention2_2 = self.add_weight(name='attention2_2',
                                      shape=(self.units, 1),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.biase2 = self.add_weight(name='b2',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
    def step_do(self, step_in, states):

        x_i =( K.dot(step_in, self.kernel_i) )
        x_f =( K.dot(step_in, self.kernel_f) ) 
        x_c =( K.dot(step_in, self.kernel_c) )
        x_o = ( K.dot(step_in, self.kernel_o) )
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        h2 = states[2]
        c_tm2 = states[3]
        h_next = h2
        c_next = c_tm2
        h_tm3 = states[4]
        c_tm3 = states[5]
     
        if self.inter_attention:
          #  step_in2 = K.relu(K.dot(step_in,self.transform)+self.biase3)
            a = K.concatenate([h_tm1,step_in],axis=1); a1 = K.tanh(K.dot(a,self.attention_h));a1 = K.sqrt(K.sum(a1**2,axis=1))#a1 = K.squeeze((K.dot(a1,self.attention_h2)),1)# 
            a = K.concatenate([h2,step_in],axis=1);    a2 = K.tanh(K.dot(a,self.attention_h));a2 = K.sqrt(K.sum(a2**2,axis=1))#a2 = K.squeeze((K.dot(a2,self.attention_h2)),1)# 
            a = K.concatenate([h_tm3,step_in],axis=1); a3 = K.tanh(K.dot(a,self.attention_h));a3 = K.sqrt(K.sum(a3**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_h2)),1)# 
            a = K.concatenate([K.expand_dims(a1,1),K.expand_dims(a2,1)],axis=1); a = K.concatenate([a,K.expand_dims(a3,1)],axis=1)
            a = K.softmax(a); a1 = K.expand_dims(a[:,0],1); a2 = K.expand_dims(a[:,1],1); a3 = K.expand_dims(a[:,2],1); 
            h2 =  K.relu(a1*h_tm1 + a2*h2 + a3*h_tm3)

            a = K.concatenate([c_tm1,step_in],axis=1); a1 = K.tanh(K.dot(a,self.attention_c));a1 = K.sqrt(K.sum(a1**2,axis=1))#a1 = K.squeeze((K.dot(a1,self.attention_c2)),1)# 
            a = K.concatenate([c_tm2,step_in],axis=1); a2 = K.tanh(K.dot(a,self.attention_c));a2 = K.sqrt(K.sum(a2**2,axis=1))#a2 = K.squeeze((K.dot(a2,self.attention_c2)),1)# 
            a = K.concatenate([c_tm3,step_in],axis=1); a3 = K.tanh(K.dot(a,self.attention_c));a3 = K.sqrt(K.sum(a3**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_c2)),1)# 
            a = K.concatenate([K.expand_dims(a1,1),K.expand_dims(a2,1)],axis=1); a = K.concatenate([a,K.expand_dims(a3,1)],axis=1)
            a = K.softmax(a); a1 = K.expand_dims(a[:,0],1); a2 = K.expand_dims(a[:,1],1); a3 = K.expand_dims(a[:,2],1); 
            c2 =  K.relu(a1*c_tm1 + a2*c_tm2 + a3*c_tm3)


        else:
            h2 =  (h_tm1 + h2 + h_tm3)/3 
            c2 =  (c_tm1 + c_tm2 + c_tm3)/3


        i = K.hard_sigmoid(x_i + K.dot(h2,self.recurrent_kernel_i))
        f = K.hard_sigmoid(x_f + K.dot(h2,self.recurrent_kernel_f))
        o = K.hard_sigmoid(x_o + K.dot(h2,self.recurrent_kernel_o))
        m = K.hard_sigmoid(x_c + K.dot(h2,self.recurrent_kernel_c))

        c = (f * c2 + i * m )

        h = (o * K.tanh( c ))

        return h, [h,c,h_tm1,c_tm1,h_next,c_next]   #80.59,81.79



    def call(self, inputs):
        s = K.zeros((K.shape(inputs)[0],self.units))
        init_states = [s,s,s,s,s,s]
        outputs = K.rnn(self.step_do, inputs, init_states)[1]
        '''
        if self.attention:
           self.attention1_1 = self.attention1[:self.units,:]
           self.attention1_2 = self.attention1[self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs[:,i,:]

                h_atten=K.tanh(K.dot(h,self.attention1_1) + 0*self.biase1)     ##################tanh
                h_atten=(K.dot(h_atten,self.attention2))

                h_b=K.tanh(K.dot(step_in,self.attention1_2)+0*self.biase2)     ##################tanh
                h_b=(K.dot(h_b,self.attention2_2))

                h_atten = K.tanh(h_atten*h + h_b)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs = Reshape((inputs.shape[1],self.units))(output_atten)       

        '''
        init_states2 = [s,s,s,s,s,s]
        input2 = K.reverse(inputs,axes=1)
        outputs2 = K.rnn(self.step_do, input2, init_states2)[1]
        '''
        if self.attention:
           self.attention1_1 = self.attention1[:self.units,:]
           self.attention1_2 = self.attention1[self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs2[:,i,:]

                h_atten=K.tanh(K.dot(h,self.attention1_1) + 0*self.biase1)     ##################0
                h_atten=(K.dot(h_atten,self.attention2))

                h_b=K.tanh(K.dot(step_in,self.attention1_2)+1*self.biase2)     ##################1
                h_b=(K.dot(h_b,self.attention2_2))

                h_atten = K.tanh(h_atten*h + h_b)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs2 = Reshape((inputs.shape[1],self.units))(output_atten)   
        '''

        outputs2 = K.reverse(outputs2,axes=1)
        outputs = (K.concatenate([outputs,outputs2]))



        if self.intra_attention:
           self.attention1_1 = self.attention1[:2*self.units,:]
           self.attention1_2 = self.attention1[2*self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs[:,i,:]

                h_atten=K.relu(K.dot(h,self.attention1_1) + 0*self.biase1)     ##################0
                h_atten=(K.dot(h_atten,self.attention2))

                h_b=K.relu(K.dot(step_in,self.attention1_2)+0*self.biase2)     ##################1
                h_b=(K.dot(h_b,self.attention2_2))

                h_atten = K.tanh(1*h_atten*h + 1*h_b)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs = Reshape((inputs.shape[1],2*self.units))(output_atten)   
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.units*2)  #80.80,best_so_far



class ABS_LSTM5_4(Layer):

    def __init__(self, units, intra_attention=True, inter_attention=True, **kwargs):
        self.units = units 
        self.attention = attention
        self.inter_attention = inter_attention
        super(ABS_LSTM5_4, self).__init__(**kwargs)

    def build(self, input_shape): 
        

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units *4 ),
                                      initializer='glorot_normal',
                                      trainable=True)

        self.recurrent_kernel = self.add_weight(
                                shape=(self.units, self.units * 4),
                                name='recurrent_kernel',
                                initializer='glorot_normal',
                                trainable = True
                                                )    
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.inter_attention:

                 self.attention_h = self.add_weight(name='attention_h',
                                      shape=(1*input_shape[-1]+1*self.units, self.units ),
                                      initializer='glorot_normal',
                                      trainable=True)

                 self.attention_c = self.add_weight(name='attention_c',
                                      shape=(1*input_shape[-1]+1*self.units, self.units ),
                                      initializer='glorot_normal',
                                      trainable=True)

        if self.intra_attention:
             
             self.attention1 = self.add_weight(name='attention',
                                      shape=(2*self.units+input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.biase1 = self.add_weight(name='b',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.attention2 = self.add_weight(name='attention2',
                                      shape=(self.units, 1),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.attention2_2 = self.add_weight(name='attention2_2',
                                      shape=(self.units, 1),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.biase2 = self.add_weight(name='b2',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             
    def step_do(self, step_in, states):

        x_i =( K.dot(step_in, self.kernel_i) )
        x_f =( K.dot(step_in, self.kernel_f) ) 
        x_c =( K.dot(step_in, self.kernel_c) )
        x_o = ( K.dot(step_in, self.kernel_o) )
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        h2 = states[2]
        c_tm2 = states[3]
        h_next = h2
        c_next = c_tm2
        h_tm3 = states[4]
        c_tm3 = states[5]
        h_tm4 = states[6]
        c_tm4 = states[7] 
    
        if self.inter_attention:
          #  step_in2 = K.relu(K.dot(step_in,self.transform)+self.biase3)
            a1 = K.concatenate([h_tm1,step_in],axis=1)
            a1 = K.tanh(K.dot(a1,self.attention_h))
            a1 = K.sqrt(K.sum(a1**2,axis=1))#a1 = K.squeeze((K.dot(a1,self.attention_h2)),1)# 
            a2 = K.concatenate([h2,step_in],axis=1)
            a2 = K.tanh(K.dot(a2,self.attention_h))
            a2 = K.sqrt(K.sum(a2**2,axis=1))#a2 = K.squeeze((K.dot(a2,self.attention_h2)),1)# 
            a3 = K.concatenate([h_tm3,step_in],axis=1)
            a3 = K.tanh(K.dot(a3,self.attention_h))
            a3 = K.sqrt(K.sum(a3**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_h2)),1)# 
            a4 = K.concatenate([h_tm4,step_in],axis=1)
            a4 = K.tanh(K.dot(a4,self.attention_h))
            a4 = K.sqrt(K.sum(a4**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_h2)),1)# 
            a = K.concatenate([K.expand_dims(a1,1),K.expand_dims(a2,1)],axis=1); a = K.concatenate([a,K.expand_dims(a3,1)],axis=1); a = K.concatenate([a,K.expand_dims(a4,1)],axis=1)
            a = K.softmax(a); w1 = K.expand_dims(a[:,0],1); w2 = K.expand_dims(a[:,1],1); w3 = K.expand_dims(a[:,2],1); w4 = K.expand_dims(a[:,3],1)
            h2 =  K.relu(w1*h_tm1 + w2*h2 + w3*h_tm3+ w4*h_tm4)

            a1 = K.concatenate([c_tm1,step_in],axis=1); a1 = K.tanh(K.dot(a1,self.attention_c));a1 = K.sqrt(K.sum(a1**2,axis=1))#a1 = K.squeeze((K.dot(a1,self.attention_c2)),1)# 
            a2 = K.concatenate([c_tm2,step_in],axis=1); a2 = K.tanh(K.dot(a2,self.attention_c));a2 = K.sqrt(K.sum(a2**2,axis=1))#a2 = K.squeeze((K.dot(a2,self.attention_c2)),1)# 
            a3 = K.concatenate([c_tm3,step_in],axis=1); a3 = K.tanh(K.dot(a3,self.attention_c));a3 = K.sqrt(K.sum(a3**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_c2)),1)# 
            a4 = K.concatenate([c_tm4,step_in],axis=1); a4 = K.tanh(K.dot(a4,self.attention_h));a4 = K.sqrt(K.sum(a4**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_h2)),1)# 
            a = K.concatenate([K.expand_dims(a1,1),K.expand_dims(a2,1)],axis=1); a = K.concatenate([a,K.expand_dims(a3,1)],axis=1); a = K.concatenate([a,K.expand_dims(a4,1)],axis=1)
            a = K.softmax(a); a1 = K.expand_dims(a[:,0],1); a2 = K.expand_dims(a[:,1],1); a3 = K.expand_dims(a[:,2],1); a4 = K.expand_dims(a[:,3],1)
            c2 =  K.relu(a1*c_tm1 + a2*c_tm2 + a3*c_tm3+ a4*c_tm4)


        else:
            h2 =  (h_tm1 + h2 + h_tm3+ h_tm4)/4 
            c2 =  (c_tm1 + c_tm2 + c_tm3 + c_tm4)/4


        i = K.hard_sigmoid(x_i + K.dot(h2,self.recurrent_kernel_i))
        f = K.hard_sigmoid(x_f + K.dot(h2,self.recurrent_kernel_f))
        o = K.hard_sigmoid(x_o + K.dot(h2,self.recurrent_kernel_o))
        m = K.hard_sigmoid(x_c + K.dot(h2,self.recurrent_kernel_c))

        c = (f * c2 + i * m )

        h = (o * K.tanh( c ))

        return h, [h,c,h_tm1,c_tm1,h_next,c_next,h_tm3,c_tm3]   #80.59,81.79



    def call(self, inputs):
        s = K.zeros((K.shape(inputs)[0],self.units))
        init_states = [s,s,s,s,s,s,s,s]
        outputs = K.rnn(self.step_do, inputs, init_states)[1]

        init_states2 = [s,s,s,s,s,s,s,s]
        input2 = K.reverse(inputs,axes=1)
        outputs2 = K.rnn(self.step_do, input2, init_states2)[1]

        outputs2 = K.reverse(outputs2,axes=1)
        outputs = (K.concatenate([outputs,outputs2]))



        if self.intra_attention:
           self.attention1_1 = self.attention1[:2*self.units,:]
           self.attention1_2 = self.attention1[2*self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs[:,i,:]

                h_atten=K.relu(K.dot(h,self.attention1_1) +0.0*self.biase1 )     ##################0
                h_atten=(K.dot(h_atten,self.attention2))

                h_b=K.relu(K.dot(step_in,self.attention1_2)+0.0*self.biase2)     ##################1
                h_b=(K.dot(h_b,self.attention2_2))

                h_atten = K.tanh(1*h_atten*h + 1*h_b)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs = Reshape((inputs.shape[1],2*self.units))(output_atten)   
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.units*2)  #80.80,best_so_far




class ASB_LSTM6(Layer):

    def __init__(self, units, attention=True, inter_attention=True, **kwargs):
        self.units = units 
        self.attention = attention
        self.inter_attention = inter_attention
        super(ASB_LSTM6, self).__init__(**kwargs)

    def build(self, input_shape): 
        

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units *4 ),
                                      initializer='glorot_normal',
                                      trainable=True)

        self.recurrent_kernel = self.add_weight(
                                shape=(self.units, self.units * 4),
                                name='recurrent_kernel',
                                initializer='glorot_normal',
                                trainable = True
                                                )    
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.inter_attention:

                 self.attention_h = self.add_weight(name='attention_h',
                                      shape=(1*input_shape[-1]+1*self.units, self.units ),
                                      initializer='glorot_normal',
                                      trainable=True)
                 self.attention_c = self.add_weight(name='attention_c',
                                      shape=(1*input_shape[-1]+1*self.units, self.units ),
                                      initializer='glorot_normal',
                                      trainable=True)
 
        if self.attention:
             
             self.attention1 = self.add_weight(name='attention',
                                      shape=(self.units+input_shape[-1], self.units),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.biase1 = self.add_weight(name='b',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.attention2 = self.add_weight(name='attention2',
                                      shape=(self.units, 1),
                                      initializer='glorot_normal',
                                      trainable=True)
             
             self.attention2_2 = self.add_weight(name='attention2_2',
                                      shape=(self.units, 1),
                                      initializer='glorot_normal',
                                      trainable=True)
             '''
             self.biase2 = self.add_weight(name='b2',
                                      shape=(self.units,),
                                      initializer='glorot_normal',
                                      trainable=True)
             '''
    def step_do(self, step_in, states):

        x_i =( K.dot(step_in, self.kernel_i) )
        x_f =( K.dot(step_in, self.kernel_f) ) 
        x_c =( K.dot(step_in, self.kernel_c) )
        x_o = ( K.dot(step_in, self.kernel_o) )
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        h2 = states[2]
        c_tm2 = states[3]
        h_next = h2
        c_next = c_tm2
        h_tm3 = states[4]
        c_tm3 = states[5]
     
        if self.inter_attention:
          #  step_in2 = K.relu(K.dot(step_in,self.transform)+self.biase3)
            a = K.concatenate([h_tm1,step_in],axis=1); a1 = K.tanh(K.dot(a,self.attention_h));a1 = K.sqrt(K.sum(a1**2,axis=1))#a1 = K.squeeze((K.dot(a1,self.attention_h2)),1)# 
            a = K.concatenate([h2,step_in],axis=1);    a2 = K.tanh(K.dot(a,self.attention_h));a2 = K.sqrt(K.sum(a2**2,axis=1))#a2 = K.squeeze((K.dot(a2,self.attention_h2)),1)# 
            a = K.concatenate([h_tm3,step_in],axis=1); a3 = K.tanh(K.dot(a,self.attention_h));a3 = K.sqrt(K.sum(a3**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_h2)),1)# 
            a = K.concatenate([K.expand_dims(a1,1),K.expand_dims(a2,1)],axis=1); a = K.concatenate([a,K.expand_dims(a3,1)],axis=1)
            a = K.softmax(a); a1 = K.expand_dims(a[:,0],1); a2 = K.expand_dims(a[:,1],1); a3 = K.expand_dims(a[:,2],1); 
            h2 =  K.relu(a1*h_tm1 + a2*h2 + a3*h_tm3)

            a = K.concatenate([c_tm1,step_in],axis=1); a1 = K.tanh(K.dot(a,self.attention_c));a1 = K.sqrt(K.sum(a1**2,axis=1))#a1 = K.squeeze((K.dot(a1,self.attention_c2)),1)# 
            a = K.concatenate([c_tm2,step_in],axis=1); a2 = K.tanh(K.dot(a,self.attention_c));a2 = K.sqrt(K.sum(a2**2,axis=1))#a2 = K.squeeze((K.dot(a2,self.attention_c2)),1)# 
            a = K.concatenate([c_tm3,step_in],axis=1); a3 = K.tanh(K.dot(a,self.attention_c));a3 = K.sqrt(K.sum(a3**2,axis=1))#a3 = K.squeeze((K.dot(a3,self.attention_c2)),1)# 
            a = K.concatenate([K.expand_dims(a1,1),K.expand_dims(a2,1)],axis=1); a = K.concatenate([a,K.expand_dims(a3,1)],axis=1)
            a = K.softmax(a); a1 = K.expand_dims(a[:,0],1); a2 = K.expand_dims(a[:,1],1); a3 = K.expand_dims(a[:,2],1); 
            c2 =  K.relu(a1*c_tm1 + a2*c_tm2 + a3*c_tm3)


        else:
            h2 =  (h_tm1 + h2 + h_tm3)/3 
            c2 =  (c_tm1 + c_tm2 + c_tm3)/3


        i = K.hard_sigmoid(x_i + K.dot(h2,self.recurrent_kernel_i))
        f = K.hard_sigmoid(x_f + K.dot(h2,self.recurrent_kernel_f))
        o = K.hard_sigmoid(x_o + K.dot(h2,self.recurrent_kernel_o))
        m = K.hard_sigmoid(x_c + K.dot(h2,self.recurrent_kernel_c))

        c = (f * c2 + i * m )

        h = (o * K.tanh( c ))

        return h, [h,c,h_tm1,c_tm1,h_next,c_next]   #80.59,81.79



    def call(self, inputs):
        s = K.zeros((K.shape(inputs)[0],self.units))
        init_states = [s,s,s,s,s,s]
        outputs = K.rnn(self.step_do, inputs, init_states)[1]
        
        if self.attention:
           self.attention1_1 = self.attention1[:self.units,:]
           self.attention1_2 = self.attention1[self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs[:,i,:]

                h_atten=K.elu(K.dot(h,self.attention1_1) )     ##################tanh
                h_atten=(K.dot(h_atten,self.attention2))

                h_b=K.tanh(K.dot(step_in,self.attention1_2)+ self.biase1)     ##################tanh
                h_b=(K.dot(h_b,self.attention2_2))

                h_atten = K.elu(h_atten*h + h_b)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs = Reshape((inputs.shape[1],self.units))(output_atten)       

        
        init_states2 = [s,s,s,s,s,s]
        input2 = K.reverse(inputs,axes=1)
        outputs2 = K.rnn(self.step_do, input2, init_states2)[1]
        
        if self.attention:
           self.attention1_1 = self.attention1[:self.units,:]
           self.attention1_2 = self.attention1[self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs2[:,i,:]

                h_atten=K.elu(K.dot(h,self.attention1_1) )     ##################0
                h_atten=(K.dot(h_atten,self.attention2))

                h_b=K.tanh(K.dot(step_in,self.attention1_2)+ self.biase1)     ##################1
                h_b=(K.dot(h_b,self.attention2_2))

                h_atten = K.elu(h_atten*h + h_b)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs2 = Reshape((inputs.shape[1],self.units))(output_atten)   
        

        outputs2 = K.reverse(outputs2,axes=1)
        outputs = (K.concatenate([outputs,outputs2]))


        '''
        if self.attention:
           self.attention1_1 = self.attention1[:2*self.units,:]
           self.attention1_2 = self.attention1[2*self.units:,:]
           for i in range(inputs.shape[1]):
                step_in = inputs[:,i,:]
                h = outputs[:,i,:]

                h_atten=K.relu(K.dot(h,self.attention1_1) + 0*self.biase1)     ##################0
                h_atten=(K.dot(h_atten,self.attention2))

                h_b=K.relu(K.dot(step_in,self.attention1_2)+0*self.biase2)     ##################1
                h_b=(K.dot(h_b,self.attention2_2))

                h_atten = K.tanh(1*h_atten*h + 1*h_b)
                if i ==0:
                   output_atten = h_atten
                else:
                   output_atten = K.concatenate([output_atten,h_atten])
           outputs = Reshape((inputs.shape[1],2*self.units))(output_atten) 
        '''  
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.units*2)  #80.80,best_so_far




class ourLSTM(Layer):

    def __init__(self, units, attention=True, inter_attention=True, **kwargs):
        self.units = units 
        self.attention = attention
        self.inter_attention = inter_attention
        super(ourLSTM, self).__init__(**kwargs)

    def build(self, input_shape): 
        

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units *4 ),
                                      initializer='glorot_normal',
                                      trainable=True)

        self.recurrent_kernel = self.add_weight(
                                shape=(self.units, self.units * 4),
                                name='recurrent_kernel',
                                initializer='glorot_normal',
                                trainable = True
                                                )    
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

             
             
    def step_do(self, step_in, states):

        x_i =( K.dot(step_in, self.kernel_i) )
        x_f =( K.dot(step_in, self.kernel_f) ) 
        x_c =( K.dot(step_in, self.kernel_c) )
        x_o = ( K.dot(step_in, self.kernel_o) )
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state


        i = K.hard_sigmoid(x_i + K.dot(h_tm1,self.recurrent_kernel_i))
        f = K.hard_sigmoid(x_f + K.dot(h_tm1,self.recurrent_kernel_f))
        o = K.hard_sigmoid(x_o + K.dot(h_tm1,self.recurrent_kernel_o))
        m = K.tanh(x_c + K.dot(h_tm1,self.recurrent_kernel_c))

        c = (f * c_tm1 + i * m )

        h = (o * K.tanh( c ))

        return h, [h,c]   #80.59,81.79



    def call(self, inputs):
        s = K.zeros((K.shape(inputs)[0],self.units))
        init_states = [s,s]
        outputs = K.rnn(self.step_do, inputs, init_states)[1]
        '''
        init_states2 = [s,s,s,s,s,s,s,s]
        input2 = K.reverse(inputs,axes=1)
        outputs2 = K.rnn(self.step_do, input2, init_states2)[1]

        outputs2 = K.reverse(outputs2,axes=1)
        outputs = (K.concatenate([outputs,outputs2]))
        '''
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.units)  #80.80,best_so_far


class BLSTM(Layer):

    def __init__(self, units, attention=True, inter_attention=True, **kwargs):
        self.units = units 
        self.attention = attention
        self.inter_attention = inter_attention
        super(BLSTM, self).__init__(**kwargs)

    def build(self, input_shape): 
        

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units *8 ),
                                      initializer='glorot_normal',
                                      trainable=True)

        self.recurrent_kernel = self.add_weight(
                                shape=(self.units, self.units * 8),
                                name='recurrent_kernel',
                                initializer='glorot_normal',
                                trainable = True
                                                )    
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:self.units * 4]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:self.units * 4]



        self.kernel_i2 = self.kernel[:, self.units * 4 :self.units * 5]
        self.kernel_f2 = self.kernel[:, self.units * 5: self.units * 6]
        self.kernel_c2 = self.kernel[:, self.units * 6: self.units * 7]
        self.kernel_o2 = self.kernel[:, self.units * 7:]

        self.recurrent_kernel_i2 = self.recurrent_kernel[:, self.units * 4 :self.units * 5]
        self.recurrent_kernel_f2 = self.recurrent_kernel[:, self.units * 5: self.units * 6]
        self.recurrent_kernel_c2 = self.recurrent_kernel[:, self.units * 6: self.units * 7]
        self.recurrent_kernel_o2 = self.recurrent_kernel[:, self.units * 7:]

             
             
    def step_do(self, step_in, states):

        x_i =( K.dot(step_in, self.kernel_i) )
        x_f =( K.dot(step_in, self.kernel_f) ) 
        x_c =( K.dot(step_in, self.kernel_c) )
        x_o = ( K.dot(step_in, self.kernel_o) )
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state


        i = K.hard_sigmoid(x_i + K.dot(h_tm1,self.recurrent_kernel_i))
        f = K.hard_sigmoid(x_f + K.dot(h_tm1,self.recurrent_kernel_f))
        o = K.hard_sigmoid(x_o + K.dot(h_tm1,self.recurrent_kernel_o))
        m = K.tanh(x_c + K.dot(h_tm1,self.recurrent_kernel_c))

        c = (f * c_tm1 + i * m )

        h = (o * K.tanh( c ))

        return h, [h,c]   #80.59,81.79

    def step_do2(self, step_in, states):

        x_i =( K.dot(step_in, self.kernel_i2) )
        x_f =( K.dot(step_in, self.kernel_f2) ) 
        x_c =( K.dot(step_in, self.kernel_c2) )
        x_o = ( K.dot(step_in, self.kernel_o2) )
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state


        i = K.hard_sigmoid(x_i + K.dot(h_tm1,self.recurrent_kernel_i2))
        f = K.hard_sigmoid(x_f + K.dot(h_tm1,self.recurrent_kernel_f2))
        o = K.hard_sigmoid(x_o + K.dot(h_tm1,self.recurrent_kernel_o2))
        m = K.tanh(x_c + K.dot(h_tm1,self.recurrent_kernel_c2))

        c = (f * c_tm1 + i * m )

        h = (o * K.tanh( c ))

        return h, [h,c]   #80.59,81.79

    def call(self, inputs):
        s = K.zeros((K.shape(inputs)[0],self.units))
        init_states = [s,s]
        outputs = K.rnn(self.step_do, inputs, init_states)[1]
        
        init_states2 = [s,s]
        input2 = K.reverse(inputs,axes=1)
        outputs2 = K.rnn(self.step_do2, input2, init_states2)[1]

        outputs2 = K.reverse(outputs2,axes=1)
        outputs = (K.concatenate([outputs,outputs2]))
        
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.units*2)  #80.80,best_so_far








