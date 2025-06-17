import pandas as pd
import tensorflow as tf
import numpy as np
    #creating an trasnformer 
    #starting with creating an Postioning eccoding which allows the user to give relative postions
def position_encoding(seq_len,dim_model):
        
        #first we will declrare position 
        pos=np.arange(seq_len)[:, np.newaxis]
        #the axis will be like [[1,2,3...{vertically}]
        i=np.arange(dim_model)[np.newaxis, :]
        #the axis defines the number of  dimeison for the model
        angle_value=pos*(1/10000**(2*i/dim_model))
        #this gives the angel radians for each 
        angle_value[:,1::2]=np.cos(angle_value[:,1::2]) #according to formula given in All you need is Attention by googel in 2017
        angle_value[:,::2]=np.sin(angle_value[:,0::2])
        return tf.cast(tf.expand_dims(angle_value, axis=0), dtype=tf.float32)  # 
def Single_dot(q,v,k,dim_model,mask=None):
        q_k_mt=tf.matmul(q,k,transpose_b=True)
        d=tf.cast(tf.shape(k)[-1],tf.float32)
        scaled_logits=q_k_mt/np.sqrt(d)
        if mask is not   None:
                scaled_logits+=(mask*-1e9)
        att_weight=tf.nn.softmax(scaled_logits,axis=-1)
        output=tf.matmul(att_weight,v)
        return output
class Multi_head_att(tf.keras.layers.Layer):
    def __init__(self, dim_model, head):
        super().__init__()
        self.head = head
        self.dim_model = dim_model
        assert dim_model % head == 0, "dim_model must be divisible by head"
        self.depth = dim_model // head

        self.Wq = tf.keras.layers.Dense(dim_model)
        self.Wk = tf.keras.layers.Dense(dim_model)
        self.Wv = tf.keras.layers.Dense(dim_model)
        self.dense = tf.keras.layers.Dense(dim_model)

    def split_head(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.head, self.depth))  # ✅ Must match dimensions
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_head(q, batch_size)
        k = self.split_head(k, batch_size)
        v = self.split_head(v, batch_size)

        scaled_attention = Single_dot(q, v, k, self.dim_model, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dim_model))

        return self.dense(concat_attention)
def Feed_forward_N(dim_model,dff):


        return tf.keras.Sequential([ 
                tf.keras.layers.Dense(dff,activation='relu'),
                tf.keras.layers.Dense(dim_model)  
         ])
class Layering(tf.keras.layers.Layer):
        def __init__(self,dim_model,dff,head,drop_out_rate=0.3):
                super().__init__()
                self.M_head=Multi_head_att(dim_model,head)
                self.Forwrd_n=Feed_forward_N(dim_model,dff)
                self.Lnormal1=tf.keras.layers.LayerNormalization()
                self.Lnormal2=tf.keras.layers.LayerNormalization()
                self.dropout1=tf.keras.layers.Dropout(drop_out_rate)
                self.dropout2=tf.keras.layers.Dropout(drop_out_rate)
        def call(self,x,mask=None,training=False):
                attention_output=self.M_head(x,x,x,mask)
                attention_output=self.dropout1(attention_output)
                normli=self.Lnormal1(x+attention_output)
                ffn=self.Forwrd_n(normli)
                ffn=self.dropout2(ffn)
                return self.Lnormal2(normli+ffn)
class Decoder_layering(tf.keras.layers.Layer):
        def __init__(self,dim_model,dff,head,drop_out_rate=0.3):
                super().__init__()
                self.M_head_1=Multi_head_att(dim_model,head)
                self.M_head_2=Multi_head_att(dim_model,head)
                self.Forwrd_n=Feed_forward_N(dim_model,dff)
                self.Lnormal1=tf.keras.layers.LayerNormalization()
                self.Lnormal2=tf.keras.layers.LayerNormalization() 
                self.Lnormal3=tf.keras.layers.LayerNormalization()
                self.dropout1=tf.keras.layers.Dropout(drop_out_rate)
                self.dropout2=tf.keras.layers.Dropout(drop_out_rate)
                self.dropout3=tf.keras.layers.Dropout(drop_out_rate) 
        def call(self,x,eno,training=False,dec_mask=None,dpadding_mask=None):
                atten1=self.M_head_1(x,x,x,dec_mask)
                #attention_output=self.M_head(x,x,x,mask)
                atten1=self.dropout1(atten1)
                dec_atten_output_1=self.Lnormal1(x+atten1)
                atten2=self.M_head_2(eno,eno,dec_atten_output_1,dec_mask)
                #attention_output=self.M_head(x,x,x,mask)
                atten2=self.dropout2(atten2)
                dec_atten_output_2=self.Lnormal2(dec_atten_output_1+atten2)


                ffn1=self.Forwrd_n(dec_atten_output_2)
                ffn1=self.dropout3(ffn1)
                return self.Lnormal3(dec_atten_output_2+ffn1)
class Encoder(tf.keras.layers.Layer):
        def __init__(self,num_layers,dim_model,head,dff,vocabulary_size,m_seq_len):
                super().__init__()
                self.dim_model=dim_model
                self.pos_encoding=position_encoding(m_seq_len,dim_model)
                self.Embedding=tf.keras.layers.Embedding(vocabulary_size,dim_model)
                self.layer=[Layering(dim_model,dff,head) for _ in range(num_layers) ]
                self.dropout=tf.keras.layers.Dropout(0.1)
        def call(self,x,training=False,mask=None):   
                 seq_lenth=tf.shape(x)[1]
                 x=self.Embedding(x)
                 x*=tf.math.sqrt(tf.cast(self.dim_model,tf.float32))
                 x+=self.pos_encoding[:,:seq_lenth,:]
                 x=self.dropout(x,training=training)

                 for layers in self.layer:
                        x=layers(x,training=training)
                 return x     
class Decoder(tf.keras.layers.Layer):
        def __init__(self,num_layers,dim_model,head,dff,vocabulary_size,m_seq_len):
                super().__init__()
                self.dim_model=dim_model
                self.pos_encoding=position_encoding(m_seq_len,dim_model)
                self.Embedding=tf.keras.layers.Embedding(vocabulary_size,dim_model)
                self.dec_layer=[Decoder_layering(dim_model,dff,head) for _ in range(num_layers) ]
                self.dropout=tf.keras.layers.Dropout(0.1)
        def call(self,x,encoder_output,training=False,amask=None,padding_mask=None):   
                 seq_lenth=tf.shape(x)[1]
                 x=self.Embedding(x)
                 x*=tf.math.sqrt(tf.cast(self.dim_model,tf.float32))
                 x+=self.pos_encoding[:,:seq_lenth,:]
                 x=self.dropout(x,training=training)

                 for layers in self.dec_layer:

                        x=layers(
        x,
        encoder_output,
        training=training,
       
    )
                 return x    
class Transformer(tf.keras.Model):
           def __init__(self,num_layers,dim_model,head,dff,vocabulary_size,m_seq_len,target_vacab_Size):
                   super().__init__()
                   self.encoder=Encoder(num_layers,dim_model,head,dff,vocabulary_size,m_seq_len)
                   self.Decoder=Decoder(num_layers,dim_model,head,dff,vocabulary_size,m_seq_len)
                   self.final_layer=tf.keras.layers.Dense(target_vacab_Size)
           def call(self,inp,tar,training=False,padding_mask=None,amask=None,dpadding_mask=None):
                   ecn_output=self.encoder(x=inp,training=False,mask=padding_mask)#training,padding_mask)
                   dec_output = self.Decoder(
     x=tar,
    encoder_output=ecn_output,
    training=training,
                   )
#amask,dpadding_mask)
                   return self.final_layer(dec_output)
# Instantiate the model
sample_transformer = Transformer(
    num_layers=2,
    dim_model=512,
    head=8,
    dff=2048,
    vocabulary_size=8500,
    m_seq_len=100,
    target_vacab_Size=8000
)
dummy_input = tf.random.uniform((1, 10), minval=0, maxval=8500, dtype=tf.int32)  # (batch_size=1, seq_len=10)
dummy_target = tf.random.uniform((1, 10), minval=0, maxval=8000, dtype=tf.int32)
output = sample_transformer(dummy_input, dummy_target, training=False)
#training=False, padding_mask=None, amask=None,dpadding_mask=None)



# Show output shape and partial content
print("Output shape:", output.shape)           # e.g., (1, 10, 8000)
print("Output sample (first word):", output[0, 0, :10]) 
predicted_ids = tf.argmax(output, axis=-1)
print("Predicted token IDs:", predicted_ids.numpy())
 # logits for first word position



                   

           
                   

        









        

     


                 

                                               


                


    

        
        
    