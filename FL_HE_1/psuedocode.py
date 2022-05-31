"""
Pseudocode

Func orchestrator:
    - init model
    - train_round(), on all clients in par
    - collect params
    - fedAvg

Func train_round(loc, train_params)
    - data = data(loc)
    - tf.model.train()

Func prepair_data()
    - for i in bc
        * 85% train
        * 15% test

"""



"""
Stappen:
Client -> encrypt train data -> train model -> send weights
Server -> FedAvg, ..? Weigths -> test new computed weights

Test data?
-> Encrypt -> Send -> test model
-> new model -> test locally? 

"""

#%%

# location_list = ['Cleveland', 'Switzerland', 'VA', 'Hungarian']
# def create_keras_model():
#   return tf.keras.models.Sequential([
#       tf.keras.layers.Dense(
#           1,
#           activation='sigmoid',
#           input_shape=(NUM_FEATURES,),
#           kernel_regularizer=tf.keras.regularizers.l2(0.01),
#       )
#   ])
# def model_fn():
#   keras_model = create_keras_model()
#   return tff.learning.from_keras_model(
#       keras_model,
#       input_spec=hungarian_tf.element_spec,
#       loss=tf.keras.losses.BinaryCrossentropy(),
#       metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#                tf.keras.metrics.AUC(name='auc')])

# iterative_process = tff.learning.build_federated_averaging_process(
#     model_fn,
#     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
#     server_optimizer_fn=lambda: tf.keras.optimizers.Nadam(learning_rate=0.5),
#     use_experimental_simulation_loop = True
# )

# state = iterative_process.initialize()
# tff_model = create_keras_model()
# tff_auc = defaultdict(lambda:0)



# for tf in fs:
  
#     # federated_train_data, federated_test_data, labels_train, labels_test = data_prep(df)
#     state, metrics = iterative_process.next(state, federated_train_data)
#     print( str(metrics))
#     state.model.assign_weights_to(tff_model)
#     labels_proba = tff_model.predict(federated_test_data)
#     fpr, tpr, threshold = sklearn.metrics.roc_curve(labels_test, labels_proba)
#     test_loss = tf.keras.losses.binary_crossentropy(labels_test, np.reshape(labels_proba, [-1]))
#     print('validation auc={}, loss={}'.format(sklearn.metrics.auc(fpr, tpr), test_loss))

# %%
# def process_data(tf):
#     # inputs = {}
#     # for name, column in tf.items():
#     #     if type(column[0]) == str:
#     #         dtype = tf.string
#     #     elif (name in categorical_feature_names or
#     #             name in binary_feature_names):
#     #         dtype = tf.int64
#     #     else:
#     #         dtype = tf.float32

#     #     inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

#     # preprocessed = []

#     # for name in binary_feature_names:
#     #     inp = inputs[name]
#     #     inp = inp[:, tf.newaxis]
#     #     float_value = tf.cast(inp, tf.float32)
#     #     preprocessed.append(float_value)

#     numeric_features = hungarian_df[numeric_feature_names]

#     normalizer = tf.keras.layers.Normalization(axis=-1)
#     # normalizer.adapt(stack_dict(dict(numeric_features)))

#     numeric_inputs = {}
#     for name in numeric_feature_names:
#     numeric_inputs[name]=inputs[name]

#     numeric_inputs = stack_dict(numeric_inputs)
#     numeric_normalized = normalizer(numeric_inputs)

#     preprocessed.append(numeric_normalized)

#     for name in categorical_feature_names:
#     vocab = sorted(set(hungarian_df[name]))
#     print(f'name: {name}')
#     print(f'vocab: {vocab}\n')

#     if type(vocab[0]) is str:
#         lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
#     else:
#         lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

#     x = inputs[name][:, tf.newaxis]
#     x = lookup(x)
#     preprocessed.append(x)

#     return preprocess




    # self.weights = np.zeros(X.shape[1])

    # def fit(self, n_iter, eta=0.01):
    #     """Linear regression for n_iter"""
    #     for _ in range(n_iter):
    #         gradient = self.compute_gradient()
    #         self.gradient_step(gradient, eta)

    # def gradient_step(self, gradient, eta=0.01):
    #     """Update the model with the given gradient"""
    #     self.weights -= eta * gradient

    # def compute_gradient(self):
    #     """Compute the gradient of the current model using the training set
    #     """
    #     delta = self.predict(self.X) - self.y
    #     return delta.dot(self.X) / len(self.X)

    # def predict(self, X):
    #     """Score test data"""
    #     return X.dot(self.weights)


    # def encrypted_gradient(self, sum_to=None):
    #     """Compute and encrypt gradient.

    #     When `sum_to` is given, sum the encrypted gradient to it, assumed
    #     to be another vector of the same size
    #     """
    #     gradient = self.compute_gradient()
    #     encrypted_gradient = encrypt_vector(self.pubkey, gradient)

    #     if sum_to is not None:
    #         return sum_encrypted_vectors(sum_to, encrypted_gradient)
    #     else:
    #         return encrypted_gradient