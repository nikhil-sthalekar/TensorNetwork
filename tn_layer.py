from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.enable_v2_behavior()
# Import tensornetwork
import tensornetwork as tn
# Settting backend to tensorflow otherwise numpy is usedB

tn.set_default_backend("tensorflow")

class TNLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(TNLayer, self).__init__()
        # Creating variables for the layer.
        self.a_var = tf.Variable(
            tf.random.normal(
                shape=(8, 8, 2),
                stddev=1.0/16.0
            ),
            name="a",
            trainable=True
        )
        self.b_var = tf.Variable(
            tf.random.normal(
                shape=(8, 8, 2),
                stddev=1.0/16.0
            ),
            name="b",
            trainable=True
        )
        
        self.bias = tf.Variable(
            tf.zeros(shape=(8, 8)),
            name="bias",
            trainable=True
        )

    def call(self, inputs):
        # Define the contraction
        # Break it out so we can parallelize a batch using 
        # tf.vectorized_map
        def f(input_vec, a_var, b_var, bias_var):
            # Reshaping to a matrix instead of a vector.
            input_vec = tf.reshape(input_vec, (8,8))

            # Creating the network
            a = tn.Node(a_var)
            b = tn.Node(b_var)
            x_node = tn.Node(input_vec)
            tn.connect(a[1], x_node[0])
            tn.connect(b[1], x_node[1])
            tn.connect(a[2], b[2])

            # The TN now looks like this
            #  |    |
            #  a -- b
            #   \   |
            #       x

            # Contracting the nodes
            c = a @ x_node
            result = (c @ b).tensor 
            

            # Adding in the bias layer
            return result + bias_var

        # To deal with a batch of items, we can use the tf.vectorized_map fnc
        result = tf.vectorized_map(
            lambda vec: f(vec, self.a_var, self.b_var, self.bias), inputs)

        return tf.nn.swish(tf.reshape(result, (-1, 64)))


Dense = tf.keras.layers.Dense


tn_model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(64, activation=tf.nn.swish),
        # Here we have out Matrix Product State  in lieu of a dense layer
        TNLayer(),
        Dense(1, activation=None)
    ]
)

summary = tn_model.summary()

print(summary)

X = np.concatenate(
        [
            np.random.randn(20, 2) + np.array([3, 3]),
            np.random.randn(20, 2) + np.array([-3, -3]),
            np.random.randn(20, 2) + np.array([-3, 3]),
            np.random.randn(20, 2) + np.array([3, -3]),
        ]
)

Y = np.concatenate([np.ones((40)), -np.ones((40))])

tn_model.compile(optimizer="adam", loss="mean_squared_error")

fit = tn_model.fit(X, Y, epochs=300, verbose=1)

#print(fit)

# Plotting Code
h = 1.0
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5

# Making a meshgrid for plotting
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h),
)

#print(xx)
#print(xx.shape)
# The Predictio


input =  np.c_[xx.ravel(), yy.ravel()]
print(X.shape)
print(xx.shape)
print(yy.shape)
print(input.shape)
Z = tn_model.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z.shape)

Z = Z.reshape(xx.shape)
print(Z.shape)

# Putting the results in a color plot



plt.contourf(xx, yy, Z)



plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)



plt.savefig('ex1.png')
plt.show()
#print(Z)
#print(Z.shape)
