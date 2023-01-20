""" Parameters to define the CNN model"""

parameter_names = ["learning_rate", "training_iters", "batch_size", "l2_regularization_penalty", "dropout_keep_prob",
                       "fc1_n_neurons", "fc2_1_n_neurons", "fc2_2_n_neurons", "fc2_3_n_neurons",
                       "conv1_kernel", "conv2_kernel", "conv3_kernel",
                       "conv1_filters", "conv2_filters", "conv3_filters",
                       "conv1_stride", "conv2_stride", "conv3_stride",
                       "pool1_kernel", "pool2_kernel", "pool3_kernel",
                       "pool1_stride", "pool2_stride", "pool3_stride"]
parameters = [
# First column: Keeps the best best parameter based on accuracy score
# Other columns contain the parameter options to try


        # learning_rate
        [0.000500,0.00002,         0.0005, 0.0007, 0.0010, 0.0030, 0.0050, 0.0070],
        # training_iters
        [100000,100000],
        # batch_size
        [400,700,           400, 500, 600, 700, 850, 1000],
        # l2_regularization_penalty
        [0.005,0.005,         0.01, 0.008, 0.005, 0.003],
        # dropout_keep_prob
        [0.9,0.98,          0.75, 0.9, 0.95, 0.98, 1],
        # fc1_n_neurons
        [500,350,           200, 350, 500, 700, 900, 1500],
        # fc2_1_n_neurons
        [700,200,           200, 350, 500, 700, 900, 1500],
        # fc2_2_n_neurons
        [500,350,           200, 350, 500, 700, 900, 1500],
        # fc2_3_n_neurons
        [150,150,           200, 350, 500, 700, 900, 1500],
        # conv1_kernel
        [40,32,            20, 22, 24, 26, 28, 32, 40, 48, 54],
        # conv2_kernel
        [32,16,            10, 14, 16, 20, 24, 28, 32, 34],
        # conv3_kernel
        [20,16,            10, 14, 16, 20, 24, 28, 32, 34],
        # conv1_filters
        [100,100,           64, 80, 90, 100, 110, 120, 140, 160, 200],
        # conv2_filters
        [256,96,            80, 96, 128, 192, 256],
        # conv3_filters
        [128,96,            80, 96, 128, 192, 256],
        # conv1_stride
        [5,3,             2, 3, 4, 5, 6, 8],
        # conv2_stride
        [2,1,             1, 2, 3, 4, 5, 6],
        # conv3_stride
        [1,1,             1, 2, 3, 4, 5, 6],
        # pool1_kernel
        [7,7,             3, 4, 5, 6, 7, 8, 9],
        # pool2_kernel
        [4,6,             4, 5, 6, 7, 8, 9, 10],
        # pool3_kernel
        [6,6,             4, 5, 6, 7, 8, 9, 10],
        # pool1_stride
        [1,4,             1, 2, 4, 5, 6],
        # pool2_stride
        [5,4,             1, 2, 3, 4, 5, 6, 7, 8],
        # pool3_stride
        [6,4,             1, 2, 3, 4, 5, 6, 7, 8]
    ]
