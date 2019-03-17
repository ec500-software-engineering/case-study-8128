# Case Study: TensorFlow

## Brief Intro

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

This is my case study repository for tensorflow.

- Language used: Python

- Environment: Anaconda3.7

- Framwork/Libraries: Tensorflow, Pandas, TensorFlow-gpu, cudas

- Prerequisites:

  Make sure you have python installed in your computer

  ```bsh
   pip install tensorflow==2.0.0-alpha0 
  ```

  If you want to use to use the gpu version

  ```bsh
  pip install tensorflow-gpu  # stable
  
  pip install tf-nightly-gpu  # preview
  ```

  The anaconda version of python is preferred , as it will help you set up the driver. Use the command under instead

  ```bsh
  conda install tensorflow-gpu 
  ```

![1552819146984](D:\git\case-study-8128\documentation\tensorflow-ecosystem.jpg)

##### Why python is chosen as the language of the tensorflow?

Many people would consider python as a rather slow language. Actually, for the most part, the core is not written in Python: it's written in a combination of highly-optimized C++ and CUDA (Nvidia's language for programming GPUs). Much of that happens, in turn, by using [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (a high-performance C++ and CUDA numerical library) and [NVidia's cuDNN](https://developer.nvidia.com/cudnn) (a very optimized DNN library for [NVidia GPUs](https://developer.nvidia.com/cuda-gpus), for functions such as [convolutions](https://en.wikipedia.org/wiki/Convolutional_neural_network)).

The model for TensorFlow is that the programmer uses "some language" (most likely Python!) to express the model. This model, written in the TensorFlow constructs such as:

```
h1 = tf.nn.relu(tf.matmul(l1, W1) + b1)
h2 = ...
```

is not actually executed when the Python is run. Instead, what's actually created is a [dataflow graph](https://www.tensorflow.org/get_started/graph_viz) that says to take particular inputs, apply particular operations, supply the results as the inputs to other operations, and so on. *This model is executed by fast C++ code, and for the most part, the data going between operations is never copied back to the Python code*.

Then the programmer "drives" the execution of this model by pulling on nodes -- for training, usually in Python, and for serving, sometimes in Python and sometimes in raw C++:

```
sess.run(eval_results)
```

This one Python (or C++ function call) uses either an in-process call to C++ or an [RPC](https://en.wikipedia.org/wiki/Remote_procedure_call) for the distributed version to call into the C++ TensorFlow server to tell it to execute, and then copies back the results.

**So, with that said, let's re-phrase the question: Why did TensorFlow choose Python as the first well-supported language for expressing and controlling the training of models?**

The answer to that is simple: Python is probably *the* most comfortable language for a large range of data scientists and machine learning experts that's also that easy to integrate and have control a C++ backend, while also being general, widely-used both inside and outside of Google, and open source. Given that with the basic model of TensorFlow, the performance of Python isn't that important, it was a natural fit. It's also a huge plus that [NumPy](http://www.numpy.org/) makes it easy to do pre-processing in Python -- also with high performance -- before feeding it in to TensorFlow for the truly CPU-heavy things.

There's also a bunch of complexity in expressing the model that isn't used when executing it -- shape inference (e.g., if you do matmul(A, B), what is the shape of the resulting data?) and automatic [gradient](https://en.wikipedia.org/wiki/Gradient) computation. It turns out to have been nice to be able to express those in Python, though I think in the long term they'll probably move to the C++ backend to make adding other languages easier.

Quote from <https://stackoverflow.com/questions/35677724/tensorflow-why-was-python-the-chosen-language>

