"""
Subtask 3.1: Prepare a short presentation (around 200 words) on a data science topic of your choice.
It could be a specific algorithm, tool, or concept. Please write it in English, focusing on clarity and coherence.

My Answer:
I want to explain generative pretrained models which are so popular nowadays.
GPTs uses specific neural network models named transformers.
Transformer models are also neural networks but have a little different architecture.
They have encoding and decoding parts.

In encoding part,
it has a multi-head self-attention mechanism that separates the input and focuses on different parts of that.
It calculates weights and attention parameters for each part with multiple heads.
Then the attention outputs passed through a neural network.

In the decoding part, it has 2 multi-head self-attention layers and a neural network layer.
The first self-attention layer takes the output of the encoder layer.
And the second takes the shifted inputs but as masked to prevent the model looking ahead.
And finally, a feed-forward neural network that takes all the outputs before himself.

In the final layer,
the output of the decoder transformed a series that has probabilities of each word in the embedding vocabulary.
This architecture has revolutionized the field of NLP by providing a more serializable,
and scalable approach to sequence modeling.
They have led to state-of-the-art models like BERT and GPT and others and continue to be a key area of research,
and development in deep learning.



Subtask 3.2: Imagine you explaining a complex data science concept to a non-technical audience.
Choose a concept (e.g., Neural Networks),
and write a brief explanation (around 150 words) that simplifies the idea without using technical jargon.

My Answer:
I will explain the shining topic of the data science field today. Its name is neural networks.
Neural networks aka deep learning consist of lots of points named neurons and connections between them named weights.
This is a computer learning method and needs to be fed by vast amounts of data.
We order the neurons layer by layer and make connections only between layers.
Neurons in a layer cannot be connected to each other,
and neurons of layers can only have connections with the previous and the next layer adjacent to itself.
We can increase or decrease the number of layers,
and the number of neurons in each layer depending on our data size and complexity.
The complexity increases, the number of layers also increases.
When the architecture of the network is ready, we can start training. We can feed the data row by row to the network.
The first layer takes the data and multiplies it with connections aka weights.
A numerical value named bias summed the result of the multiplication,
and the result passed through a function that detects the importance of the result.
Softmax, relu, sigmoid, and tanh are some of them.
This operation is processed for each layer.
And the output layer we get results.
These results show us a picture contains a cat or dog, the weather is rainy or shiny,
or our stock price will increase or decrease.
"""