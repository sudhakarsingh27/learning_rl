# Playing with OpenAI Gym

### What's this 

Tried to convert the Q-table based Q-learning algorithm ([discussed here](https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f
)) to a neural network based solution.

- [x] Change `mountain_car_basic.py` to `mountain_car_nn.py`.
- [x] Use jax based neural network



### Results

1. Visible shortcomings:
    - How to train the network with masked loss (not with a scalar but only propagate the loss on the `cur_action`'s path)
    - Using a very small neural network right now
2. Future scope:
    - [ ] [More bare-metal solution](http://www.pinchofintelligence.com/introduction-openai-gym-part-2-building-deep-q-network/)
    - [ ] [Keras based solution](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)
    - [ ] [Ray based solution](https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google)