# wavenet
Implementation of the WaveNet(2016) paper by DeepMind to model character sequences rather than audio.

--

**NOTE: This repository will implement the WaveNet architecture in multiple stages:**

---

**Part 1: building a 3-layer Batchnormalized MLP but with progessive concatenation of inputs to the Linear layers(similar to how it is in the WaveNet paper).**

--

**Part 2: Updating Part 1 with dilated causal convolutions (making it possible to more efficiently train the model on a particular word like 'expelliarmus'). In part 1, we would iterating over 13 possible character contexts like ..e, .ex, exp etc. to train the model on 1 word but not anymore.**

--

**Part 3: Update Part 2 with residual connections and skip connection.**

--

**NOTE: This repo is maintained individual to avoid cramming all the code witihn the nn_zero_to_hero repository. Link to which is:**
    - [nn_zero_to_hero repository](https://github.com/sganesa2/nn_zero_to_hero)
