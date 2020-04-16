"""
The goal of this code is to make a function that can run the MCMC sampling to
fit a diffusion model with a degassing boundary condition to line profiles for
water in olivine.

Parameters:
Diffusivity- Elizabeth's but with some noise; this can include temperature error based on Dan's system
Partitioning Coefficient- Ol/melt
Time or dP/dt: This is the hardest to include. The model needs to be evaluated
for the first 2 parameters before time is fit

Proposed workflow:
Sample D and K --> fit best time to data.
    - step though time with model for given parameters
        - evaluate liklihood at each time
        - stop when liklihood increases below a threshold from maximum.

    - Can I handle this with PyMC3 or do I need my own sampler?
        Determine if PYMC can plot can track deterministic varables in this way
        What way is that exactly?
            Iterating through a loop and stopping at a certain threshold.

An alternate approach might be a work around but could work.
2 level Heirarchical model.
Level 1) Evaluates D and K into an array of diffusion profiles that stop when the max value is below the average of the data points by some threshold.

Level 2) Picks times from a discrete range of times to get the most likely.
            - This range can be made smaller by ony including profiles that have near the max of the DataFrame

For all of these I need to make sure to interpolate the profiles to match the x-coordiantes of the data

"""

"""
Example of this type of model from PyMc3 website
https://docs.pymc.io/notebooks/multilevel_modeling.html

with Model() as partial_pooling:

    # Priors
    mu_a = Normal('mu_a', mu=0., sigma=1e5)
    sigma_a = HalfCauchy('sigma_a', 5)

    # Random intercepts
    a = Normal('a', mu=mu_a, sigma=sigma_a, shape=counties)

    # Model error
    sigma_y = HalfCauchy('sigma_y',5)

    # Expected value
    y_hat = a[county]

    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sigma=sigma_y, observed=log_radon)

model_to_graphviz(partial_pooling)
"""
