# Regularizing Attention with Bootstrapping

Bootstrapping, a resampling technique in statistics, provides valuable ways to estimate the distribution of a statistic by repeatedly sampling with replacement from the data. When a statistical distribution is not well characterized, the bootstrap could prove to be  advantageous due to its flexibility and capacity to assess variability and uncertainty in model predictions.

We introduce **attention regularization** which generate a baseline distribution of the attention scores by bootstrapping input features to identify and understand spurious attention attributable to noise rather than informative features. By establishing this distribution, we can estimate the significance of attention scores and calculate local false discovery rates (LFDR), thereby enhancing the model's interpretability.

[https://arxiv.org/abs/2604.01339](https://arxiv.org/abs/2604.01339)
