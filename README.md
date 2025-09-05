# Mental Health Causal Discovery

Please install required dependencies in `requirements.txt`, and clone repositories [`scm-identify`](https://github.com/dongxinshuai/scm-identify) and [`latent-causal-models`](https://github.com/parjanya20/latent-causal-models) in the causal_discovery directory.

We provide code to generate synthetic data, example scripts to run experiments, as well as the results (`results_all`) we obtained as part of the work. We are unable to provide the mental health dataset due to information security. 

Data generation:
`$ python causal_discovery/synthetic_data.py`

RLCD:
`$ python causal_discovery/rlcd_experiment.py -t <test> <graph>`


Differentiable Causal Discovery:
`$ python causal_discovery/diff_causal_experiment.py all`


## References

Xinshuai Dong, Biwei Huang, Ignavier Ng, Xiangchen Song, Yujia Zheng, Songyao Jin,
Roberto Legaspi, Peter Spirtes, and Kun Zhang. A versatile causal discovery frame-
work to allow causally-related hidden variables. In Proceedings of the 12th International
Conference on Learning Representations, 2024. URL `https://openreview.net/
forum?id=FhQSGhBlqv`.

Xinshuai Dong, Ignavier Ng, Boyang Sun, Haoyue Dai, Guang-Yuan Hao, Shunxing Fan,
Peter Spirtes, Yumou Qiu, and Kun Zhang. Permutation-based rank test in the presence
of discretization and application in causal discovery with mixed data. In Proceedings
of the 42nd International Conference on Machine Learning, 2025. URL `https://
openreview.net/forum?id=VBTHduhm4K`.

Parjanya Prashant, Ignavier Ng, Kun Zhang, and Biwei Huang. Differentiable causal
discovery for latent hierarchical causal models. In Proceedings of the 13th International
Conference on Learning Representations, 2025. URL `https://openreview.net/
forum?id=Bp0HBaMNRl`.
