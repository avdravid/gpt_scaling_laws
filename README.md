# Scaling Laws of Neural Language Models

![kaplan_scaling](assets/kaplan_scaling.png)

This repository contains an implementation of scaling laws as first found by Kaplan et al in [Scaling Laws of Neural Language Models](https://arxiv.org/abs/2001.08361). We find how the training loss of a language model scales with parameter count, dataset size, total compute, and the number of training steps. We also find the optimal number of parameters for a given compute budget, and the critical batch size at which a language model must be trained for the optimal trade-off between time efficiency and compute efficiency (as suggested in [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) by McCandlish, Kaplan, et al).

Our results qualitatively match their results in all cases. We also obtain quantitative matches in some cases despite significant differences in our experimental settings (for example, we train much smaller models, and our dataset OpenWebText is different from the WebText dataset used internally at OpenAI). Perhaps, the most interesting result is that we find a similar scaling law for the dependence of optimal model size $N_{\text{opt}}$ on a compute budget $C$ as they do:

$$N_{\text{opt}} \propto C^{0.72}$$

[Kaplan et al](https://arxiv.org/abs/2001.08361) obtained $N_{\text{opt}} \propto C^{0.73}$. 

We carried out all of our experiments with [nanoGPT](https://github.com/karpathy/nanoGPT). The repository is accompanied by a [Weights and Biases Project](https://wandb.ai/shehper/owt-scaling) containing the training curves of all experiments. The analysis of our results is presented in the Jupyter notebook [kaplan_scaling_laws.ipynb](kaplan_scaling_laws.ipynb). 

> **Note:** We carried out experiments at a scale much smaller than that of [Kaplan et al](https://arxiv.org/abs/2001.08361) due to computational constraints. They varied model size over six orders of magnitude with the largest model having 1.5B parameters, while we could vary model size over only three orders of magnitude (with the largest model having 6.3M parameters). It's nice that already at this small scale, we can see all of the trends they discovered as we review below. If you would like to see the results of this repository extended to larger models and have compute available, please get in touch!

> :warning: Since their work, [Hoffman et al](https://arxiv.org/abs/2203.15556) have discovered that with a different choice of training configurations, the optimal model size grows much more slowly as a function of compute, i.e. $N_{\text{opt}} \propto C^{0.5}$. We are currently working on reproducing their results.

# main results

## scaling laws with parameter count, dataset size, and compute

[Kaplan et al](https://arxiv.org/abs/2001.08361) discovered that the test loss of a language model scales as a power law with the number of (non-embedding) model parameters $N$, the number of tokens $D$ in the training dataset, and the amount of compute $C$ used to train a model: 

$$L(X) = \left( \frac{X_c}{X} \right)^{\alpha_X}$$

where $X = N, D,$ or $C$. When scaling with $N$ or $D$, the non-scaling variable is fixed at a large value so that it is not a bottleneck in affecting the model performance. If both $N$ and $D$ are allowed to vary, the scaling laws $L(N)$ and $L(D)$ can be combined into a single equation:

$$L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{ \frac{\alpha_N}{\alpha_D} } + \frac{D_c}{D}  \right]^{\alpha_D}$$

We trained several models with varying model and dataset sizes, using (almost) the same set of hyperparameters as theirs, and found good fits to these equations. (See the figure below.)

![our_scaling](assets/merged_image.png)

While the trends match up qualitatively, there are differences in some scaling exponents and coefficients. (See the table below.) In retrospect, this was expected as our training dataset is different from theirs, and hence it is not meaningful to compare test losses. However, we note that some results are strikingly close. For example, our $\alpha_N$ is the same as theirs in the fit of $L(N, D)$, and is close to theirs in the fit of $L(N)$ (0.082 vs 0.076). This is perhaps evidence for a statement that $\alpha_N$ is independent of the choice of a dataset as long as it is large enough (though it might depend on other choices such as that of a tokenizer or hyperparameter configurations).

<div align="center">

| Equation                  | Kaplan et al's fit | Our fit | 
| :---------------- | :------: | ----: |
| $L(N) = (N_c/N)^{\alpha_N}$  | $$\alpha_N = 0.076$$ $$N_c = 8.8 \times 10^{13} $$  | $$\alpha_N = 0.082 $$ $$N_c = 3.8 \times 10^{13} $$  |  
| $L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}  \right]^{\alpha_D}$         |   $$\alpha_N = 0.076$$  $$\alpha_D = 0.103$$ $$N_c = 6.4 \times 10^{13}  $$ $$D_c = 1.8 \times 10^{13}  $$   | $$\alpha_N = 0.076$$ $$\alpha_D = 0.122$$ $$N_c = 1.32 \times 10^{14}  $$ $$D_c = 1.22 \times 10^{12}  $$  |   
| $L(C) = (C_c/C)^{\alpha_C}$  |   $$\alpha_C = 0.057$$ $$C_c = 1.6 \times 10^{7} $$   |  $$\alpha_C = 0.074$$ $$C_c = 1.8 \times 10^{5}$$ |

</div>


[Kaplan et al](https://arxiv.org/abs/2001.08361) used their results to obtain optimal model size as a function of compute, i.e. the number of parameters that, for a fixed amount of compute, achieve the lowest value of test loss. They found that it obeys a power law:

$$N^{\text{Kaplan}}_{\text{opt}}(C) = 1.6 \times 10^9 \ C^{0.88}$$

We found a close match with their result,  

$$N^{\text{nanoGPT}}_{\text{opt}}(C) = 2.8 \times 10^9 \ C^{0.90}$$

indicating that while the test loss depends on the choice of a dataset, the trend in the optimal number of parameters as a function of compute is largely independent of it. 

> **Remark**: Note that we did not find a fit to $L(D)$ as this requires training a very large model. [Kaplan et al](https://arxiv.org/abs/2001.08361) found $\alpha_D$ and $D_c$ in the fit to $L(D)$ to be quite close to the ones in the fit to $L(N, D)$. We suspect that the same should be true for us.


## critical batch size

<p align="middle">
  <img src="./assets/critical_batch.png" width="440" />
</p>

The optimal number of parameters, $N_{\text{opt}}(C)$, is only optimal if we were to train all models at the same fixed batch size (and other hyperparameter configurations) used to conduct our scaling laws experiments. 

But what if this batch size is too large? Generally speaking, increasing batch size reduces noise in gradient descent but this benefit dies off once the batch size crosses some threshold value. Training at a batch size larger than this threshold is wasteful of compute, as it does not help obtain significantly better performance. Training at smaller batch sizes, on the other hand, could save us some compute (at the cost of increasing the number of training steps). 


The trade-off between time-efficiency and compute-efficiency happens around a task-dependent *critical batch size* $\mathcal{B}_{\text{crit}}$ as observed by [McCandlish et al](https://arxiv.org/abs/1812.06162). This batch size is independent of the model size but it depends on the target loss value. [Kaplan et al](https://arxiv.org/abs/2001.08361) found the following scaling law for critical batch size as a function of training loss in a language modeling task:

$$
\mathcal{B}^{\text{Kaplan}}_{\text{crit}}(L) = 2.0 \times 10^8 \ L^{-4.76}
$$

We conducted several experiments to obtain critical batch size for language modeling with nanoGPT and obtained:

$$
\mathcal{B}^{\text{nanoGPT}}_{\text{crit}}(L) = 2.2 \times 10^7 \ L^{-4.26}
$$
 

## optimal allocation of compute budget

<p align="middle">
  <img src="./assets/Kaplan_LCmin.png" width="400" /> 
  <img src="./assets/Kaplan_NCmin.png" width="400" /> 
</p>

[Kaplan et al](https://arxiv.org/abs/2001.08361) used their estimate of critical batch size to adjust the scaling laws with compute. That is, if they were to train at a batch size much smaller than the critical batch size for maximal compute-efficiency, their scaling laws $L(C)$ and $N_{\text{opt}}(C)$ would be different. We performed the same exercise. A comparison of our results is given below. 

<div align="center">

| Equation                  | Kaplan et al's fit | Our fit | 
| :---------------- | :------: | ----: |
| $L(C_{\min}) = (C^{\min}_c/C)^{\alpha^{\min}_C}$  |   $$\alpha^{\min}_C = 0.050$$  $$C^{\min}_c = 3.1 \times 10^{8}$$   |  $$\alpha^{\min}_C = 0.056$$ $$C^{\min}_c = 4.2 \times 10^{6}$$ |
| $N_{\text{opt}}(C_{\text{min}}) = N_e C^{p_N}_{\min} $  | $$p_N= 0.73$$ $$N_e = 1.3 \times 10^{9}$$  | $$p_N= 0.72$$ $$N_e = 4.2 \times 10^{9}$$  |  

</div>

Significantly, while many of the scaling exponents and coefficients differ, perhaps the most important of them all --- $p_N$, the scaling exponent of optimal model size with compute budget, is almost the same (0.73 vs 0.72). 

This scaling law for $N_{\text{opt}}(C_{\text{min}})$ says that for a 10x increase in compute budget, the model size must increase $10^{0.72} \sim 5.24$ times. How should we distribute the rest of the increase in compute budget over increases in batch size and the number of training steps? [Kaplan et al](https://arxiv.org/abs/2001.08361) find that the batch size and the minimum of training steps $S_{\text{min}}$ to obtain a specific loss value must increase with compute as

$$B^{\text{Kaplan}} \propto C_{\min}^{0.24}; \quad  S_{\text{min}}^{\text{Kaplan}} \propto C_{\min}^{0.03}$$

We find:

$$B^{\text{nanoGPT}} \propto C_{\min}^{0.24}; \quad  S_{\text{min}}^{\text{nanoGPT}} \propto C_{\min}^{0.04}$$

See the Jupyter Notebook [kaplan_scaling_laws.ipynb](kaplan_scaling_laws.ipynb) for more details.

## install

This repository is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT). In addition to their dependencies, we used matplotlib, scikit-learn, and scipy for analysis of results. You may install all dependencies with

```
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib scikit-learn scipy
```

## reproduction

To reproduce scaling laws, we must first tokenize the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/) dataset as in nanoGPT.

```
python data/openwebtext/prepare.py
```

This downloads and tokenizes the dataset, and stores the output in `train.bin` and `val.bin` files, which hold the [OpenAI Byte-Pair Encoding](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) token ids in one sequence, stored as raw uint16 bytes.

Next, we included two new configuration files in the 'config' folder, which mimic the training configurations of [Kaplan et al](https://arxiv.org/abs/2001.08361). [scale_gpt.py](config/scale_gpt.py) may be used to run experiments with varying model sizes and subsets of datasets. For example, to change the model size, run

```
python train.py config/scale_gpt.py --scale_N=True --n_layer=8 --n_embd=64    
```

and to change the model size as well as the fraction of the dataset, run

```
python train.py config/scale_gpt.py --scale_D=True  --n_layer=4 --n_embd=64 --fraction_of_data=0.01 
```

The former run may be used to model $L(N)$, and the latter may be used to model $L(N, D)$.

To measure critical batch size, use the other configuration file [estimate_critical_batch.py](config/estimate_critical_batch.py) to run experiments with various batch sizes and learning rates. 

```
python train.py config/estimate_critical_batch.py --batch_size=8 --gradient_accumulation_steps=1 --learning_rate=1e-2  
```

Loss curves for all experiments are available on the associated [Weights and Biases project page](https://wandb.ai/shehper/owt-scaling). Use filters 'scale_N=True', 'scale_D=True', and 'estimate_B_crit=True' to select the experiments used to model $L(N)$, $L(N, D)$, and $\mathcal{B}_{\text{crit}}(L)$ respectively. Results are analyzed and fit to these trends in [kaplan_scaling_laws.ipynb](kaplan_scaling_laws.ipynb).

All models for scaling experiments were trained either on one A100 GPU for 3-4 days, or on two A100 GPUs for ~2 days using PyTorch DataDistributedParallel. For estimating critical batch size, we trained models with varying batch sizes. When the batch size was sufficiently small, we could train the model on an RTX 2080 Ti or an RTX 3090s in a few hours. For larger batch sizes, we had to use one A100 GPU for roughly 1 day. 

## todos

As in [nanoGPT](https://github.com/karpathy/nanoGPT), we present some directions for future work and improvements. 

- Reproduce Chinchilla Scaling Laws with nanoGPT (work in progress)
- Train larger models, extending scaling laws to higher orders of magnitude
- Our fits for $S_{\min}$ and $E_{\min}$ do not look great. Improve on these fits.
- Use (Chinchilla) scaling laws and critical batch size to train a large model (1B+ parameters) in a compute-optimal way.
- Experiment with different parameterizations. Does [maximal-update](https://arxiv.org/abs/2203.03466) or [neural-tangent](https://arxiv.org/abs/2304.02034) parameterization have a significant impact on scaling trends?
- Perform ablation studies. Determine how various scaling coefficients and exponents depend on various choices made in our experiments, such as dataset, tokenizer, learning rate schedules, etc.

## acknowledgments

I would like to thank [Andrej Karpathy](https://karpathy.ai/) for the beautiful set of lectures in [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series, especially the lecture on [building GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=8&t=868s) and the [nanoGPT](https://github.com/karpathy/nanoGPT) repository. Using these resources, I went from zero to non-zero in a matter of a few months and the journey forward continues.

I would also like to thank the administrators of the Amarel Cluster at Rutgers University for providing free access to the cluster to all members of the Rutgers community, and for providing prompt help whenever required. Without free access to these resources, the experiments for this repository would not have been possible. 
