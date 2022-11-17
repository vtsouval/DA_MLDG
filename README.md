# DA_MLDG: Domain-shift Aware MLDG in SER

Speech Emotion Recognition (SER) refers to the recognition of human emotions from natural speech, vital for building human-centered context-aware intelligent systems. Here, domain shift, where models' trained on one domain exhibit performance degradation when exposed to an unseen domain with different statistics, is a major limiting factor in SER applicability, as models have a strong dependence on speakers and languages  characteristics used during training. Meta-Learning for Domain Generalization (MLDG) has shown great success in improving models' generalization capacity and alleviate the domain shift problem in the vision domain; yet, its' efficacy on SER remains largely explored. In this work, we propose a ``<em>Domain-shift Aware</em>'' MLDG approach (DA-MLDG) to learn generalizable models across multiple domains in SER. Based on our extensive evaluation, we identify a number of pitfalls that contribute to poor models' DG ability, and demonstrate that log-mel spectrograms representations lack distinct features required for MLDG in SER. We further explore the use of appropriate features to achieve DG in SER as to provide insides to future research directions for DG in SER.

<div align="center">
	<p class="figure-caption"> <b>Overview of DA-MLDG in SER </b></p>
	<img src="img/overview.png" style="width:70%" alt="DA-MLDG Overview"/>
</div>


## Executing experiments
From the root directory of this repo, run:

```console
foo@bar:~$ python run.py
```

> **_NOTE:_**  You can adjsut the exexuted methods from with `methods` parameter [here](./run.py#L9).

## Reference

If you use this repository, please consider citing:

<pre>@misc{Charles2013,
  author={King Gandhi, Raeshak, and Tsouvalas, Vasileios and Meratnia, Nirvana},
  title={On applicability of Meta-Learning to Speech Emotion Recognition},
  year = {2022},
  publisher = {GitHub},
  url = {https://github.com/vtsouval/DA_MLDG},
}</pre>
