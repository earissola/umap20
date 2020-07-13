# A Dataset for Research on Depression in Social Media

Language provides a unique windowinto thoughts, enabling direct assessment of mental-state alterations. Due to their increasing popularity, online social media platforms have become promisingmeans to study different mental disorders. However, the lack ofavailable datasets can hinder the development of innovative diagnostic methods. Tools to assist health practitioners in screeningand monitoring individuals under potential risk are essential.

In this paper, we present a new a dataset to foster the research onautomatic detection of depression. To this end, we present a methodology for automatically collecting large samples of depression and non-depression posts from online social media. Furthermore, we perform a benchmark on the dataset to establish a point of referencefor researchers who are interested in using it. More details are available in our paper [1].

## Methodology
Consider a set of social media users that we have definitive knowledge that they are suffering from depression. This knowledge could come from a survey or it could be self-declared. Given a chronology of textual posts and based on previous findings in the literature, we propose different heuristics to characterise depression signs and use this information to automatically selecting posts for building the dataset.

Let D<sup>+</sup> be the candidate set of positive posts samples. We retrieve such posts from a set of users suffering from depression. Since the goal is to filter out less useful messages, we define two heuristics:

* Filtering posts by their sentiment polarity score
* Filtering posts by their topical similarity with a depression taxonomy

Let D<sup>-</sup> be the control posts samples, that are the posts not providing any reference to depression signs. Such posts are randomly collected from a set of users which are not affected by the mental disorder.

## Citation
```
@inproceedings{Rissola:2020_umap,
    title = {A Dataset for Research on Depression in Social Media},
    author={\textbf{Esteban Andr{\'{e}}s R{\'{\i}}ssola} and Seyed Ali Bahrainian and Fabio Crestani},
    booktitle = {Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization},
    series = {UMAP'20},
    address = {Genoa, Italy, July 14-17},
    year = {2020}
}
```

## References
[1] Esteban Andrés Ríssola, Seyed Ali Bahrainian, and Fabio Crestani. 2020. A dataset for research on depression in social media. In Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization, UMAP’20.