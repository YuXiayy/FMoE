# Dataset Attribution and Licensing

## Dataset used

F-MoE uses the [SonicMaster Dataset](https://huggingface.co/datasets/amaai-lab/SonicMasterDataset) for training and evaluation. The dataset provides paired degraded and high-quality music and metadata describing the source recordings, degradation operations, and data splits.

Official resources:

- Dataset: https://huggingface.co/datasets/amaai-lab/SonicMasterDataset
- Project repository: https://github.com/AMAAI-Lab/SonicMaster
- Project page: https://amaai-lab.github.io/SonicMaster/
- Published paper: https://proceedings.mlr.press/v306/melechovsky26a.html

## Academic citation

If you use the SonicMaster dataset, please cite the original paper:

Jan Melechovsky, Ambuj Mehrish, Abhinaba Roy, and Dorien Herremans. 2026. *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering*. Proceedings of the 43rd International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 306, pp. 87512–87540.

BibTeX (from the PMLR proceedings page):

```bibtex
@InProceedings{pmlr-v306-melechovsky26a,
  title     = {{S}onic{M}aster: Towards Controllable All-in-One Music Restoration and Mastering},
  author    = {Melechovsky, Jan and Mehrish, Ambuj and Roy, Abhinaba and Herremans, Dorien},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {87512--87540},
  year      = {2026},
  editor    = {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/melechovsky26a.html}
}
```

## Dataset and source-recording licences

The Hugging Face dataset card currently identifies the SonicMaster dataset release as **CC BY 2.0**. The SonicMaster project page states that the source recordings were collected from Jamendo under Creative Commons licences using the official Jamendo API.

These are related but distinct licensing layers:

1. **Dataset release:** the licence declaration shown on the SonicMaster Hugging Face dataset card describes the dataset release.
2. **Underlying recordings:** Jamendo's API terms state that each item is published under the Creative Commons licence indicated on that item's Jamendo page. Users must use each item in accordance with its applicable licence and must not misrepresent ownership or the source of the content.
3. **SonicMaster software:** the Apache-2.0 licence displayed on the SonicMaster GitHub software repository applies to that software repository. It is not, by itself, a blanket licence for the music recordings.

Therefore, the dataset-level CC BY 2.0 label should not be interpreted as proof that every underlying recording has identical terms or that every recording can be commercially redistributed or redistributed in modified form. Before sharing original tracks, excerpts, degraded versions, or other audio derived from a source recording, check the applicable source-track licence, attribution requirements, and any restrictions relevant to adaptation or redistribution. If the applicable rights cannot be verified, do not redistribute that audio.

## Distribution policy for this repository

- This repository provides F-MoE source code and documentation; it does not bundle or redistribute SonicMaster audio files or dataset shards.
- Obtain the dataset from its official Hugging Face release.
- Preserve the SonicMaster paper citation and dataset attribution when describing experiments that use the dataset.
- Do not treat the F-MoE software licence as a licence for the dataset or the underlying music.
- Users are responsible for checking and complying with the applicable dataset and source-recording terms for their own use.

This document is an attribution and compliance note, not legal advice. For questions about rights in a particular recording or the scope of the dataset-level licence, consult the dataset authors and the relevant rights holders.
