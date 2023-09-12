<h1 align="center">CLIP-TSA: <br> CLIP-Assisted Temporal Self-Attention for <br> Weakly-Supervised Video Anomaly Detection</h1>
<h3 align="center"><a href="https://2023.ieeeicip.org/" target="_blank">IEEE International Conference on Image Processing (ICIP), 2023</h3>
<p align="center">
    <a href="https://arxiv.org/abs/2212.05136" alt="ArXiv">
        <img src="https://img.shields.io/badge/paper-arxiv-orange.svg" />
    </a>
    <a href="https://ieeexplore.ieee.org/document/10222289" alt="Proceedings">
        <img src="https://img.shields.io/badge/paper-proceedings-orange.svg" />
    </a>
    <a href="https://hyekang.info/bibtex/clip-tsa.txt" alt="Cite">
        <img src="https://img.shields.io/badge/cite-bibtex-orange.svg" />
    </a>
     <a href="https://twitter.com/cokecoda" alt="Twitter">
        <img src="https://img.shields.io/twitter/url/https/twitter.com/cokecoda.svg?style=social&label=Follow%20%40cokecoda" />
     </a>
<h5 align="center">The implementation of the paper <br> "CLIP-TSA: CLIP-Assisted Temporal Self-Attention for Weakly-Supervised Video Anomaly Detection" <br> using the PyTorch framework.</h5>
<hr>
<h3>Requirements</h3>
<ul>
    <li>pytorch</li>
    <li>matplotlib</li>
    <li>tqdm</li>
    <li>scipy</li>
    <li>scikit-learn</li>
</ul>
    
<br/>

<h3>FAQ</h3>

> Q1) I get the following error: "RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'"

A1) Please go to venv/lib/python3.8/site-packages/torch/utils/data/sampler.py, and find \_\_iter__ function within RandomSampler class.  Then change the line `generator = torch.Generator()` to `generator = torch.Generator(device="cuda")`.

> Q2) I keep getting CUDA OUT OF MEMORY error

A2) Each dataset requires varying amounts of VRAM, and a significant amount of VRAM is expected to be used with the TSA feature enabled. If you would like to test out only the power of CLIP within the model, please disable the TSA by adding `--disable_HA` to the command. 

## Citations:

```
@article{joo2023cliptsa,
  author = {Joo, Hyekang Kevin and Vo, Khoa and Yamazaki, Kashu and Le, Ngan},  
  doi = {10.48550/ARXIV.2212.05136},
  url = {https://arxiv.org/abs/2212.05136},  
  title = {CLIP-TSA: CLIP-Assisted Temporal Self-Attention for Weakly-Supervised Video Anomaly Detection},  
  publisher = {IEEE International Conference on Image Processing (ICIP), 2023},  
  year = {2023}
}
```


<h4>The codes have been adapted in part from Yu Tian's <a href="https://github.com/tianyu0207/RTFM">RTFM</a>.</h4>
