<h1 align="center"><a href="https://ieeexplore.ieee.org/document/10222289">CLIP-TSA: <br> CLIP-Assisted Temporal Self-Attention for <br> Weakly-Supervised Video Anomaly Detection</a></h1>
<h3 align="center"><a href="https://2023.ieeeicip.org/">IEEE International Conference on Image Processing (ICIP), 2023</a></h3>
<div align="center" class="font-size: 40px;"><strong><em>Oral Presentation</em></strong></div><br />

<p align="center">
    <a href="https://arxiv.org/abs/2212.05136" alt="ArXiv">
        <img src="https://img.shields.io/badge/paper-arxiv-orange.svg" />
    </a>
    <a href="https://ieeexplore.ieee.org/document/10222289" alt="Proceedings">
        <img src="https://img.shields.io/badge/paper-proceedings-orange.svg" />
    </a>
<!--     <a href="https://hyekang.info/bibtex/clip-tsa.txt" alt="Cite">
        <img src="https://img.shields.io/badge/cite-bibtex-orange.svg" />
    </a> -->
     <a href="https://twitter.com/kevinhjoo" alt="Twitter">
        <img src="https://img.shields.io/twitter/follow/KevinHJoo" />
     </a>
<div align="center">The repository discusses the implementation of the paper <br> "CLIP-TSA: CLIP-Assisted Temporal Self-Attention for Weakly-Supervised Video Anomaly Detection" <br> using the PyTorch framework.</div>
<hr>
<h3>Paper</h3>
    
> [**CLIP-TSA: CLIP-Assisted Temporal Self-Attention for Weakly-Supervised Video Anomaly Detection**](https://arxiv.org/pdf/2212.05136.pdf) (Oral, ICIP 2023)
>
> *[**Kevin Hyekang Joo**](https://hyekang.info/), Khoa Vo, Kashu Yamazaki, Ngan Le*

<h3>Requirements</h3>
<ul>
    <li>pytorch</li>
    <li>matplotlib</li>
    <li>tqdm</li>
    <li>scipy</li>
    <li>scikit-learn</li>
</ul>

<h3>CLIP Features</h3>

- [ShanghaiTech Campus Dataset](https://drive.google.com/file/d/1FvU8-qiVwiGF5BXAdM00-YhMZ7xt_vvy/view?usp=sharing)
- [UCF-Crime Dataset](https://drive.google.com/file/d/1bsVTixDxWdycDJhcTwqZV75suFrv76LB/view?usp=sharing)
- [XD-Violence Dataset](https://drive.google.com/file/d/1HdN4_RcxvSp5scJ4k1PDgHHSZpEhGoZp/view?usp=sharing)
    
<h3>FAQ</h3>

> Q1) I get the following error: "RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'"

- A1) Please go to venv/lib/python3.8/site-packages/torch/utils/data/sampler.py, and find \_\_iter__ function within RandomSampler class.  Then change the line `generator = torch.Generator()` to `generator = torch.Generator(device="cuda")`.

> Q2) I keep getting CUDA OUT OF MEMORY error

- A2) Each dataset requires varying amounts of VRAM, and a significant amount of VRAM is expected to be used with the TSA feature enabled. Thus, please be advised if you want to run tests on big public datasets such as ShanghaiTech Campus, XD-Violence, and UCF-Crime Datasets. If you would like to test out only the power of CLIP within the model, please disable the TSA by adding `--disable_HA` to the command, which requires less amount of VRAM and should be operable on most GPUs. 

<h3>How to Run</h3>

> python main.py

Please change the hyperparameters & parameters accordingly by first looking at the main.py file. Otherwise, it will be run under default settings.

<hr>

<h3>Citations</h3>

```
@inproceedings{joo2023cliptsa,
  title={CLIP-TSA: CLIP-Assisted Temporal Self-Attention for Weakly-Supervised Video Anomaly Detection},
  author={Joo, Hyekang Kevin and Vo, Khoa and Yamazaki, Kashu and Le, Ngan},
  doi={10.1109/ICIP49359.2023.10222289},
  url={https://ieeexplore.ieee.org/document/10222289}
  publisher={IEEE International Conference on Image Processing (ICIP)},
  pages={3230--3234},
  year={2023},
  organization={IEEE}
}
```

<h3>Contacts</h3>

*Kevin Hyekang Joo - khjoo@usc.edu or hkjoo@umd.edu*

<hr>

<h4>The codes have been adapted in part from Yu Tian's <a href="https://github.com/tianyu0207/RTFM">RTFM</a>.</h4>
