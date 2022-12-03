# SEEG 

This project is a pytorch implementation of *SEEG: Semantic Energized Co-speech Gesture Generation*. 

# Insight 

* Only learning beat gestures already performs comparably with the SOTA methods. 
* Introducing additional semantic-aware supervision can influence the gestures expressions. 

## Environment & Training 

This repository is developed and tested on Ubuntu 18.04, Python 3.6+, and PyTorch 1.3+. The environment is the same to [Trimodal Context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context).

This project is mainly developed based on [Trimodal Context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context). You can run this project by ``` bash train.sh ``` or the same commands in [Trimodal Context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context). 


## Citation

Please cite our CVPR2022 paper if you find our SEEG is helpful in your work:

```
@inproceedings{liang2022seeg,
  title={SEEG: Semantic Energized Co-speech Gesture Generation},
  author={Liang, Yuanzhi and Feng, Qianyu and Zhu, Linchao and Hu, Li and Pan, Pan and Yang, Yi}, 
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
