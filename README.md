# SHM-repo

## Related information
Here is the code of "End-to-end learning the partial permutation matrix for robust 3D point cloud registration" (``https://ojs.aaai.org/index.php/AAAI/article/view/20250``), which proposes a two-stage matching module to achieve end-to-end hard matching.

<!--Note: the code is being prepared. -->

## Implementation
The code is tested with Pytorch 1.6.0 with CUDA 10.2.89. Prerequisites include scipy, h5py, tqdm, etc. Your can install them by yourself.

The ModelNet40 dataset, 3DMatch and KITTI datasets can be download from:
```
https://github.com/WangYueFt/dcp;https://github.com/yewzijian/RPMNet;https://github.com/chrischoy/DeepGlobalRegistration
```

Start training with the command:
```
python main.py 
```

Start testing with the command:
```
python main.py --eval True --mdoel_path YOUR_CHECKPOINT_DIRECTORY
```

Start finetuen with the command:
```
python main.py --finetune True --tune_path YOUR_CHECKPOINT_DIRECTORY
```

## Acknowledgement
The code is insipred by DCP, PRNet, RPMNet, DGR, etc.

## Please cite:
```
@inproceedings{zhang_SHM_AAAI_2022,
  title={End-to-End Learning the Partial Permutation Matrix for Robust 3D Point Cloud Registration},
  author={Zhiyuan Zhang and Jiadai Sun and Yuchao Dai and Dingfu Zhou and Xibin Song and Mingyi He},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}} 
```
