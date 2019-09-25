# COCO-GAN

Authors' implementation of "COCO-GAN: Generation by Parts via Conditional Coordinating" in TensorFlow.

[**\[Project Page\]**](https://hubert0527.github.io/COCO-GAN/)
[**\[Paper\]**](http://bit.ly/COCO-GAN)
[**\[Paper (Full Resolution)\]**](http://bit.ly/COCO-GAN-full)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coco-gan-generation-by-parts-via-conditional/image-generation-on-celeba)](https://paperswithcode.com/sota/image-generation-on-celeba?p=coco-gan-generation-by-parts-via-conditional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coco-gan-generation-by-parts-via-conditional/image-generation-on-lsun-bedroom-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-bedroom-256-x-256?p=coco-gan-generation-by-parts-via-conditional)

## 0. Pre-requirements
```
Tensorflow==1.9.0~1.13.0
```

Note: Other Tensorflow versions may be applicable, but you may face some errors during the FID calculation.

You may directly disable FID calculation, or try to comment/uncomment different variants of `TensorShape` setting codes I used for different TF versions at `fid_utils/fid.py:86~89`.

You may need to write your own if you face errors with TF version <1.5.0 or >1.11.0.

## 1. Data Download and Preprocessing

#### [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
You may directly run the following command to setup everythong:
```
python scripts/download_celeba.py
```

#### [LSUN (Bedroom)](https://github.com/fyu/lsun)
You may directly run the following command to setup everythong:
```
sh scripts/download_lsun_bedroom.sh
```

#### [Matterport3D](https://niessner.github.io/Matterport/)

1. You must first contact the authors of Matterport3D dataset, sign an user license agreement, then obtain the downaload script from the authors.
2. After you obtained the `download_mp.py`, download the `matterport_skybox_images` category and place it under `./data/matterport3d/`. Please refer to the official document: `https://github.com/niessner/Matterport/blob/master/data_organization.md`.
3. Install [cube2sphere](https://github.com/Xyene/cube2sphere) library with `pip install cube2sphere`.
4. Execute the preprocessing script with `python scripts/preprocessing_mp3d.py`, which outputs the final images at `./data/matterport3d_panorama/`.

Note: I don't have access to the original dataset now, so I'm not entirely sure about the detailed filenames and paths. You may need to make some small modifications in `scripts/preprocessing_mp3d.py:14:16`. 

#### CelebA-syn
A synthesis variant of CelebA that mess-ups the alignment of faces in CelebA dataset.
```
python ./scripts/generate_celeba_syn --data_dir="./data/CelebA/*.jpg" --syn_type="inward"
python ./scripts/generate_celeba_syn --data_dir="./data/CelebA/*.jpg" --syn_type="outward"
```

#### [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)
I forgot how I downloaded it. I believe there was a script somewhere though...

Please manually download it from the official site: https://github.com/tkarras/progressive_growing_of_gans

## 2. Convert TF-Records
To speed up training, we process the images into tfrecords:
```
# Run the one(s) you need
python ./scripts/compute_tfrecord.py --dataset celeba --resolution 64
python ./scripts/compute_tfrecord.py --dataset celeba --resolution 128
python ./scripts/compute_tfrecord.py --dataset lsun --resolution 64
python ./scripts/compute_tfrecord.py --dataset lsun --resolution 256
python ./scripts/compute_tfrecord.py --dataset matterport3d
python ./scripts/compute_tfrecord.py --dataset celeba-hq
```

If your data is in a directory other than `./data/`, another way to do the conversion is:
```
python ./scripts/compute_tfrecord.py --dataset celeba --resolution 64 --img_paths="./path/to/images/*.jpg"
```

## 3. Pre-calculate FID statistics
Calculation of FID requires some statistic from real data, you may disable FID calcuation in the config files.
```
python ./fid_utils/precalc_fid_stats.py --dataset lsun --data_path "./data/lsun/*" 
```
This will generate statistic files under `./stats/`.

## 4. Training
We provide the configurations for standard COCO-GAN training on CelebA, LSUN and Matterport3D:
```
python main.py --config="./configs/CelebA_128x128_N2M2S64.yaml"
python main.py --config="./configs/LSUN_256x256_N2M2S128.yaml"
python main.py --config="./configs/MP3D_258x768_N2M2S128.yaml"
```
Note: `N,M` is the number of micro patches in a macro patch, `S` is the macro patch size. 

---

CelebA-HQ is not well-tested (actually, we only tune the hyperparameters for CelebA):
```
python main.py --config="./configs/CelebAHQ_1024x1024_N2M2S512.yaml"
```

---

Also the extrapolation experiment (this requires a well-pretrained `LSUN_256x256_N2M2S128` checkpoint):
```
python main.py --config="./configs/LSUN_256x256_N2M2S128_Extrapolation.yaml"
```
Note: This experiment is a little unstable, I usually run 3 to 5 epochs and pick the best model by personal preference.

---

With the **Patch-Guided Image Generation**, we need to train an additional critic `Q` by setting `Q_update_period=1` and `code_loss_w=100`. Here's an exmple config:
```
python main.py --config="./configs/CelebA_128x128_N2M2S64_PatchGuidedGeneration.yaml"
```

## Pretrained checkpoints

We provide following pretrained checkpoints:
- CelebA_64x64_N2M2S32
- CelebA_128x128_N2M2S64
- CelebA_128x128_N2M2S64_PatchGuidedGeneration
- LSUN_64x64_N2M2S32
- LSUN_256x256_N2M2S128

Please download them from: https://drive.google.com/drive/folders/1Mr5BknOrTebQgxdARxJVV95pZA-NkVXt?usp=sharing

Then, you can use `force_load_from_dir: "path/to/the/pretrained/directory"` argument in each of the config file to load the parameters (this does not override the hyperparameters).

Note:
These models are trained with Tensorflow 1.13.0, you may potentially face some loading errors if you use other Tensorflow versions.

## Some Additional Notes
- Some computers take forever to calculate FID, it is usually an unknown problem in the CPU, please switch a computer or disable FID calculation!
- Though our implementation theretically supports (a) patch size H!=W, and (b) micro patch N!=M cases, but these functions are never tested. Please let me know (or create a pull request with a fix if you are willing) if you face any problem while using this feature!
- In general, with smaller micro patches, as the computation graph becomes too complex for Tensorflow, it will take lots of time an GPU memory to build the graph. The performance may be improved with Pyotrch and using `torch.no_grad()` smartly. 
- This implementation is different to our private codebase, please kindly let me know if you found anything bizzard.
- The coordinate generation part looks complicated since I made it generic to different coordinate designs.

## Special Thanks

The following open-source repositories largely facilitate our research!
- Spectral Normalization: 
  https://github.com/minhnhat93/tf-SNDCGAN
- cGANs with Projection Discriminator (we replicate its architecture): 
  https://github.com/pfnet-research/sngan_projection
- FID: https://github.com/bioinf-jku/TTUR
- Gradient Penalty: 
  https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
- Some GAN-Utilities: 
  https://github.com/carpedm20/DCGAN-tensorflow
- Cube to Sphere transform: 
  https://github.com/Xyene/cube2sphere

## Citation
```
@inproceedings{lin2019cocogan,
  author    = {Chieh Hubert Lin and
               Chia{-}Che Chang and
               Yu{-}Sheng Chen and
               Da{-}Cheng Juan and
               Wei Wei and
               Hwann{-}Tzong Chen},
  title     = {{COCO-GAN:} Generation by Parts via Conditional Coordinating},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2019},
}
```






