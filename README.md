# 2025 DIP PJ

This is the repository for Fudan University 25Spring DIP final PJ: Flare-Free Pro: Depth-Pro-Guided Monocular Depth
 Prior for Robust Lens Flare Removal.

## Environment Setup

```bash
conda create -n deflare python=3.10 -y
conda activate deflare
pip install -r requirements.txt
python setup.py develop
# Because of the application of Depth Pro, so it is necessary to clone its repository and install the dependencies into current environment. 
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
source get_pretrained_models.sh
```

## Dataset

You can download Flare7K++ and Flickr24K datasets [here](https://drive.google.com/file/d/1rQ2ZG3HHoBOogYw_qnH3SgLlNlsQtPST/view). The filtered ExDark
dataset can be found [here]().

The datasets should be placed into [dataset](./deflare/dataset/).

## Training

### Training with default Flickr24K + Flare7K++

```bash
python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml
```

### Training with filtered ExDark dataset + Flare7K++

Some modifications should be done before you do so. 

First, copy the content of [dataset.py](./deflare/dataset.py) into [flare7kpp_dataset.py](./deflare/basicsr/data/flare7kpp_dataset.py).

Second, modify line 92 to your path to coordinate json file: 

```python
...
        fixed_flare_coords_json_path = 'PATH/TO/YOUR/new_dataset.json'
        self.flare_coords = {}
...
```

Third, modify line 17 to the path of filtered ExDark dataset in [uformer_flare7kpp_baseline_option.yml](./deflare/options/uformer_flare7kpp_baseline_option.yml).

Finally you can run the training with the same instruction above:

```bash
python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml
```

## Testing

### Inference

Pretrained model weight should be placed into `experiments/flare7kpp/`.

To start inference on test dataset, you can run the following command:

```bash
python basicsr/inference.py --input dataset/Flare7Kpp/test_data/real/input/ --output result/real/pretrained/ --model_path experiments/flare7kpp/pretrained.pth --flare7kpp
```

### Evaluation

To evaluate the performance of the model using PSNR, SSIM, LPIPS, Glare PSNR, and Streak PSNR, you can run the following command:

```bash
python evaluate.py --input result/real/pretrained/blend/ --gt dataset/Flare7Kpp/test_data/real/gt/ --mask dataset/Flare7Kpp/test_data/real/mask/
```