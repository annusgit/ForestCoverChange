# AI-ForestWatch
The aim of this project is to use Landsat-8 imagery to perform forest cover change detection in the Billion Tree Tsunami Afforestation Regions in Pakistan. We do binary land cover segmentation of an image into forest/non-forest classes for our Areas of Interest (AOI), then repeat the same for a whole 7-year temporal series of images from 2014 to 2020 and lastly, compare them to see what forestation changes occured in selected areas. The selected image below shows our results for Battagram district from 2014 to 2020, where red pixels are non-forest labels, green pixels are forest labels and the last image shows overall gain/loss map from 2014 to 2020.

<p align='center'>
    <img src='results/final-battagram-change.png' width="600" height="600"/>
</p>

Our paper contains much more detailed explanation of our methodology, dataset retrieval and preparation, Machine Learning application, model design and band combinations used in our experiments. PDF of the paper is available as `jars-spie-accepted-work.pdf` in the main repository and it may be accessed [online](https://www.spiedigitallibrary.org/journals/journal-of-applied-remote-sensing/volume-15/issue-02/024518/AI-ForestWatch--semantic-segmentation-based-end-to-end-framework/10.1117/1.JRS.15.024518.full) at JARS website.


## Results
We analyse the following labelled regions in Pakistan from 2014 to 2020.
<p align='center'>
    <img src='results/final-regions.png' width="600" height="600"/>
</p>
Essentially, we extract a per-pixel median image representative of a full year for every given region from Landsat-8. This is done in order to minimize effect of clouds and other weather sensitivities in the results. Google Earth Engine was heavily utilized for the retrieval and preprocessing of data. The pipeline including the preprocessing and there onwards is summarized in the following diagram.
<p align='center'>
    <img src='results/final-pipeline.png' width="600" height="600"/>
</p>

## Getting Started

All of the models in this repo are written with [pytorch](https://github.com/pytorch/pytorch).

### Dependencies

You will need the following modules to get the code running

* [pytorch](https://github.com/pytorch/pytorch)
* [torchsummary](https://github.com/sksq96/pytorch-summary)
* [torchviz](https://github.com/szagoruyko/pytorchviz)
* [gdal](https://pypi.org/project/GDAL/)
* [imgaug](https://github.com/aleju/imgaug)

# Usage
## Inference on custom sentinel images downloaded from [Sentinel](https://scihub.copernicus.eu/) or EarthExplorer.
```
cd location/of/ForestCoverChange/
python -m pipeline.detect --images path/to/folder/containing/images/ --model_type model_name --channels number_of_channels_depending_on_the_model --trained_model path/to/pretrained/model --bands list_of_bands_to_use --save_dir path/to/save/results/ --device cpu_or_gpu
``` 
**For example,**
```
cd /home/annus/PycharmProjects/ForestCoverChange/
python -m pipeline.detect --images /home/annus/Desktop/13bands_european_image_time_series/ --model_type VGG_N --channels 3 --trained_model patch_classification/trained_models/vgg3.pt --bands 4 3 2 --save_dir temp-3 --device cpu
``` 
Both pretrained models VGG_3 and VGG_5 are available in `patch_classification/trained_models/`

## Authors

* **Annus Zulfiqar**

See also the list of [contributors](https://github.com/annusgit/ForestCoverChange/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

* [Sentinel](https://scihub.copernicus.eu/) program for providing complete coverage of Earth for free
* [Eurosat](https://arxiv.org/pdf/1709.00029.pdf) for providing a labeled dataset
* [Pytorch Forum](http://discuss.pytorch.org/) for providing valuable help when needed







