<h1>The purpose of this project</h1>
Build a semantic segmentation model with UNet architecture using Keras

<h1>Models that were used</h1>
<h2>Classic Unet</h2>

U-Net is a convolutional neural network that was developed for biomedical image segmentation. </br>
The network consists of a contracting path and an expansive path, which gives it the u-shaped architecture. The contracting path is a typical convolutional network that consists of repeated application of convolutions, each followed by a rectified linear unit (ReLU) and a max pooling operation. During the contraction, the spatial information is reduced while feature information is increased. The expansive pathway combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the contracting path. Sourсe: <a href="https://arxiv.org/abs/1505.04597"> U-Net: Convolutional Networks for Biomedical Image Segmentation </a> </br>
<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" alt="str_unet">
After training the Unet, the following results were obtained: </br>
This model gave score: val_loss: 0.0641 - val_dice_coef: 0.8973 </br>
Sample prediction on value dataset: </br>
<img src="https://github.com/Tayge/semantic_segmentation/blob/master/image/Pred_Unet.jpg" alt="unet">

<h2>U-Nets with ResNet Encoders</h2>
ResNet is a CNN architecture, made up of series of residual blocks (ResBlocks)  with skip connection. Now by replacing convolutions in U-Net on each level with ResBlock, we can get better performance than the original UNet almost every time. Below is the detailed model architecture diagram. Sourсe:<a href="https://medium.com/@nishanksingla/unet-with-resblock-for-semantic-segmentation-dd1766b4ff66"> UNet with ResBlock for Semantic Segmentation </a></br>
<img src="https://miro.medium.com/max/3544/1*eKrh8FqJL3jodebYlielNg.png" alt="str_resnet+unet">

After training the Unet with ResNet Encoders, the following results were obtained: </br>
This model gave score val_loss: 0.0594 - val_dice_coef: 0.9349 </br>
Sample prediction on value dataset: </br>
<img src="https://github.com/Tayge/semantic_segmentation/blob/master/image/Pred_Unet_Resnet.jpg" alt="resnet+unet">

<h1>Files in the repository</h1>
<b>train.py</b> - script for training all model, after launch, script will offer to choose a model. The resulting execution will create a model file. </br>
<b>model_classic_unet.h5</b> - Already trained model Classic Unet</br>
<b>preprocessing.py</b> - predict mask, after launch, script will offer to choose a model. </br>
<b>Model creation and analysis.ipynb</b> - Jupiter nootebook with main logic ptoject. </br>
<b>stage_1.zip</b> - Image data. <em>The dataset was taken from here:</em> https://www.kaggle.com/c/data-science-bowl-2018</br>
<b>requirements.txt</b> - Requirements</br>
</br>
Created by Hennadii Horenskyi. horenskyih@outlook.com
