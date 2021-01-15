<h1>The purpose of this project</h1>
Build a semantic segmentation model with UNet architecture using Keras

<h1>Models that were used</h1>
<h2>Classic Unet</h2>
This model gave score: val_loss: 0.0641 - val_dice_coef: 0.8973 </br>
Sample prediction on value dataset: </br>
<img src="https://raw.githubusercontent.com/Tayge/quantum/master/image/sample_pred_classic_unet.png" alt="unet">

<h2>U-Nets with ResNet Encoders</h2>
Modified Unet with pretrained encoders Resnet50 blocks.  </br>
This model gave score val_loss: 0.0594 - val_dice_coef: 0.9349 </br>
Sample prediction on value dataset: </br>
<img src="https://raw.githubusercontent.com/Tayge/quantum/master/image/sample_pred_resnet_unet.png" alt="resnet+unet">


<h1>Files in the repository</h1>
<b>train.py</b> - script for training all model, after launch, script will offer to choose a model. The resulting execution will create a model file. 
<b>model_classic_unet.h5</b> - Already trained model Classic Unet</br>
<b>preprocessing.py</b> - predict mask, after launch, script will offer to choose a model. </br>
<b>Model creation and analysis.ipynb</b> - Jupiter nootebook with main logic ptoject. </br>
<b>stage_1.zip</b> - Image data. <em>The dataset was taken from here:</em> https://www.kaggle.com/c/data-science-bowl-2018</br>
<b>requirements.txt</b> - Requirements</br>
</br>
Created by Hennadii Horenskyi. horenskyih@outlook.com
