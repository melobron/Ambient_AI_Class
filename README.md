# Ambient_AI_Class (2022 Summer)
1. Lightweight CNN for Image Classification
- Inception
- Xception
- MobileNets (2017): Depthwise Separable Convolution
- MobileNets v2 (2018): Inverted Residual and Linear Bottleneck
- MnasNet (2019): NAS with block-wise search space
- EfficientNet (2019): NAS with compound scaling

2. 2-Stage Object Detection
- RCNN (2014)
- SPPNet (2015)
- Fast RCNN (2015)
- Faster RCNN
- Mask RCNN (2017)
- FPN (2017)

3. 1-Stage Object Detection
- YoLo (2016)
- SSD (2016)
- RetinaNet (2017)
- Pelee (2018)
- EfficientDet (2020)
- FOTS (2018): object + text detection
- CRAFT (2019): object + text detection

4. DNN Quantization
- BinaryConnect (2015): binarize weight (16bit to 1bit)
- BNN (2016): binarize activation
- XNOR-Net (2016): quantize  CNN weight
- Integer-only (2018): quantization-aware training

5. DNN Pruning
- Han et.al (NIPS 2015): weight thresholding
- Han et.al (ICLR 2016): storing pruned model
- Gradual Pruning
- Frankle & Carbin (ICLR 2019): find sub-network
- Liu et.al (ICLR 2019): find sub-network
- TensorFlow Lite model optimization
- Pycoral API

6. Knowledge Distillation
- Self Distillation

7. Edge-included system design and applications
- Use both Smartphone and Cloud
	- edge: latency-sensitive, lightweight tasks
	- cloud: latency-tolerant, heavyweight tasks
- MARVEL: Mobile AR
- EagleEye
- MARLIN

8. 3D object classification and detection
- PointNet
- PointNet++
- Frustum PointNet
- VoteNet

9. On-device distributed learning (Federated Learning)
- Client selection
- Global model distribution (cloud to edge)
- Local model update
- Aggregate multiple edge model weights (instead of data)

10. On-device distributed learning (Split Learning)
- each edge train a part of the global model

11. Meta Learning
- N-way K-shot: discriminate between N classes with K examples of each
- Optimization-based Meta Learning
	- MAML (2017)
- Non-parametric Meta Learning
	- learn an embedding space


