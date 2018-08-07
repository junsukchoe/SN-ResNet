SN-VGG (GAP)
======
# Network Architectures

*See https://arxiv.org/abs/1512.04150.*

1. Conv, 3x3, 64.
2. Conv, 3x3, 64.
Max Pooling.
3. Conv, 3x3, 128.
4. Conv, 3x3, 128.
Max Pooling.
5. Conv, 3x3, 256.
6. Conv, 3x3, 256.
7. Conv, 3x3, 256.
Max Pooling.
8. Conv, 3x3, 512.
9. Conv, 3x3, 512.
10. Conv, 3x3, 512.
Max Pooling.
11. Conv, 3x3, 512.
12. Conv, 3x3, 512.
13. Conv, 3x3, 512.
14. Conv, 3x3, 1024.
Global Average Pooling.
15. Linear, 1000



# Arguments
### Common (train and test): 필수 입력
* --gpu: (int) 사용할 GPU의 숫자. 콤마(,)로 구분.
* --data: (str) dataset의 저장 위치
* --logdir: (str) log 및 checkpoint 저장 위치

### Common (train and test): 선택 입력
* --epoch: (int) training epoch
* --steps: (int) steps per epoch
* --final-size: (int) input size

### Training arguments
* --batch: (int) batch size
* --sn: (store_true) Spectral normalization 적용
* --GR: (store_true) Googlenet Resize 적용

### Test arguments
* --load: log file address
* --eval: evaluation on validation set

# Training
    $ python codes/main.py --gpu 0 --data /notebooks/dataset/ILSVRC2012 --batch 64 --sn --GR
