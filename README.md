# CANM: Causal Discovery with Cascade Nonlinear Additive Noise Models

## Overview

This code provides a method to fit the Cascade Nonlinear Additive Noise Models as well as to discover the causal direction on the data that contain the intermediate latent variables. 

Please cite "Ruichu Cai, Jie Qiao, Kun Zhang, Zhenjie Zhang, Zhifeng Hao. Causal Discovery with Cascade Nonlinear Additive Noise Models. IJCAI 2019." 
## Installation

```
install.packages("devtools")
devtools::install_github("dmirgroup/CANM")
```

## Requirement 

This code also needs the python3 and pytorch environment with packages `torch`, `scipy`, `numpy` installed. You may need to select the specify python interpreter in R by using:

```
library(reticulate)
use_python("path/to/python")
py_config() # see the configuration
```

Due to the complexity of this developing environment, we also provide a Dockerfile (`inst/Dockerfile`) for building this environment to run this code. For your convenience we have published the image to the cloud https://hub.docker.com/r/qiaojie/canm. Note that, the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is required for using GPU. 

Fetch the docker image:
```
docker pull qiaojie/canm:latest
```
Or you may want to build it by yourself:
```
cd inst
docker build . -t qiaojie/canm:latest
```

After all, you could run the demo by using:

```
$ nvidia-docker run -it qiaojie/canm
$ R
> library(CANM)
> ?CANM
>      set.seed(0)
>      data=CANM_data(depth=2,sample_size=5000)
>      lxy=CANM(data[,1],data[,2])
>      lyx=CANM(data[,2],data[,1])
>      if(max(lxy$train_score)>max(lyx$train_score)){
+       print("X->Y")
+      }else{
+       print("Y->X")
+      }

```

Or just give up the GPU accelerate and run the image using: `docker run -it qiaojie/canm`


### Quick Start

This package contains the data synthetic process for CANM that present in the paper (see `CANM_data()`). Here are some examples to make a quick start:

```
set.seed(0)
 data=CANM_data(depth=2,sample_size=5000)
 lxy=CANM(data[,1],data[,2])
 lyx=CANM(data[,2],data[,1])
 if(max(lxy$train_score)>max(lyx$train_score)){
  print("X->Y")
 }else{
  print("Y->X")
 }

# IGCI method

set.seed(0)
x=rnorm(100)
y=exp(x)
f=IGCI(x,y,refMeasure=2,estimator=1)
if(f<0){
  print("X->Y")
 }else{
  print("Y->X")
 }
 
# ANM using xgboost regression

set.seed(0)
x=rnorm(1000)
y=exp(x)+rnorm(1000)
result=ANM_XGB(x,y)
if(result$HSIC_xy<result$HSIC_yx){
 print("X->Y")
}else{
 print("Y->X")
}


```

If you are not an R user, you may use the `inst/python/CANM.py` directly, however, it does not include the algorithm for searching the number of latent variables. So you have to specify the number of latent variables `N`.



 
 

