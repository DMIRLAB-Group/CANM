#' @title Synthetic data for CANM.
#' @description Amore general version for Generating data from transitive ANM.
#' @param depth The depth of the CANM (the number of the latent intermediated variables).
#' @param sample_size The number of samples.
#' @param ratio The ratio of noise.
#' @param k The number of components in the mixture Gaussian distribution of X.
#' @param changepoints The mapping function at each layer are generated from a cubic spline interpolation using a 6-dimensional grid with respect to 6 random generated points as knots for the interpolation; Here the changepoint control the number of knots as controlling the nonlinearity of the mapping function.
#' @param sd The standard error of the random generative knots.
#' @param noisedist The noise distribution at each layer.
#' @param ... Additional parameters for the specify noise distribution.
#' @return Data
#' @export
#' @examples
#' set.seed(0)
#' data=CANM_data(depth=2,ratio=0.2)
#' plot(data[,1],data[,2])
#'
#'
CANM_data<-function(depth=1,sample_size=5000,ratio=1,k=3,changepoints=6,sd=1,noisedist="rnorm",...){
#  k=3
  p=abs(rnorm(k))
  p=p/sum(p)
  components <- sample(1:k,prob=p,size=sample_size,replace=TRUE)
  mus <- rnorm(k)
  sds <- (2*rnorm(k)+1)^2/5
  x <- rnorm(n=sample_size,mean=mus[components],sd=sds[components])

  #x=rnorm(sample_size,sd=5)

  data=data.frame(x=x)
  # changepoints=4
  # sd=3

  if(!(noisedist %in% c("mixgauss","supgauss"))){
    distf=get(noisedist)
  }

  if(depth==0){
    f=splinefun(seq(min(data[,1]),max(data[,1]),length.out = changepoints),rnorm(changepoints,sd=sd))
    data[,2]<-f(data[,1])+distf(sample_size,...)*ratio
  }else{
    for (d in 1:depth) {
      f=splinefun(seq(min(data[,d]),max(data[,d]),length.out = changepoints),rnorm(changepoints,sd=sd))

      if(noisedist=="mixgauss"){
        p=abs(rnorm(k))
        p=p/sum(p)
        components <- sample(1:k,prob=p,size=sample_size,replace=TRUE)
        mus <- rnorm(k)
        sds <- (2*rnorm(k)+1)^2/5
        noise <- rnorm(n=sample_size,mean=mus[components],sd=sds[components])
        noise<- noise-mean(noise)
        data[,d+1]<-f(data[,d])+noise*ratio
      }else if(noisedist=="supgauss"){
        eps=rnorm(sample_size,...)

        data[,d+1]<-f(data[,d])+sign(eps)*eps^2*ratio
      }else if(noisedist=="runif"){
        data[,d+1]<-f(data[,d])+distf(sample_size,-1,1)*ratio
      }else{
        data[,d+1]<-f(data[,d])+distf(sample_size,...)*ratio
      }

    }
    f=splinefun(seq(min(data[,depth+1]),max(data[,depth+1]),length.out = changepoints),rnorm(changepoints,sd=sd))


    if(noisedist=="mixgauss"){
      p=abs(rnorm(k))
      p=p/sum(p)
      components <- sample(1:k,prob=p,size=sample_size,replace=TRUE)
      mus <- rnorm(k)
      sds <- (2*rnorm(k)+1)^2/5
      noise <- rnorm(n=sample_size,mean=mus[components],sd=sds[components])
      noise<- noise-mean(noise)
      data[,depth+2]<-(f(data[,depth+1]))+noise*ratio
    }else{
      data[,depth+2]<-(f(data[,depth+1]))+distf(sample_size,...)*ratio
    }

    data=data[,c(1,2+depth)]
  }
  return(data)
}

