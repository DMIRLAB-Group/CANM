#' @title CANM with auto search the number latent variables.
#' @description Fit the Latent variable with ANM.
#' @param X The cause variable.
#' @param Y The effect variable.
#' @param N The number of hidden intermediated variables. If N=NULL, it will perform an auto search from minN to maxN.
#' @param epochs The training epochs
#' @param batch_size The batch size
#' @param prior_sdy The initialization of the standard error at noise distribution.
#' @param minN The minimum N of the searching progress. It will search N from minN to maxN.
#' @param maxN The maximum N of the searching progress. It will search N from minN to maxN.
#' @param update_sdy Whether update the noise distribution by using gradient decent.
#' @param seed The random seed
#' @param ... Other parameters for CANM, see CANM.py for more details.
#' @return The likelihood of the model.
#' @export
#' @examples
#' set.seed(0)
#' data=CANM_data(depth=2,sample_size=5000)
#' lxy=CANM(data[,1],data[,2])
#' lyx=CANM(data[,2],data[,1])
#' if(max(lxy$train_score)>max(lyx$train_score)){
#'  print("X->Y")
#' }else{
#'  print("Y->X")
#' }
#'
CANM<-function(X,Y,N=NULL,epochs=50L,batch_size=128L,prior_sdy=0.5,minN=1L,maxN=5L,seed=0L,update_sdy=TRUE,...){

  seed=as.integer(seed)
  epochs=as.integer(epochs)
  minN=as.integer(minN)
  maxN=as.integer(maxN)

  set.seed(seed)

  vae_anm=import_from_path("CANM",path = system.file("python",package="CANM",mustWork = TRUE))
  batch_size=as.integer(batch_size)
  epochs=as.integer(epochs)
  f <- stats::approxfun(stats::density(X,from=min(X),to=max(X)))
  logpx<-mean(log(f(X)))

  if(!is.null(N)){
    if(N<=0){
      stop("N must greater than 0.")
    }
    N=as.integer(N)
    return(vae_anm$fit(traindata = as.matrix(data.frame(X,Y)),
                       N=N,logpx=logpx,epochs = epochs,
                       batch_size=batch_size,prior_sdy=prior_sdy,seed=seed,update_sdy=update_sdy,...))
  }else{
    # Using validation set to selection the N
    data=data.frame(X,Y)
    m=nrow(data)
    train_index<-sample.int(m,0.8*m)
    train_data<-data[train_index,]
    test_data<-data[-train_index,]

    bestN=1L
    bestScore=-Inf
    for (N in minN:maxN) {
      result=vae_anm$fit(traindata = as.matrix(train_data),testdata=as.matrix(test_data),
                         N=N,logpx=logpx,epochs = epochs,batch_size=batch_size,prior_sdy=prior_sdy,seed=seed,update_sdy=update_sdy,...)
      if(is.nan(result$train_likelihood)||is.nan(max(result$test_score))){
        next
      }
      if(max(result$test_score)>bestScore){
        bestScore=max(result$test_score)
        bestN=N
      }
    }
    L<-vae_anm$fit(traindata = as.matrix(data.frame(X,Y)),
                   N=bestN,logpx=logpx,epochs = epochs,
                   batch_size=batch_size,prior_sdy=prior_sdy,seed=seed,update_sdy=update_sdy,...)
    L["bestN"]<-bestN
    L["logpx"]<-logpx
    return(L)

  }
}


#' @title Information-geometric approach to inferring causal directions.
#' @description This is the implementation of the IGCI method for causal discovery.
#' @param x The observation of the cause.
#' @param y The observation of the effect.
#' @param refMeasure reference measure to use: 1: uniform 2: Gaussian
#' @param estimator estimator to use:1: entropy (eq. (12) in [1]),
#'                  2: integral approximation (eq. (13) in [1]).
#'                 3: new integral approximation (eq. (22) in [2]) that should
#'                     deal better with repeated values
#' @return  f < 0:       the method prefers the causal direction x -> y \cr
#'          f > 0:       the method prefers the causal direction y -> x
#' @export
#' @references
#'
#' Janzing, Dominik, et al. "Information-geometric approach to inferring causal directions." Artificial Intelligence 182 (2012): 1-31.
#' @examples
#' set.seed(0)
#' x=rnorm(100)
#' y=exp(x)
#' f=IGCI(x,y,refMeasure=2,estimator=1)
#' if(f<0){
#'  print("X->Y")
#' }else{
#'  print("Y->X")
#' }
#'
IGCI<-function(x,y,refMeasure,estimator){
  m=length(x)
    if (refMeasure==1) {
      # uniform reference measure
      x=scales::rescale(x)
      y=scales::rescale(y)
    }else if(refMeasure==2){
      # gaussian reference measure
      x=scale(x)
      y=scale(y)
    }

  if (estimator==1) {
    # difference of entropies
    sx=sort(x,index.return=T)
    x1=sx$x
    indXs=sx$ix

    sy=sort(y,index.return=T)
    y1=sy$x
    indYs=sy$ix

    n1 = length(x1)
    hx = 0;
    for(i in 1:(n1-1)){
    delta = x1[i+1]-x1[i]
      if (delta){
        hx = hx + log(abs(delta))
      }
    }
    hx = hx / (n1 - 1) + digamma(n1) - digamma(1)

    n2 = length(y1)
    hy = 0
    for (i in 1:(n2-1)){
      delta = y1[i+1]-y1[i];
      if (delta){
        hy = hy + log(abs(delta))
      }
    }

    hy = hy / (n2 - 1) + digamma(n2) - digamma(1)
    f = hy - hx
  }else if(estimator==2){
    # integral-approximation based estimator

    a = 0;
    b = 0;

    s1=sort(x,index.return=T)
    sx=s1$x
    ind1=s1$ix

    s2=sort(y,index.return=T)
    sy=s2$x
    ind2=s2$ix

    for (i in 1:(m-1)){
    X1 = x[ind1[i]];  X2 = x[ind1[i+1]];
    Y1 = y[ind1[i]];  Y2 = y[ind1[i+1]];
    if ((X2 != X1) && (Y2 != Y1)){
    a = a + log(abs((Y2 - Y1) / (X2 - X1)));
    }
    X1 = x[ind2[i]];  X2 = x[ind2[i+1]];
    Y1 = y[ind2[i]];  Y2 = y[ind2[i+1]];
    if ((Y2 != Y1) && (X2 != X1)){
     b = b + log(abs((X2 - X1) / (Y2 - Y1)));
    }
    }
    f = (a - b) / m;
  }else if(estimator==3){
    # integral-approximation based estimator
    # improved handling of values that occur multiple times
    f = (improved_slope_estimator(x,y) - improved_slope_estimator(y,x)) / m
  }
  return(f)
}

improved_slope_estimator<-function(x,y){
  m = length(x)

  s1=sort(x,index.return=T)
  sx=s1$x
  ind=s1$ix

  Xs = c(); Ys = c(); Ns = c();
  last_index = 1;
  for (i in 2:m){
  if (x[ind[i]] != x[ind[last_index]]){
    Xs = c(Xs, x[ind[last_index]])
    Ys = c(Ys, y[ind[last_index]])
    Ns = c(Ns, i - last_index)
    last_index = i;
  }
  }

  Xs = c(Xs, x[ind[last_index]])
  Ys = c(Ys, y[ind[last_index]])
  Ns = c(Ns, m+1 - last_index)

  m = length(Xs)
  a = 0
  Z = 0
  for (i in 1:(m-1)){
  X1 = Xs[i];  X2 = Xs[i+1];
  Y1 = Ys[i];  Y2 = Ys[i+1];
  if ((X2 != X1) && (Y2 != Y1)){
    a = a + log(abs((Y2 - Y1) / (X2 - X1))) * Ns[i]
    Z = Z + Ns[i]
  }
  }
  a = a / Z
  return(a)
}

#' @title Additive noise model using XGBoost regression.
#' @description Additive noise model using the XGBoost regression.
#' @param x The observation of the cause.
#' @param y The observation of the effect.
#' @param booster The regression function, "gbtree" or "gblinear"
#' @param nrounds The number of tree
#' @param gamma The gamma parameter that controls the pruning process. The higher gamma the less overfitting.
#' @param nthread The number of CPU.
#' @param ... Other parameters see ?xgboost
#' @export
#' @examples
#' set.seed(0)
#' x=rnorm(1000)
#' y=exp(x)+rnorm(1000)
#' result=ANM_XGB(x,y)
#' if(result$HSIC_xy<result$HSIC_yx){
#'  print("X->Y")
#' }else{
#'  print("Y->X")
#' }
#'
ANM_XGB<-function(x,y,booster="gbtree",nrounds=30,gamma=0,nthread=1,...){
  model<-xgboost(data=as.matrix(data.frame(x)),label=as.numeric(y),
                 verbose=0,nrounds=nrounds,gamma=gamma,booster=booster,nthread=nthread,...)
  y_hat<-stats::predict(model,as.matrix(data.frame(x)))
  N<-as.numeric(y)-as.numeric(y_hat)
  # fx <- stats::approxfun(stats::density(x,from=min(x),to=max(x)))
  # fn <- stats::approxfun(stats::density(N,from=min(N),to=max(N)))
  # score_xy=sum(log(fn(N)))+sum(log(f(x))) #x->y
  HSIC_xy=hsic.gamma(x,N)

  temp=x
  x=y
  y=temp
  model<-xgboost(data=as.matrix(data.frame(x)),label=as.numeric(y),
                 verbose=0,nrounds=nrounds,gamma=gamma,booster=booster,nthread=nthread,...)
  y_hat<-stats::predict(model,as.matrix(data.frame(x)))
  N<-as.numeric(y)-as.numeric(y_hat)
  # fx <- stats::approxfun(stats::density(x,from=min(x),to=max(x)))
  # fn <- stats::approxfun(stats::density(N,from=min(N),to=max(N)))
  # score_yx=sum(log(fn(N)))+sum(log(f(x)))

  # HSIC_yx=hsic.gamma(x,N)
  HSIC_yx=hsic.gamma(x,N) # HSIC is sensity to the scale of the data, so we need to standardized the data



  return(list(HSIC_xy=HSIC_xy$statistic,p_xy=HSIC_xy$p.value,
              HSIC_yx=HSIC_yx$statistic,p_yx=HSIC_yx$p.value
              ))

}

#' @title Additive Noise Model using MLP regression.
#' @description Additive noise model using the MLP regression.
#' @param x The observation of the cause.
#' @param y The observation of the effect.
#' @param epochs The training epochs
#' @param D_in Dimension of input
#' @param D_H1 Dimension of hidden layer 1
#' @param D_H2 Dimension of hidden layer 2
#' @param D_out Dimension of output
#' @param batch_size Batch size
#' @param cuda Use GPU
#' @param seed Random Seed
#' @param log_interval The option of verbose that output the training detail at each interval.
#' @param learning_rate The learning rate.
#' @param verbose print the output
#' @param ... Other parameters
#' @export
#' @examples
#' set.seed(0)
#' x=rnorm(1000)
#' y=exp(x)+rnorm(1000)
#' result=ANM_MLP(x,y)
#' if(result$HSIC_xy<result$HSIC_yx){
#'  print("X->Y")
#' }else{
#'  print("Y->X")
#' }
ANM_MLP<-function(x,y,epochs=20L,D_in=1L,D_H1=7L,D_H2=5L,D_out=1L, batch_size=128L,cuda=FALSE, seed=0L,
                  log_interval=10L, learning_rate=1e-2,verbose=FALSE,...){

  epochs=as.integer(epochs)
  batch_size=as.integer(batch_size)
  D_in=as.integer(D_in)
  D_H2=as.integer(D_H2)
  D_out=as.integer(D_out)
  seed=as.integer(seed)
  log_interval=as.integer(log_interval)

  mlp=import_from_path("mlp",path = system.file("python",package="CANM",mustWork = TRUE))
  data=data.frame(x,y)
  yhat<-mlp$mlp(traindata = as.matrix(data[,c(1,2)]),epochs = epochs,D_in=D_in,D_H1=D_H1,D_H2=D_H2,D_out=D_out, batch_size=batch_size,cuda=FALSE, seed=seed,
                log_interval=log_interval, learning_rate=learning_rate,verbose=verbose,...)
  N<-as.numeric(y)-as.numeric(yhat)
  HSIC_xy=hsic.gamma(x,N)

  yhat<-mlp$mlp(traindata = as.matrix(data[,c(2,1)]),epochs = epochs,...)
  N<-as.numeric(y)-as.numeric(yhat)
  # fx <- stats::approxfun(stats::density(x,from=min(x),to=max(x)))
  # fn <- stats::approxfun(stats::density(N,from=min(N),to=max(N)))
  # score_yx=sum(log(fn(N)))+sum(log(f(x)))

  # HSIC_yx=hsic.gamma(x,N)
  HSIC_yx=hsic.gamma(x,N) # HSIC is sensity to the scale of the data, so we need to standardized the data

  return(list(HSIC_xy=HSIC_xy$statistic,p_xy=HSIC_xy$p.value,
              HSIC_yx=HSIC_yx$statistic,p_yx=HSIC_yx$p.value
  ))

}
