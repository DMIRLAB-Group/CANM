FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# set mirror
RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial main restricted" > /etc/apt/sources.list  \
&& echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-updates main restricted" >> /etc/apt/sources.list \
&& echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial universe" >> /etc/apt/sources.list \
&& echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-updates universe" >> /etc/apt/sources.list \
&& echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial multiverse" >> /etc/apt/sources.list \
&& echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-updates multiverse" >> /etc/apt/sources.list \
&& echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-backports main restricted universe multiverse" >> /etc/apt/sources.list \
&& echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-security main restricted" >> /etc/apt/sources.list \
&& echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-security universe" >> /etc/apt/sources.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------start Install pytorch-----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
   && rm -rf /var/lib/apt/lists/*

#https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh


RUN curl -o ~/conda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/conda.sh && \
     ~/conda.sh -b -p /opt/conda && \
     rm ~/conda.sh

ENV PATH /opt/conda/bin:$PATH


RUN  /opt/conda/bin/conda install numpy pandas pyyaml scipy ipython mkl mkl-include && \
     /opt/conda/bin/conda install -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda clean -ya

WORKDIR /opt/pytorch

RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch \
  && conda clean -ya

# -----------------------------End Install pytorch-----------------------

# -----------------------------Start Install R-----------------------
# From https://github.com/rocker-org/rocker/blob/dd21f0b706/r-apt/xenial/Dockerfile
## Set a default user. Available via runtime flag `--user docker`
## Add user to 'staff' group, granting them write privileges to /usr/local/lib/R/site.library
## User should also have & own a home directory (for rstudio or linked volumes to work properly).
RUN sudo useradd docker \
	&& sudo mkdir /home/docker \
	&& sudo chown docker:docker /home/docker \
	&& sudo addgroup docker staff

RUN sudo apt-get update \
	&& sudo apt-get install -y --no-install-recommends \
		software-properties-common \
                ed \
		less \
		locales \
		vim-tiny \
		wget \
		ca-certificates \
        && printf "deb https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/linux/ubuntu xenial-cran35/" | sudo tee -a /etc/apt/sources.list \
         && sudo add-apt-repository --enable-source --yes "ppa:marutter/rrutter" \
	&& sudo add-apt-repository --enable-source --yes "ppa:marutter/c2d4u"

## Configure default locale, see https://github.com/rocker-org/rocker/issues/19

RUN printf "en_US.UTF-8 UTF-8" | sudo tee --append /etc/locale.gen \
	&& sudo locale-gen en_US.utf8 \
	&& sudo /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

## Now install R and littler, and create a link for littler in /usr/local/bin
## Default CRAN repo is now set by R itself, and littler knows about it too
## r-cran-docopt is not currently in c2d4u so we install from source

#fix the tsinghua mirror
RUN sudo apt-get install -y apt-transport-https \
  &&sudo apt-key adv --recv-keys --keyserver keyserver.ubuntu.com 51716619E084DAB9

RUN sudo apt-get update \
        && sudo apt-get install -y --no-install-recommends \
 		 r-base \
 		 r-base-dev

RUN printf "export R_HOME=/usr/lib/R" | sudo tee -a /etc/profile

# -----------------------------End Install R-----------------------

# -----------------------------Start Install Package-----------------------
RUN sudo apt-get update && sudo apt-get -y --no-install-recommends install \
  libxml2-dev \
  libcairo2-dev \
  libsqlite3-dev \
  libmariadbd-dev \
  libmariadb-client-lgpl-dev \
  libpq-dev \
  libssh2-1-dev \
  unixodbc-dev \
  libcurl4-openssl-dev



RUN R -e "install.packages(c('devtools','formatR','remotes','selectr','caTools','BiocManager'),repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')"


RUN R -e "install.packages(c('dplyr','data.table','xgboost','reticulate'), repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN/')"



# install pcalg



RUN R -e "install.packages('BiocManager', repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')" \
 && R -e "options('BioC_mirror'='http://mirrors.ustc.edu.cn/bioc/');BiocManager::install(c('graph','RBGL','Rgraphviz'))"



RUN sudo apt-get install -y libv8-3.14-dev \
 && mv /opt/conda/lib/libgfortran.so.4.0.0 /opt/conda/lib/libgfortran.so.4.0.0.bak \
 && mv /opt/conda/lib/libgfortran.so.4 /opt/conda/lib/libgfortran.so.4.bak \
 && R -e "install.packages(c('pcalg'), repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')" \
 && mv /opt/conda/lib/libgfortran.so.4.0.0.bak /opt/conda/lib/libgfortran.so.4.0.0 \
 && mv /opt/conda/lib/libgfortran.so.4.bak /opt/conda/lib/libgfortran.so.4


#RUN sudo add-apt-repository ppa:jonathonf/gcc-7.1 \
# && sudo apt-get update \
# && sudo apt-get install -y gcc-7 g++-7 --no-install-recommends \
# && R -e "install.packages(c('pcalg'), repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')"

# install kpcalg
RUN R -e "install.packages(c('kpcalg'), repos = 'https://mirrors.tuna.tsinghua.edu.cn/CRAN')"


# install CANM
RUN R -e "devtools::install_github('DMIRLAB-Group/CANM')"



# -----------------------------End Install Package-----------------------

CMD ["/bin/bash"]
