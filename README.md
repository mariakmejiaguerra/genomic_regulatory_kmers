#### Code in this repository aims to be reproducible, easy to deploy across environments and machines based on the use of [Docker](https://docs.docker.com/engine/docker-overview/) containers.

<a name="top"></a>

- [About containers](#containers)
- [About this project](#about)
- [General view of the repository](#repoview)
- [Launch Docker In a CBSU machine](#launchcbsu)
  * [Example 1 - run bash](#ie1)
  * [Example 2 - run jupyter lab](#ie2)
  * [Example 3 - link volumes](#ie3)
- [Not in CBSU?](#notcbsu)
- [Goals](#goals)
- [General repository guidelines](#guidelines)
  * [Folders](#foldersguide)
  * [Files](#filesguide)


<a name="containers"></a>

#### About containers

* A container includes necessary dependencies to run. 
* Containers can run on any operating system including Windows and Mac via the [Docker engine](https://docs.docker.com/engine/).
* Containers can also be deployed in [CBSU machines](https://biohpc.cornell.edu/lab/userguide.aspx?a=software&i=340#c).  
* Containers can also be deployed in the cloud using [Amazon Elastic Container Service](https://aws.amazon.com/ecs/)
or [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine/).

<a name="about"></a>

#### About this project
This repository includes:
* Folder for the src code.   
* Folders for notebooks (*i.e.,* jupyter and R Notebooks) to develop and report on exploratory data analysis.     
* Folder for data, which ideally will be hosted in a data server, and is never synchronized with git.        
* A container to reproduce development environments using [docker](https://docs.docker.com/)   

<a name="repoview"></a>

##### General view of the repository:

    |-- .gitignore
    |-- LICENSE.md
    |-- README.md
    |-- docker-compose.yml
    |-- code/
        |-- kmergrammar/
        |-- README.md
        |-- MANISFEST.in
        |-- INSTALL.rst
        |-- setup.py
        |-- env_kmergrammar.yml
    |-- data/
        |--.gitignore
    |-- docker/
        |--Dockerfile
    |-- notebooks/
        |--report/
        |--analysis/

<a name="launchcbsu"></a>

##### When working at Cornell I launch Docker In a CBSU machine 
1. First you need a reservation in a `cbsuxxx` machine  
2. Login and navigate to workdir with `cd /workdir` and make a folder with the netid `mkdir mynetid`  
3. Navigate to the folder `cd mynetid` and clone the repository `git clone`  
4. Navigate to the docker folder in the repository and build the image  
`docker1 build -t testdocker /workdir/mynetid/myrepo/docker/`

<a name="ie1"></a>

##### Example 1 

* Run Bash shell on the container
	`docker1 run -it --rm biohpc_mynetid/testdocker bash`

<a name="ie2"></a>

##### Example 2
* Start the container and turn on the jupyter lab server
`docker1 run -it --rm -p 8888:8888 biohpc_mynetid/testdocker jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='local_dev'`  
* Navigate to your browser `http://cbsuxxx.biohpc.cornell.edu:8888/` and use the token `local_dev` to use the server
* Turn the container off.

<a name="ie3"></a>

##### Example 3
* Start the container and link the volumes that you want available
```
  docker1 run -it --rm \  
  -v '/workdir/mynetid/myrepo/notebooks:/workdir/notebooks:Z' \
  -v '/workdir/mynetid/myrepo/code:/workdir/code' \
  -v '/workdir/mynetid/myrepo/data:/workdir/data' \
  -p 8888:8888 biohpc_mynetid/testdocker jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='local_dev'  
```
* Navigate to your browser `http://cbsuxxx.biohpc.cornell.edu:8888/` and use the token `local_dev` to use the server
* Turn the container off.

__*\*Note\**__ [*Get familiar with docker at cbsu*](https://biohpc.cornell.edu/lab/userguide.aspx?a=software&i=340#c)  
CBSU machines don't use docker, but an alias docker1 that wraps many docker functionalities.
You can mount volumes and directories to Docker on BioHPC using -v option. For security reasons you cannot mount any directories, but only those owned by and located under `/workdir`, `/local/workdir` and `/local/storage`. Directory `/workdir/mynetid` is mounted automatically in any docker1 container as `/workdir`.  

<a name="notcbsu"></a>

##### Not in CBSU? I use Docker in a different machine

1. To use the docker images you first need to install Docker:  
For Mac: https://docs.docker.com/docker-for-mac/  
For Windows: https://docs.docker.com/docker-for-windows/  
For Linux: https://docs.docker.com/engine/installation/  

2. Create a `.env_dev` file with development environment variables  
  We want to keep passwords secret, so this file won't be in the repository and the .gitignore has been instructed to ignore it
```
# random seed for reproducible models
random_seed=42

#database password
db_password=1234
```
3. Run `docker-compose build --no-cache`. This will build the development image with all the packages I defined installed within it.
4. Run `docker-compose up` and navigate to your browser to find jupyter server running on [http://localhost:8888](http://localhost:8888). 
5. Access it by entering in the token `local_dev`.
6. Once you are done, remember to shutdown jupyter server and docker   
  Go to the terminal and Press Ctrl+C   
  `docker-compose down`
  
__*\*Note\**__ [*Get familiar with docker*](https://docs.docker.com)  
You may need to be `sudo` to run `docker` commands. Try them without `sudo` first.


<a name="goals"></a>

### Goals 

A few of key goals working with Docker containers are.
 - To keep the image size at a manageable size.
 - Make the images easy to extend, and easy to maintain
 - Adopt a "best practices" outline to facilitate the use of the containers.

<a name="guidelines"></a>

## General repository guidelines

<a name="foldersguide"></a>

##### Folders should have a README
Useful to describe files that are not self-explanatory

<a name="filesguide"></a>

##### Files - some conventions
1.  Use a filename that describe the file content
2.  Use dashes to separate words in filenames
3.  Do not use underscores to separate words in filenames
4.  Do not use camelCase for filenames
5.  Use markdown for text-based documentation to facilitate reading while browsing the repo.