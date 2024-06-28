# oculus.jl
one-class image classification implementation for julia

## getting started

- this code is divided in two modules that can be used independently and one that orchestrates all the work:

  ### datafolder.jl
  
  - it loads a dataset in the form of
    
    > folder (dataset)
    > 
    > > subfolder (class 0)
    > >
    > > subfolder (class 1)

  ### monovision.jl

  - it stores the architectures of the models available
 
  ### oculus.jl

  - it enables the training and inference of models

## activating the project

once you downloaded julia and have it opened on a terminal:

1. access to pkg typing "]", it will appear something like this:

`(oculus.jl) pkg>`

2. then you can activate the project with all the dependencies using `activate .`

et voilà
