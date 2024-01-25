# Babytorch:  autograd engine

Table of contents:

-   [About](#about)
-   [Justfile](#justfile)
-   [Local Developmnent](#local-development)
    -   [Pre-requisites](#pre-requisites)
    -   [Initialization](#initialization)
    -   [Usage](#usage)
    -   [Testing](#testing)
-   [Todos](#todo-list)

## About

Currently fully supports auto-differentiation for scalar values.  

An engine was built around scalars at first to prototype the architecture that would enable the project to be easily extensible, loosely coupled and highly modular. For implementation details you may look [here](../src/babytorch/scalar.hpp). 


Now adding further support for Tensors. In progress ... 

![28](https://geps.dev/progress/28)


## Justfile
The project uses [Just](https://github.com/casey/just) for command automation and ease of access, a better alternative to Makefile. 

All shortucts are located in `justfile` in root directory. To view the commands you can also do:
```shell
just
```
Info about installation can be found [here](https://github.com/casey/just#packages).


## Local Development

Make sure to have `just` installed on you system as all of the utility scripts and commands are stored there.

### Pre-requisites

To check whether all required dependencies are present on host machine you can run: 
```bash
just check
```
*NOTE: At this point this project is only targeted at Linux OS. So, be wary of that.*

### Initialization

The project relies on a combination of tools to facilitate the development process.
The main components are tightly integrated `cmake` build system with a `vcpkg` as external dependency manager. 

Everything is configured with these tools. As a quick reminder:
- CmakeLists.txt for cmake build process 
- vcpkg.json for the list of external dependencies
- CmakePreset.json for env. variables requried for the build system.

Dependency manager is dependencecy itself. To make the project more self-contained, `vcpkg` comes as a git sub-module and all of the calls to it are configured for that path, via *./extern*, so there is no need to have it installed on host machine. Packages which are **not** suported by vcpkg are exptected to be integrated as a submodule.

Adding a new dependency is trivial. A matter of editing **vcpkg.json** and adding requried commands to **Cmakelists.txt**. As an example, the project initially have integraded `fmt` and `spdlog` to showcase and test the integration process. Now, the project also contains `Catch2` library for tests and `Boost` for an upcoming python bindings.

To build the project build folder, configure with default presets, set the project root path and project build path you can run:
```bash
just init
```

### Usage

There is a sample **main.cpp** file in *./src* that showcases a simplest usage of an engine. There isn't much to it as of now. It proves to serve rather as a 'checkpoint' to test everything out and play around if there is a need to do so, as well as a showcase of current implementation state. 

To build and run use the following command:

```bash
just run
```

This command will output an example for both Scalar and Tensors

#### Scalar Example

To showcase the Scalar automatic differentiation, there are 5 variables that represent all types of operations, i.e. ...
```c++
auto x = Scalar::create(1.0);
auto y = Scalar::create(2.0);
auto z = Scalar::create(3.0);
auto k = Scalar::create(4.0);
auto j = Scalar::create(5.0);

auto result = (x * y + z - k / j);
auto result2 = (x * y - 12 + z / 1.2 - k * 0.2 / j); // This also works
```

Printing to std::out all of the values we can see how each Scalar has been updated by `.backward()` called on a `result` variable. This would render something akin to this:
```
x = Scalar(data=1, grad=2)
y = Scalar(data=2, grad=1)
z = Scalar(data=3, grad=1)
k = Scalar(data=4, grad=-0.1999999996)
j = Scalar(data=5, grad=0.159999999936)
```

Scalar values also support type interoperability with all arithmetic types. So any arithmetic value can be passed in anywhere in an expression, as was shown in `result2`. Running the same print expressions would result in these values:
``` 
x = Scalar(data=1, grad=2)
y = Scalar(data=2, grad=1)
z = Scalar(data=3, grad=0.833333326388889)
k = Scalar(data=4, grad=-0.03999999992)
j = Scalar(data=5, grad=0.031999999987200003)
```

Running 
Each variable has data that holds the value of a scalar `.data` and `.grad`, its gradient.


#### Tensor Example

At this point Tensors are still work in progress, but it already has full support for initialization and indexing. Tensor data is resided in `TensorData`, which is a wrapper around 1-D vector of values of a Tensor. This design is crucial for efficient tensor manipulations, without chaging the underlying data structure. This is achieved by the usage of strides and indexing to required location in 1-D storage given arbitrary shape.

Tensors can be created by either providing a vector or any number of integer arguments that represent the dimensionality. To create a tensor with dimensions 3x3, we simply do `Tensor(3,3);`. Consider this example:

```c++
// creating a tensor with shape (3,3,5)
auto tensor = Tensor(3,3,5);

// sub-tensor accesss returns another tensor of according shape and size
auto tensor1 = tensor[1];
auto tensor2 = tensor[1,2];
auto tensor3 = tensor[1,2,3];
```

Having these tensors printed out we would get result similar to this ->
```
// i.e. Tensor
Tensor([[[-0.905293,  0.527793, -0.975776, -0.716542, -0.176362]
         [ 0.322527,  0.140777,  0.469231, -0.773779, -0.559799]
         [ 0.392686,  0.509423,  0.008176,  0.201130, -0.811616]]
       
        [[ 0.877142, -0.306769, -0.436334,  0.496036,  0.376576]
         [-0.570975, -0.126152, -0.369967,  0.484594, -0.785906]
         [ 0.622890, -0.326047,  0.134703, -0.741307, -0.883508]]
       
        [[ 0.605673, -0.217171, -0.549314, -0.053289,  0.898330]
         [ 0.924623, -0.460411,  0.225380,  0.413370,  0.575194]
         [-0.797811,  0.016582, -0.311041, -0.922726,  0.568440]]])

// i.e. Tensor[1]
Tensor([[ 0.877142, -0.306769, -0.436334,  0.496036,  0.376576]
        [-0.570975, -0.126152, -0.369967,  0.484594, -0.785906]
        [ 0.622890, -0.326047,  0.134703, -0.741307, -0.883508]])

// i.e. Tensor[1,2]
Tensor([ 0.622890 -0.326047  0.134703 -0.741307 -0.883508])

// i.e. Tensor[1,2,3]
Tensor([-0.741307])

```
### Testing

The choice for a testing framework have lied upon `Catch2` for its simplicity and ease of use. 

At this point tests are primarily covering Scalars, starting with base operators and ending with Scalar overloaded operators, to make sure that basic building blocks are working properly. 

To test the project, simply run:
```bash
just test
```

### Todo list:
> - [x] Setup project build system and structure
> - [x] Add basic operators
> - [x] Add operators tests
> - [x] Add Scalar autograd
>   - [x] Add Scalar
>   - [x] Add ScalarFunction
>   - [x] Add forward functions
>   - [x] Add backward functions
>   - [x] Add Scalar operator overloads
>   - [x] Add Scalar tests
>   - [x] Add topological sort for backpropagation
>   - [x] Add backpropagation
- [ ] Add Tensor autograd 
    - [x] Add Tensor
    - [x] Add TensorStorage
    - [x] Add indexing, shape, strides
    - [x] Add Tensor string views
    - [ ] Add TensorOperations
    - [ ] Add TensorFunctions
    - [ ] Add forward functions
    - [ ] Add backward functions
    - [ ] Add operator overloads
    - [ ] Add backpropagation
- [ ] CPU parallelization
- [ ] GPU parallelization
- [ ] Python bindings
- [ ] SOTA architectures and algorithms
