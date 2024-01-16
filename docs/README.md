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

Currently is a fully developed auto-differentiation engine built around scalar values. 

An engine was built around scalars at first to prototype the architecture that would enable the project to be easily extensible, loosely coupled and highly modular. For implementation details you may look [here](../src/babytorch/scalar.hpp). 

Now adding Tensors and Tensor Storage. In progress ... 

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

### Testing

The choice for a testing framework have lied upon `Catch2` for its simplicity and ease of use. 

To test the project, simply run:
```bash
just test
```

### Todo list:
> - [x] Setup project build system and structure
> - [x] Add Scalar auto-differention
>   - [x] Add basic operators
>   - [x] Add operators tests
>   - [x] Add basic functions
>   - [x] Add Scalar
>   - [x] Add forward functions
>   - [x] Add backward functions
>   - [x] Add Scalar operator overloads
>   - [x] Add Scalar tests
>   - [x] Add topological sort for backpropagation
>   - [x] Add backpropagation
- [ ] Tensor and Tensor storage
- [ ] CPU parallelization
- [ ] GPU parallelization
- [ ] Python bindings
- [ ] SOTA architectures and algorithms
