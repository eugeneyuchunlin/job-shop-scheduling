Job Shop Problem
===
![quality](https://www.code-inspector.com/project/23598/score/svg)
![score](https://www.code-inspector.com/project/23598/status/svg)


## Problem description

In the semiconductor manufacture process, jobs are assigned to resource in particular time. The lot information is shown below.

![](https://i.imgur.com/wMhZh53.png)


## Algorithm


The genetic algorithm is employed to solve the problem. In the program, `list_ele_t` is a structure type used to perfrom doubly linked list. `job_t` is a structure type used to store the information related to a lot. `job_base_t` is also a structure type embeded in `job_t`. `job_base_t` is used to store key information of a job. `job_t` is extension of `job_base_t`. `list_ele_t` is also embeded in `job_t`.

```c=
typedef struct job_t{
    job_base_t base;
    list_ele_t list;
    ...
}job_t;
```

The information of machine is store in `machine_t` in the program. The `machine_base_t` is used to store the key information of machine embeded in `machine_t`. `machine_t` is extension of `machine_base_t`.

```c=
typedef struct machine_t{
    machine_base_t base;
    ...
}machine_t;
```

In the genetic algorithm, the chromosome is represented as a solution. The main idea of algorithm used in the program is decoding the chromosomes in parallel. The data parallelism is significant in parallel computing without using a mutex. In the program, `jobs` is a variable whose type is a pointer to a pointer to `job_t` . The variable jobs points to a device memory where the elements point to another device memory where the element is an instance of job_t. The data structure is shown below. In the CUDA kernel function, the $x$ dimension is subject to the number of chromosomes and $y$ dimension is subject to the number of jobs. In the CUDA kernel, $x$ is equal to `threadIdx.x + blockIdx.x * blockDim.x` and $y$ is equal to `threadIdx.y + blockIdx.y * blockDim.y`. Each job in the memory occupies a thread to do machine selection. The machine of job is determined by gene in chromosome.

![](https://i.imgur.com/EjD0ED4.png)

After machine selection, each machine has a bunch of jobs. The relation between jobs and machine is maintained by doubly linked list. The sorting algorithm used to sort the jobs by gene's value is merge sort. In the CUDA kernel, $x$ dimension is subject to the number of chromosomes and $y$ dimension is subject to the number of machines. Each machine occupies a thread to sort the jobs in parallel. The data structure and dimension of CUDA kernel is shown below.

![](https://i.imgur.com/vLkH1O7.png)

The genetic algoritm runs with multi-population. The POSIX threads are in charge of an evolution of a population. The topology of migration between populations is directed and circular. The migration topology is shown below. Each circle is represented as a population.


![](https://i.imgur.com/d7FU6S1.png)



Experiment
---
![](https://i.imgur.com/UBfQeVE.png)


## Enviornment

* CPU : intel core i5-9400
* GPU : NVIDIA 2060 Super 8GB
* MB : H370-f
* Memory : 32 GB
* OS : Ubuntu 20.04
* CUDA version : 11.1
* Programming language : C++ 11

## Build

```shell=
mkdir build
cd build
cmake -D CUDA_TOOLKIT_ROOT_DIR=<> .. 
make
```

## Execution
```shell=
./main ../files/
```
