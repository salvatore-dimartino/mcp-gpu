# Maximum Clique on GPUs

This repository stores the software presented in '_Efficiently Maximum Clique Computation on Many Core Graphic Processing Unit_', ICSOFT 2024 Conference.

## Minimum Requirements

In Order to reproduce at least our experiment it is recommenaded a GPU with at least 8GBs VRAM
and compute capability of 8.6 (RTX 3000 series), NVIDIA Developer Toolkit 12.2.

## Build

Easy build can be done by launching the following command:

```
make all
```
The executable `parallel_mcp_on_gpus` will be created in the main directory.
In case you use different hardware changes the SM Architecture that has been set by default at `sm_86` and `compute_86`.

## Usage

Options will be available launching the command

```
./parallel_mcp_on_gpus -h
```
Pay attention you cannot launch the raw asii adjacency list input graph format it, first of all must be converted in a binary format. for instance if the input graph is a market format:

```
./parallel_mcp_on_gpus -g <input_graph.mtx> -r <output_graph.bel> -m convert
```

This command gives you the output graph in binary format (.bel)

### Tasks

Is it possible to specify a couple of task: `mcp` and `mcp-eval` both solves the maximum clique problem, the only diffenernce is that mcp-eval evaluates and check the output clique.
As already seen the task `convert` converts the input ascii format in the binary one. `mcp-eval` works just with the default coloring strategy

### Coloring Algorithm

Is it possible to specify more than one pruning strategy (`c` option):

- `psanse`: uses the default coloring proposed by San Segundo in BBMC[SP]/BBMCI Algorithm
- `recolor`: uses the ReColor prunyng strategy from San Segund's BBMCR
- `number`: uses Tomita's NUMBER Algorithm to perform Coloring
- `renumber`: uses Tomita's Re-NUMBER Algorithm to improve Coloring
- `reduce`: (Experimental BONUS) uses Lemmas from the reduce procedure from MC-BRB algorithm + standard default San Segundo's Coloring

### Warp-wise parallelism

The coloring strategies:
- `psanse`
- `reduce`

Can be launched with `x` options. In this case the program subdivides tasks among warps from the second level subtree.