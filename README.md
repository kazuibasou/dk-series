# dK-series
dK-series [1] is a family of randomization methods for networks.
The dK-series produces randomized networks that preserve up to the individual node’s degree, node’s degree correlation, and node’s clustering coefficient of the given unweighted network, depending on the parameter value *d* = 0, 1, 2, or 2.5.

We provide code for the dK-series in C++.

## Requirements
Require gcc version 4.2.1 or later.

We have confirmed that our code works in the following environment.

- macOS 12.4

## Build
(i) Clone this repository:

	git clone git@github.com:kazuibasou/dk-series.git

(ii) Go to `dk-series/src/`:

	cd dk-series/src

(iii) Run the make command:

	make

This generates the following structure of the directory.

	dk-series/
	├ bin/
	├ data/
	├ src/
	└ rand_network/

If you find a file `dk_series` in the folder `dk-series/bin/`, the build has been successfully completed.

## Usage

### Input file

We need to feed a file, *network*.txt, where *network* indicates the name of the network and is arbitrary. 
The file should be placed in `dk-series/data/`.
In the file, each line contains two integers separated by a half-width space, which represents an edge between two nodes.
This follows the standard network data format.

Note that our code successfully works as long as each node's index is an integer. 
There is no need for the node index to start at 0 or to increment by 1.

#### Example
Let's consider a network that consists of a set of nodes *V* = {0, 1, 2, 3, 4} and a set of edges *E* = {{0, 1}, {0, 4}, {1, 2}, {2, 3}, {3, 4}}. 
Then, the input file should be as follows:

``` text:
0 1
0 4
1 2
2 3
3 4
```

### Generating randomized networks

Go to `dk-series/bin/` and run the following command:

	./dk_series <network> <d> <num_gen>

The three arguments are as follows.

#### `<network>`
The name of the network.

#### `<d>`
The value of *d*, which should be 0, 1, 2, or 2.5.

#### `<num_gen>`
The number of networks to be generated.

#### Example
To generate three randomized networks with *d* = 2.5 for the network named `example-network`, go to `dk-series/bin/` and run the following command:

	./dk_series example-network 2.5 3

### Output files
The *n* th (*n*=1, ..., *num_gen*) randomized network, i.e., *network*\_*d*\_*n*.txt will be created in the folder `dk-series/rand_network/`.

### Notes
- In general, when *d* = 0, 1, or 2, the code runs fast. When *d* = 2.5, it takes longer. 

- Multiple edges and loops are allowed in randomized networks.

## Reference

[1] Orsini, C., Dankulov, M., Colomer-de-Simón, P. et al. Quantifying randomness in real networks. Nat. Commun. 6, 8627 (2015). [<a href="https://doi.org/10.1038/ncomms9627">paper</a>]

## License

This source code is released under the MIT License, see LICENSE.txt.

## Contact
- Kazuki Nakajima (https://kazuibasou.github.io/index_en.html)
- kazuibasou[at]gmail.com

(Last update: 2022/10/14)