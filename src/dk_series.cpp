#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <chrono>
#include "network.h"
#include "rewiring.h"

int write_network(const char *network, const std::string d, const int k, Network G, Network randG){

	const char *dir = "../rand_network/";

	FILE *f;
	std::string fpath = std::string(dir) + network + "_" + d + "_" + std::to_string(k) + ".txt";
	f = fopen(fpath.c_str(), "w");
	if(f == NULL) {
		printf("Error: Could not open file %s.\n", fpath.c_str());
		exit(0);
	}

	int num_loop, i;
	for(int v=0; v<randG.N; ++v){
		num_loop = std::count(randG.nlist[v].begin(), randG.nlist[v].end(), v);
		i = 0;
		for(int w:randG.nlist[v]){
			if(w >= v){
				fprintf(f, "%d %d\n", G.index_to_node[v], G.index_to_node[w]);
			}
			else if(w == v && i < int(num_loop)/2){
				fprintf(f, "%d %d\n", G.index_to_node[v], G.index_to_node[w]);
				i += 1;
			}
		}
	}

	fclose(f);

	printf("Wrote the %d-th network randomized with d = %s.\n", k, d.c_str());

	return 0;
}

int randomizing_with_d_zero(Network G, Network &randG){

	int M = 0;
	for(int v=0; v<G.N; ++v){
		M += int(G.nlist[v].size());
	}
	M = int(M)/2;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<int> randN(0, G.N-1);

	randG.N = G.N;
	randG.nlist = std::vector<std::vector<int>>(randG.N, std::vector<int>());

	int u, v;
	for(int i=0; i<M; ++i){
		u = randN(mt);
		v = randN(mt);
		randG.nlist[u].push_back(v);
		randG.nlist[v].push_back(u);
	}

	randG.M = 0;
	randG.max_k = 0;
	for(int v=0; v<randG.N; ++v){
		randG.M += int(randG.nlist[v].size());
		if(int(randG.nlist[v].size()) > randG.max_k){
			randG.max_k = int(randG.nlist[v].size());
		}
		std::sort(randG.nlist[v].begin(),randG.nlist[v].end());
	}
	randG.M = int(randG.M)/2;

	/*
	// test
	if(G.N != randG.N || G.M != randG.M){
		printf("%d, %d, %d, %d\n", G.N, randG.N, G.M, randG.M);
		printf("Failed randomization with d = 0.\n");
		exit(0);
	}
	*/

	printf("Successfully generated a randomized network with d = 0.\n");

	return 0;
}

int randomizing_with_d_one(Network G, Network &randG){

	std::vector<int> stub_list;
	for(int v=0; v<G.N; ++v){
		int k = int(G.nlist[v].size());
		for(int i=0; i<k; ++i){
			stub_list.push_back(v);
		}
	}

	std::random_device rd;
	std::mt19937 mt(rd());
	std::shuffle(stub_list.begin(), stub_list.end(), mt);

	randG.N = G.N;
	randG.nlist = std::vector<std::vector<int>>(randG.N, std::vector<int>());

	int u, v;
	while(int(stub_list.size()) > 0){
		u = stub_list.back();
		stub_list.pop_back();		
		v = stub_list.back();
		stub_list.pop_back();
		randG.nlist[u].push_back(v);
		randG.nlist[v].push_back(u);
	}

	randG.M = 0;
	randG.max_k = 0;
	for(int v=0; v<randG.N; ++v){
		randG.M += int(randG.nlist[v].size());
		if(int(randG.nlist[v].size()) > randG.max_k){
			randG.max_k = int(randG.nlist[v].size());
		}
		std::sort(randG.nlist[v].begin(),randG.nlist[v].end());
	}
	randG.M = int(randG.M)/2;

	/*
	// test
	if(G.N != randG.N || G.M != randG.M){
		printf("Failed randomization with d = 1.\n");
		exit(0);
	}

	for(int v=0; v<randG.N; ++v){
		if(G.nlist[v].size() != randG.nlist[v].size()){
			printf("Failed randomization with d = 1.\n");
			exit(0);
		}
	}
	*/

	printf("Successfully generated a randomized network with d = 1.\n");

	return 0;
}

int randomizing_with_d_two(Network G, Network &randG){

	int k, u, v, i;

	std::unordered_map<int, std::vector<int>> stub_list;
	for(int v=0; v<G.N; ++v){
		k = int(G.nlist[v].size());
		if(stub_list.find(k) == stub_list.end()){
			stub_list[k] = std::vector<int>();
		}
		for(i=0; i<k; ++i){
			stub_list[k].push_back(v);
		}	
	}

	std::random_device seed_gen;
  	std::mt19937 engine(seed_gen());
	for(auto itr=stub_list.begin(); itr!=stub_list.end(); ++itr){
		k = itr->first;
		std::shuffle(stub_list[k].begin(), stub_list[k].end(), engine);
	}

	JDM num_jnt_deg;
	G.calc_num_jnt_deg(num_jnt_deg);

	randG.N = G.N;
	randG.nlist = std::vector<std::vector<int>>(randG.N, std::vector<int>());

 	std::vector<int> ks;
  	num_jnt_deg.get_keys(ks);
  	std::shuffle(ks.begin(), ks.end(), engine);
  	for(int k:ks){
  		std::vector<int> ls;
  		num_jnt_deg.get_keys(ls);
  		std::shuffle(ls.begin(), ls.end(), engine);
  		for(int l:ls){
  			while(num_jnt_deg.get(k, l) > 0){
				u = stub_list[k].back();
		  		stub_list[k].pop_back();
		  		v = stub_list[l].back();
		  		stub_list[l].pop_back();
		  		randG.add_edge(u,v);
		  		num_jnt_deg.subtract(k, l, 1);
		  		num_jnt_deg.subtract(l, k, 1);
			}
  		}
  	}

	randG.M = 0;
	randG.max_k = 0;
	for(int v=0; v<randG.N; ++v){
		randG.M += int(randG.nlist[v].size());
		if(int(randG.nlist[v].size()) > randG.max_k){
			randG.max_k = int(randG.nlist[v].size());
		}
		std::sort(randG.nlist[v].begin(),randG.nlist[v].end());
	}
	randG.M = int(randG.M)/2;

	/*
	// test
	if(G.N != randG.N || G.M != randG.M){
		printf("a\n");
		printf("Failed randomization with d = 2.\n");
		exit(0);
	}

	for(int v=0; v<randG.N; ++v){
		if(G.nlist[v].size() != randG.nlist[v].size()){
			printf("b\n");
			printf("Failed randomization with d = 2.\n");
			exit(0);
		}
	}

	G.calc_num_jnt_deg(num_jnt_deg);

	JDM rand_num_jnt_deg;
	randG.calc_num_jnt_deg(rand_num_jnt_deg);

	ks.clear();
  	num_jnt_deg.get_keys(ks);
  	for(int k:ks){
  		std::vector<int> ls;
  		num_jnt_deg.get_keys(ls);
  		for(int l:ls){
  			if(num_jnt_deg.get(k, l) != rand_num_jnt_deg.get(k, l)){
  				printf("%d, %d, %d, %d\n", k, l, num_jnt_deg.get(k, l), rand_num_jnt_deg.get(k, l));
				printf("Failed randomization with d = 2.\n");
				exit(0);
  			}
  			if(num_jnt_deg.get(l, k) != rand_num_jnt_deg.get(l, k)){
  				printf("%d, %d, %d, %d\n", k, l, num_jnt_deg.get(l, k), rand_num_jnt_deg.get(l, k));
				printf("Failed randomization with d = 2.\n");
				exit(0);
  			}
  		}
  	}

	ks.clear();
  	rand_num_jnt_deg.get_keys(ks);
  	for(int k:ks){
  		std::vector<int> ls;
  		rand_num_jnt_deg.get_keys(ls);
  		for(int l:ls){
  			if(num_jnt_deg.get(k, l) != rand_num_jnt_deg.get(k, l)){
  				printf("%d, %d, %d, %d\n", k, l, num_jnt_deg.get(k, l), rand_num_jnt_deg.get(k, l));
				printf("Failed randomization with d = 2.\n");
				exit(0);
  			}
  			if(num_jnt_deg.get(l, k) != rand_num_jnt_deg.get(l, k)){
  				printf("%d, %d, %d, %d\n", k, l, num_jnt_deg.get(l, k), rand_num_jnt_deg.get(l, k));
				printf("Failed randomization with d = 2.\n");
				exit(0);
  			}
  		}
  	}
  	*/

	printf("Successfully generated a randomized network with d = 2.\n");

	return 0;
}

int randomizing_with_d_two_five(Network G, Network &randG){

	randomizing_with_d_two(G, randG);

	targeting_rewiring_d_two_five(G, randG);

	/*
	// test
	if(G.N != randG.N || G.M != randG.M){
		printf("Failed randomization with d = 2.5.\n");
		exit(0);
	}

	for(int v=0; v<randG.N; ++v){
		if(G.nlist[v].size() != randG.nlist[v].size()){
			printf("Failed randomization with d = 2.5.\n");
			exit(0);
		}
	}

	JDM num_jnt_deg;
	G.calc_num_jnt_deg(num_jnt_deg);
	JDM rand_num_jnt_deg;
	randG.calc_num_jnt_deg(rand_num_jnt_deg);

	std::vector<int> ks, ls;

	ks.clear();
  	num_jnt_deg.get_keys(ks);
  	for(int k:ks){
  		ls.clear();
  		num_jnt_deg.get_keys(ls);
  		for(int l:ls){
  			if(num_jnt_deg.get(k, l) != rand_num_jnt_deg.get(k, l)){
				printf("Failed randomization with d = 2.5.\n");
				exit(0);
  			}
  			if(num_jnt_deg.get(l, k) != rand_num_jnt_deg.get(l, k)){
				printf("Failed randomization with d = 2.5.\n");
				exit(0);
  			}
  		}
  	}

	ks.clear();
  	rand_num_jnt_deg.get_keys(ks);
  	for(int k:ks){
  		ls.clear();
  		rand_num_jnt_deg.get_keys(ls);
  		for(int l:ls){
  			if(num_jnt_deg.get(k, l) != rand_num_jnt_deg.get(k, l)){
				printf("Failed randomization with d = 2.5.\n");
				exit(0);
  			}
  			if(num_jnt_deg.get(l, k) != rand_num_jnt_deg.get(l, k)){
				printf("Failed randomization with d = 2.5.\n");
				exit(0);
  			}
  		}
  	}

  	std::vector<double> ddcc;
  	G.calc_degree_dependent_clustering_coefficient(ddcc);
  	std::vector<double> rand_ddcc;
  	randG.calc_degree_dependent_clustering_coefficient(rand_ddcc);
  	if(ddcc.size() != rand_ddcc.size()){
		printf("Failed randomization with d = 2.5.\n");
		exit(0);
  	}

  	double dist = 0;
  	double norm = 0;
  	for(int k=0; k<int(ddcc.size()); ++k){
  		norm += ddcc[k];
  		dist += std::fabs(ddcc[k] - rand_ddcc[k]);
  	}

  	printf("Final L1 distance between target and present c(k): %lf\n", double(dist)/norm);
  	*/
	
	printf("Successfully generated a randomized network with d = 2.5.\n");

	return 0;
}

int main(int argc,char *argv[]){
	if(argc != 4){
		printf("Please input following:\n");
		printf("./dk_series (name of network) (value of d) (number of generation)\n");
		exit(0);
	}

	const char *network = argv[1];
	const std::string d = argv[2];
	const int num_gen = std::stoi(argv[3]);

	Network G;
	G.read_network(network);

	if(d == "0"){
		for(int k=1; k<=num_gen; ++k){
			printf("-------------------------------------------------\n");
			printf("Started %d-th generation of a randomized network with d = %s.\n", k, d.c_str());
			Network randG;
			randomizing_with_d_zero(G, randG);
			write_network(network, d, k, G, randG);
			printf("-------------------------------------------------\n");
		}
	}
	else if(d == "1"){
		for(int k=1; k<=num_gen; ++k){
			printf("-------------------------------------------------\n");
			printf("Started %d-th generation of a randomized network with d = %s.\n", k, d.c_str());
			Network randG;
			randomizing_with_d_one(G, randG);
			write_network(network, d, k, G, randG);
			printf("-------------------------------------------------\n");
		}
	}
	else if(d == "2"){
		for(int k=1; k<=num_gen; ++k){
			printf("-------------------------------------------------\n");
			printf("Started %d-th generation of a randomized network with d = %s.\n", k, d.c_str());
			Network randG;
			randomizing_with_d_two(G, randG);
			write_network(network, d, k, G, randG);
			printf("-------------------------------------------------\n");
		}
	}
	else if(d == "2.5"){
		for(int k=1; k<=num_gen; ++k){
			printf("-------------------------------------------------\n");
			printf("Started %d-th generation of a randomized network with d = %s.\n", k, d.c_str());
			Network randG;
			randomizing_with_d_two_five(G, randG);
			write_network(network, d, k, G, randG);
			printf("-------------------------------------------------\n");
		}
	}
	else{
		printf("Error: Given d is not defined. The d-value should be 0, 1, 2, or 2.5.\n");
		exit(0);
	}

	return 0;
}