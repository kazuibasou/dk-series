#include <iostream>
#include <vector>
#include <random>
#include <deque>
#include <set>
#include <unordered_map>
#include <algorithm>
#include "network.h"

JDM::JDM(){
	key_to_ids = std::unordered_map<int, int>();
	id_to_keys = std::vector<int>();
    entries = std::vector<std::vector<int>>();
}

JDM::~JDM(){
	std::unordered_map<int, int>().swap(key_to_ids);
	std::vector<int>().swap(id_to_keys);
    std::vector<std::vector<int>>().swap(entries);
}

int JDM::clear(){
	key_to_ids.clear();
	id_to_keys.clear();
	entries.clear();

	return 0;
}

int JDM::entry(const int k){
	int size = int(id_to_keys.size());

	id_to_keys.push_back(k);
	key_to_ids[k] = size;

	for(int i=0; i<size; ++i){
		entries[i].push_back(0);
	}

	entries.push_back(std::vector<int>(size+1, 0));

	return 0;
}

int JDM::get(const int k, const int l) const{
	if(key_to_ids.find(k) == key_to_ids.end()){
		return 0;
	}
	if(key_to_ids.find(l) == key_to_ids.end()){
		return 0;
	}

	int i_k = key_to_ids.at(k);
	int i_l = key_to_ids.at(l);

	return entries[i_k][i_l];
}

int JDM::add(const int k, const int l, const int value){
    if(key_to_ids.find(k) == key_to_ids.end()){
		entry(k);
	}

	if(key_to_ids.find(l) == key_to_ids.end()){
		entry(l);
	}

	int i_k = key_to_ids[k];
	int i_l = key_to_ids[l];

	entries[i_k][i_l] += value;

    return 0;
}

int JDM::subtract(const int k, const int l, const int value){
	if(key_to_ids.find(k) == key_to_ids.end()){
		entry(k);
	}

	if(key_to_ids.find(l) == key_to_ids.end()){
		entry(l);
	}

	int i_k = key_to_ids[k];
	int i_l = key_to_ids[l];

	entries[i_k][i_l] -= value;

    return 0;
}

int JDM::get_keys(std::vector<int> &keys) const{
	keys.clear();
	keys = std::vector<int>(id_to_keys);

	return 0;
}

Network::Network(){

	nlist = std::vector<std::vector<int>>();
	index_to_node = std::unordered_map<int, int>();
}

Network::~Network(){

	std::vector<std::vector<int>>().swap(nlist);
	std::unordered_map<int, int>().swap(index_to_node);
}

int Network::read_network(const char *network){

	const char *dir = "../data/";

	FILE *f;
	std::string fpath = std::string(dir) + network + ".txt";
	f = fopen(fpath.c_str(), "r");
	if(f == NULL) {
		printf("Error: Could not open file named %s.txt.\n", network);
		exit(0);
	}

	int u, v, i_u, i_v;
	N = 0;
	std::unordered_map<int, int> node_index;
	nlist = std::vector<std::vector<int>>();

	while(fscanf(f, "%d %d", &u, &v) != EOF) {
		if(node_index.find(u) == node_index.end()){
			index_to_node[N] = u;
			node_index[u] = N;
			N += 1;
		}

		if(node_index.find(v) == node_index.end()){
			index_to_node[N] = v;
			node_index[v] = N;
			N += 1;
		}

		if(N+1 > int(nlist.size())){
			nlist.resize(N+1, std::vector<int>());
		}

		i_u = node_index[u];
		i_v = node_index[v];

		nlist[i_u].push_back(i_v);
		nlist[i_v].push_back(i_u);
	}
	
	fclose(f);

	M = 0;
	max_k = 0;
	for(int v=0; v<N; ++v){
		std::sort(nlist[v].begin(),nlist[v].end());
		int k = int(nlist[v].size());
		M += k;
		if(k > max_k){
			max_k = k;
		}
	}

	M = int(M)/2;

	printf("-------------------------------------------------\n");
	printf("The given network named %s was successfully read.\n", network);
	printf("Number of nodes: %d\n", N);
	printf("Number of edges: %d\n", M);
	printf("-------------------------------------------------\n");

	return 0;
}

int Network::add_edge(const int v,const int w){
	//Add an edge between v and w.

	/*
	if(int(std::max(v,w)+1) > int(nlist.size())){
		nlist.resize(std::max(v,w)+1, std::vector<int>());
	}
	*/

	nlist[v].push_back(w);
	nlist[w].push_back(v);
	
	return 0;
}

int Network::remove_edge(const int v,const int w){
	//Remove an edge between v and w.
	
	auto itr1 = std::find(nlist[v].begin(), nlist[v].end(), w);

	/*
	if(itr1 == nlist[v].end()){
		printf("Error: node w is not included in neighbors of node v.\n");
		exit(0);
	}
	*/
	
	nlist[v].erase(itr1);

	
	auto itr2 = std::find(nlist[w].begin(), nlist[w].end(), v);
	
	/*
	if(itr2 == nlist[w].end()){
		printf("Error: node v is not included in neighbors of node w.\n");
		exit(0);
	}
	*/
	
	nlist[w].erase(itr2);
	
	return 0;
}

int Network::calc_num_jnt_deg(JDM &num_jnt_deg){
	
	num_jnt_deg.clear();

	int k, l;
	for(int v=0; v<N; ++v){
		k = int(nlist[v].size());
		for(int w:nlist[v]){
			l = int(nlist[w].size());
			num_jnt_deg.add(k, l, 1);
		}
	}

	return 0;
}

int Network::calc_degree_dependent_clustering_coefficient(std::vector<double> &ddcc){
	
	ddcc = std::vector<double>(max_k+1, 0.0);
	std::vector<int> N_k(max_k+1, 0);

	int u, w, k, i, j;
	double cc;

	for(int v=0; v<N; ++v){
		cc = 0.0;
		
		k = nlist[v].size();
		N_k[k] += 1;

		if(k < 2){
			continue;
		}

		for(i=0; i<k-1; ++i){
			u = nlist[v][i];
			for(j=i+1; j<k; ++j){
				w = nlist[v][j];
				if(v != u && v != w && u != w){
					cc += 2*std::count(nlist[u].begin(), nlist[u].end(), w);
				}
			}
		}

		cc = double(cc)/(k*(k-1));
		ddcc[k] += cc;
	}

	for(k=2; k<int(ddcc.size()); ++k){
		if(N_k[k] > 0){
			ddcc[k] = double(ddcc[k])/N_k[k];
		}
	}
	
	return 0;
}