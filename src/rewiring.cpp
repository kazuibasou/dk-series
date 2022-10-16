#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>
#include <set>
#include <algorithm>
#include "network.h"
#include "rewiring.h"

int calculate_num_tri_to_add(Network &randG, const std::vector<int> &node_degree, 
	const int &u1, const int &v1, const int &v2, std::vector<int> &num_tri_to_add){
	
	int t_minus, t_plus;

	for(int k: randG.nlist[u1]){
		if(node_degree[k] <= 1 || u1 == k){continue;}

		if(v1 != k && node_degree[v1] > 1){
			t_minus = std::count(randG.nlist[v1].begin(), randG.nlist[v1].end(), k);
			num_tri_to_add[node_degree[u1]] -= t_minus;
			num_tri_to_add[node_degree[v1]] -= t_minus;
			num_tri_to_add[node_degree[k]] -= t_minus;
		}

		if(v2 != k && node_degree[v2] > 1){
			t_plus = std::count(randG.nlist[v2].begin(), randG.nlist[v2].end(), k);
			num_tri_to_add[node_degree[u1]] += t_plus;
			num_tri_to_add[node_degree[k]] += t_plus;
			num_tri_to_add[node_degree[v2]] += t_plus;
		}
	}

	return 0;
}

int targeting_rewiring_d_two_five(Network G, Network &randG){
	
	printf("Started targeting-rewiring process with d = 2.5.\n");

	int k;
	int k_size = randG.max_k+1;

	std::vector<int> node_degree(randG.N, 0);
	std::vector<int> N_k(k_size, 0);
	for(int v=0; v<randG.N; ++v){
		k = int(randG.nlist[v].size());
		node_degree[v] = k;
		N_k[k] += 1;
	}

	std::vector<double> coeff(k_size, 0.0);
	for(k=2; k<int(coeff.size()); ++k){
		if(N_k[k] == 0){continue;}
		coeff[k] = double(2)/(k*(k-1));
		coeff[k] = double(coeff[k])/N_k[k];
	}

	std::vector<double> target_ddcc(k_size, 0.0);
	G.calc_degree_dependent_clustering_coefficient(target_ddcc);

	std::vector<double> current_ddcc(k_size, 0.0);
	randG.calc_degree_dependent_clustering_coefficient(current_ddcc);
	
	double dist = 0.0;
    double norm = 0.0;
    for(int k=0; k<int(target_ddcc.size()); ++k){
    	if(N_k[k] > 0){
	    	norm += target_ddcc[k];
	    	dist += std::fabs(target_ddcc[k]-current_ddcc[k]);
    	}
    }

	std::vector<std::pair<int, int>> edge_list;
	for(int v=0; v<randG.N; ++v){
		for(int w:randG.nlist[v]){
			if(w >= v){
				edge_list.push_back({v, w});
			}
		}
	}

	const int M = int(edge_list.size());

	std::vector<double> rewired_ddcc(k_size, 0.0);
	std::vector<int> num_tri_to_add(k_size, 0);
	int i_e1, i_e2, tmp;
	int R = 500*M;
	int r, u1, v1, u2, v2, t_minus;
	std::pair<int, int> e1, e2;
	double rewired_dist, delta_dist;

	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());
	std::uniform_int_distribution<int> randM(0, M-1);

	printf("Initial L1 distance between target and present c(k): %lf\n", double(dist)/norm);

	for(r=0; r<R; ++r){
		rewired_ddcc = std::vector<double>(current_ddcc);
		num_tri_to_add = std::vector<int>(k_size, 0);
		rewired_dist = dist;

		i_e1 = randM(engine);
		i_e2 = randM(engine);
		e1 = edge_list[i_e1];
		e2 = edge_list[i_e2];
		u1 = e1.first;
		v1 = e1.second;
		u2 = e2.first;
		v2 = e2.second;

		while(u1 == u2 || v1 == v2 || node_degree[v1] != node_degree[v2]){
			i_e1 = randM(engine);
			i_e2 = randM(engine);
			e1 = edge_list[i_e1];
			e2 = edge_list[i_e2];
			u1 = e1.first;
			v1 = e1.second;
			u2 = e2.first;
			v2 = e2.second;
		}

		calculate_num_tri_to_add(randG, node_degree, u1, v1, v2, num_tri_to_add);
		calculate_num_tri_to_add(randG, node_degree, u2, v2, v1, num_tri_to_add);

		if(node_degree[v1] > 1 && node_degree[v2] > 1){
			t_minus = std::count(randG.nlist[v1].begin(), randG.nlist[v1].end(), v2);
			num_tri_to_add[node_degree[u1]] -= t_minus;
			num_tri_to_add[node_degree[v1]] -= 2*t_minus;
			num_tri_to_add[node_degree[u2]] -= t_minus;
			num_tri_to_add[node_degree[v2]] -= 2*t_minus;
		}
		
		if(node_degree[u1] > 1 && node_degree[u2] > 1){
			t_minus = std::count(randG.nlist[u2].begin(), randG.nlist[u2].end(), u1);
			if(node_degree[v1] > 1){
				num_tri_to_add[node_degree[u1]] -= t_minus;
				num_tri_to_add[node_degree[u2]] -= t_minus;
				num_tri_to_add[node_degree[v1]] -= t_minus;
			}
			if(node_degree[v2] > 1){
				num_tri_to_add[node_degree[u1]] -= t_minus;
				num_tri_to_add[node_degree[u2]] -= t_minus;
				num_tri_to_add[node_degree[v2]] -= t_minus;
			}
		}
		
		for(k=2; k<k_size; ++k){
			if(num_tri_to_add[k] == 0){continue;}
			rewired_ddcc[k] += double(num_tri_to_add[k]*coeff[k]);
			rewired_dist += std::fabs(target_ddcc[k] - rewired_ddcc[k]) - std::fabs(target_ddcc[k] - current_ddcc[k]);
		}

		delta_dist = rewired_dist - dist;

		if(delta_dist < 0){
			randG.remove_edge(edge_list[i_e1].first, edge_list[i_e1].second);
			randG.add_edge(edge_list[i_e1].first, edge_list[i_e2].second);
			randG.remove_edge(edge_list[i_e2].first, edge_list[i_e2].second);
			randG.add_edge(edge_list[i_e1].second, edge_list[i_e2].first);

			tmp = edge_list[i_e1].second;
			edge_list[i_e1].second = edge_list[i_e2].second;
			edge_list[i_e2].second = tmp;

			current_ddcc = std::vector<double>(rewired_ddcc);
			dist = rewired_dist;
		}
	}

	printf("Final L1 distance between target and present c(k): %lf\n", double(dist)/norm);

	return 0;
}