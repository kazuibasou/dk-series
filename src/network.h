#ifndef NETWORK_H
#define NETWORK_H

class JDM{
	public:
		std::unordered_map<int, int> key_to_ids;
		std::vector<int> id_to_keys;
		std::vector<std::vector<int>> entries;

		JDM();
		~JDM();

		int clear();
		int entry(const int k);
		int get(const int k, const int l) const;
		int add(const int k, const int l, const int value);
		int subtract(const int k, const int l, const int value);
		int get_keys(std::vector<int> &keys) const;
};

class Network{
	public:
		int N;
		int M;
		int max_k;
		std::vector<std::vector<int>> nlist;
		std::unordered_map<int, int> index_to_node;

		Network();
		~Network();

		int read_network(const char *network);

		int add_edge(const int v,const int w);
		int remove_edge(const int v,const int w);
		int calc_num_jnt_deg(JDM &num_jnt_deg);
		int calc_degree_dependent_clustering_coefficient(std::vector<double> &ddcc);
};

#endif