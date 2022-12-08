from utils import sparse_to_adjlist
from scipy.io import loadmat

"""
	Read data and save the adjacency matrices to adjacency lists
	Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
"""


if __name__ == "__main__":

	prefix = './data/'

	# Amazon
	amz = loadmat('data/Amazon.mat')
	net_upu = amz['net_upu']
	net_usu = amz['net_usu']
	net_uvu = amz['net_uvu']
	amz_homo = amz['homo']

	sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
	sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
	sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
	sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')