"""
python implement for
Hinge-Shift Mechanism Modulates Allosteric Regulations in Human Pin1
by Yi He, 2nd Aug 2023,
"""
import networkx as nx
import numpy as np
from .Mdanalysis import shrinkage_covariance_estimator
from .MDTASK import MDTASK_cov, calc_MDTASK_3Ncov, MDTASK_prs, residues_from_pdb
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def parse_trajPDB(traPDB_path):
     with open(traPDB_path,'r') as f:
        lines = f.readlines()
        flag = False
        traj_coord = []
        num_frames = 0
        num_all_residues = 0
        try:
            for line in lines:
                line = line.strip()
                words = line.split()
                if(line.startswith("MODEL")):
                    flag = True
                    num_frames += 1
                    prot_coord = []
                elif(line.startswith("ATOM") and words[2] == "CA" and flag):
                    calpha_xyz = [float(words[5]),float(words[6]),float(words[7])]
                    num_all_residues += 1 
                    prot_coord.append((calpha_xyz))
                elif(line.startswith("ENDMDL") and flag):
                    flag = False
                    traj_coord.append(prot_coord)
        except:
            raise ValueError('invalid trajectory pdb file!')
        num_residues = int(num_all_residues/num_frames)
        return traj_coord, num_residues, num_frames
     
def residues_from_pdb(traj_coord, step=50):
        traj_coord = np.array(traj_coord)
        traj_coord = traj_coord.transpose(1,0,2)
        dic ={}
        for i, residue in enumerate(traj_coord):
            frames = []
            for j, frame in enumerate(residue):
                if (j+1)%step == 0:
                    coords = []
                    for value in frame:
                        coords.append(float(value))
                    frames.append(coords)
            dic[i] = frames
        # {res: (frames, xyz)}
        return dic

def calc_cov(traj_coord, step=1):
    residues_dict = residues_from_pdb(traj_coord, step)
    correlation = MDTASK_cov(residues_dict)
    return correlation

def calc_shrinkage_3Ncov(traj_coord, num_residues):
    """
    Covariance contains the data for long-range interactions, solvation effects, and biochemical specificities of all types of interactions
    C: [3N,3N]
    """
    coordinates = np.array(traj_coord)
    coordinates = np.reshape(coordinates, (coordinates.shape[0], -1))
    reference_coordinates = None
    sigma = shrinkage_covariance_estimator(coordinates, reference_coordinates)
    weights = np.full(num_residues, 12.011, dtype=float)
    weights = np.repeat(weights, 3)
    weight_matrix = np.sqrt(np.identity(len(weights))*weights)
    sigma = np.dot(weight_matrix, np.dot(sigma, weight_matrix))
    return sigma

def compute_adjacency_matrix(positions, threshold):
    positions = np.array(positions)
    N = len(positions)
    adjacency = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < threshold:
                adjacency[i, j] = 1
    return adjacency

def compute_hessian(adjacency, positions, k):
    positions = np.array(positions)
    N = len(positions)
    H = np.zeros((3*N, 3*N))

    for i in range(N):
        for j in range(N):
            if adjacency[i, j]:
                diff = positions[i] - positions[j]
                for alpha in range(3):
                    for beta in range(3):
                        H[3*i + alpha, 3*j + beta] = -k * diff[alpha] * diff[beta]
                        
            if i == j:
                for alpha in range(3):
                    H[3*i + alpha, 3*i + alpha] += k

    return H

def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def calc_cos_2vec(v1, v2):
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compute_hessian_inverse(hessian):
    return np.linalg.inv(hessian)

def compute_displacements(hessian_inverse, external_force):
    displacements = 128*hessian_inverse.dot(external_force)
    return displacements

def perturbate(hessian_inverse, f_position, f_vector):
    external_force = np.zeros(hessian_inverse.shape[0])
    external_force[(3*f_position):(3*(f_position+1))] = f_vector  
    return compute_displacements(hessian_inverse, external_force).reshape(-1,3)

def isotropic_perturbate(hessian_inverse, position, n_ticks):
    response_profiles = np.zeros((int(hessian_inverse.shape[0]/3),3))
    forces = fibonacci_sphere(n_ticks)
    for f in forces:
        response_profiles += perturbate(hessian_inverse, position, f)
    return response_profiles

def perturbation_response_scan(hessian_inverse, n_ticks):
    for i in range(int(hessian_inverse.shape[0]/3)):
        if i == 0:
            rst = np.expand_dims(np.linalg.norm(isotropic_perturbate(hessian_inverse, i ,n_ticks),axis=1), axis=1)
        else:
            response_profiles = np.expand_dims(np.linalg.norm(isotropic_perturbate(hessian_inverse, i ,n_ticks),axis=1), axis=1)
            rst = np.concatenate((rst, response_profiles), axis=1)
    return rst

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def getDFI(prs, i):
    """
    arg i: residue i
    Dynamic Flexibility Index (DFI) is the whole response of i by perturbate all residues
    DFIi = ∑j(|ΔRj|i) / ∑j∑i(|ΔRj|i)
    where |ΔRj|i is the magnitude of the response at site i due to the perturbation at site j.
    """
    return np.sum(prs[i])/np.sum(prs)

def getDCI(prs, i, list):
    """
    arg i: residue i
    arg list: functional sites list

    dynamic coupling index (DCI) metric can identify sites that are distal to functional sites 
    but impact active site dynamics through dynamic allosteric coupling
    DCIi = ∑j_Nf(|ΔRj|i)/Nf / ∑j(|ΔRj|i)/N
    where Nf is number of functional sites 
    """
    sum_deltaR4i = 0
    for j in list:
        sum_deltaR4i += prs[i][j]
    return sum_deltaR4i/len(list)/np.sum(prs[i])/len(prs[i]) 

def plot_cov(matrix,result_folder):
    plt.figure(dpi=600)
    df = pd.DataFrame(matrix)
    df.columns = range(1, matrix.shape[1]+1,1)
    df.index = range(1, matrix.shape[1]+1,1)
    ax = sns.heatmap(df, cmap="Blues", linewidth=0, vmin = 0, vmax = 1)
    ax.set_title('Heatmap of PRS Covariance')
    ax.set_xlabel('Residues')
    ax.set_ylabel('Residues')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.tight_layout()
    plt.savefig(result_folder+'/prs_cov.png')
    plt.close()


def plot_edges(matrix,result_folder):
    labels=np.arange(1,matrix.shape[0]+1)
    df = pd.DataFrame(matrix)
    df.index = labels
    df.columns = labels
    linewidth=0.1/np.ceil(matrix.shape[0]/100)
    ax = sns.heatmap(df, linewidth=linewidth,cmap="Blues", vmax=1.0, vmin=0.0)
    ax.set_xlabel('Residues')
    ax.set_ylabel('Residues')
    ax.set_title('Heatmap of Pertubation Response Scanning')
    out_filepath = os.path.join(result_folder,"prs_res.png")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(out_filepath, dpi=600)
    plt.close()

def plot_dfis(dfis,result_folder):
    dfis = np.array(dfis)*100
    x = range(1, len(dfis)+1,1)
    plt.plot(x, dfis, linestyle="solid") 
    out_filepath = os.path.join(result_folder,"prs_dfi.png")
    plt.xlabel('Residues')
    plt.ylabel('%DFI')
    plt.title('DFI of Pertubation Response Scanning')
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(out_filepath,dpi=600)
    plt.close()

def calc_DFI_matrix(prs):
    ls = []
    for i in range(len(prs)):
        ls.append(getDFI(prs, i))
    return ls 

def cal_DCI_matrix(prs,list):
    ls = []
    for i in range(len(prs)):
        ls.append(getDCI(prs, i, list))
    return ls 

def tick2link_matrix(adjacency_matrix, hessian_inverse, position, theta, n_ticks):
    position = position-1
    response_profile = isotropic_perturbate(hessian_inverse, position, n_ticks)
    link_matrix = adjacency_matrix
    for i in range(response_profile.shape[0]):
        for j in range(response_profile.shape[0]):
            cos = calc_cos_2vec(response_profile[i],response_profile[j])
            if cos < theta:
                link_matrix[i, j] = 0
            if (i - 3) < j < (i + 3):
                link_matrix[i, j] = 0
    return link_matrix
    
def cal_path(link_matrix, result_folder, source_node, target_node, reset=False):
    num_residues = len(link_matrix)
    edges_list = list()
    # Default: i->j
    for i in range(num_residues):
        for j in range(num_residues):
            if i != j:
                if link_matrix[i, j] != 0:
                    edges_list.append((i+1, j+1, {'weight': link_matrix[j, i]}))
    MDG = nx.MultiDiGraph()
    MDG.add_edges_from(edges_list)

    target_nodes = [target_node]
    if reset:
        out_file = result_folder+'/prs_path_reset.txt'
    else:
        out_file = result_folder+'/prs_path.txt'

    path_dict = dict()
    for tn in target_nodes:
        try:
            path_dict[tn] = []
            path_nodes = nx.dijkstra_path(MDG, source_node, tn)
            path_length_list = []  # save the length of shorest path
            path_length_list.append(nx.dijkstra_path_length(MDG, source_node, tn))
            path_dict[tn].append('shortest_path : ' + '->'.join(list(map(str, path_nodes))) +
                                ' : ' + str(nx.dijkstra_path_length(MDG, source_node, tn)))
            if len(path_nodes) > 2:
                for ipn in range(1, len(path_nodes)):
                    tmp_MDG = MDG.copy()
                    tmp_MDG.remove_edge(path_nodes[ipn-1], path_nodes[ipn])
                    tmp_path_nodes = nx.dijkstra_path(tmp_MDG, source_node, tn)
                    path_length_list.append(
                        nx.dijkstra_path_length(tmp_MDG, source_node, tn))
                    path_dict[tn].append('remove(%d->%d) : ' % (path_nodes[ipn-1], path_nodes[ipn]) + '->'.join(
                        list(map(str, tmp_path_nodes))) + ' : ' + str(nx.dijkstra_path_length(tmp_MDG, source_node, tn)))
                # According to the shortest path length from small to large
                path_dict[tn] = np.array(path_dict[tn])[np.argsort(path_length_list)]
        except:
            print('no')
    with open(out_file, 'w') as f:
        for k, v in path_dict.items():
            f.write('target node : %d\n' % k)
            for tmp_path in v:
                f.write('\t\t' + tmp_path + '\n')

def prs_total(
        traPDBpath = "/home/hy/Documents/Project/NRIMD/jobsdata/ca_trajs/0000AAAAA.pdb",
        dist_threshold =10,
        hessian = 0,
        save_folder = '/home/hy/Documents/Project/NRIMD/jobsdata/jobs/0000AAAAA/analysis',
        source_node = 93,
        target_node = 139,
        cosine_threshold=0.5,
        n_ticks = 100,

        ):
    traj_coord, num_residues, num_frames = parse_trajPDB(traPDBpath)
    # correlation =  calc_cov(traj_coord, 100)
    # np.savetxt(save_folder+'/prs_cov0.dat',correlation)
    # plot_cov(correlation, save_folder)


    adjacency_matrix = compute_adjacency_matrix(traj_coord[0], dist_threshold)
    if hessian == 0:
        hessian = compute_hessian(adjacency_matrix, traj_coord[0], 1)
        hessian_inverse = compute_hessian_inverse(hessian)
    elif hessian == 1:
        hessian_inverse = calc_shrinkage_3Ncov(traj_coord, num_residues)
    elif hessian == 2:
        hessian_inverse = calc_MDTASK_3Ncov(traj_coord, num_residues, num_frames)
    else:
        return ValueError
    link_matrix = tick2link_matrix(adjacency_matrix, hessian_inverse, source_node, cosine_threshold, n_ticks)

    plot_cov(link_matrix, save_folder)
    cal_path(link_matrix, save_folder, source_node, target_node)

    prs = perturbation_response_scan(hessian_inverse, n_ticks)
    # prs = normalization(prs)
    np.savetxt(save_folder+'/prs_prs.dat',prs)
    plot_edges(prs, save_folder)

    dfis = calc_DFI_matrix(prs)
    plot_dfis(dfis, save_folder)

def prs_repath(
        traPDBpath = "/home/hy/Documents/Project/NRIMD/jobsdata/ca_trajs/0000AAAAA.pdb",
        dist_threshold = 10,
        hessian = 2,
        save_folder = '/home/hy/Documents/Project/NRIMD/jobsdata/jobs/0000AAAAA/analysis',
        source_node = 93,
        target_node = 139,
        cosine_threshold=0.5,
        n_ticks = 100,   
):
    traj_coord, num_residues, num_frames = parse_trajPDB(traPDBpath)
    adjacency_matrix = compute_adjacency_matrix(traj_coord[0], dist_threshold)
    if hessian == 0:
        hessian = compute_hessian(adjacency_matrix, traj_coord[0], 1)
        hessian_inverse = compute_hessian_inverse(hessian)
    elif hessian == 1:
        hessian_inverse = calc_shrinkage_3Ncov(traj_coord, num_residues)
    elif hessian == 2:
        hessian_inverse = calc_MDTASK_3Ncov(traj_coord, num_residues, num_frames)
    link_matrix = tick2link_matrix(adjacency_matrix, hessian_inverse, source_node, cosine_threshold, n_ticks)
    cal_path(link_matrix, save_folder, source_node, target_node,reset=True)
