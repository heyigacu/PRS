"""
# Author: Yi He
# reference MDTASK
"""
import sys
import numpy as np
import math
from math import log10, floor, sqrt


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

def mean_dot(m1, m2, size):
    DOT = np.zeros(size)

    for t in range(size):
        DOT[t] = np.dot(m1[t],m2[t])

    return np.mean(DOT)

def MDTASK_cov(residues):

    sorted_residues = sorted(residues.keys())

    num_trajectories = len(residues[sorted_residues[0]])
    num_residues = len(residues)

    correlation = np.zeros((num_residues, num_residues))

    for a, key_a in enumerate(sorted_residues):
        i = residues[key_a]
        resI = np.array(i)
        meanI = np.tile((np.mean(resI, 0)),(num_trajectories, 1))
        idelta = resI - meanI
        magnitudeI = math.sqrt(mean_dot(idelta, idelta, num_trajectories))

        for b, key_b in enumerate(sorted_residues):
            j = residues[key_b]
            resJ = np.array(j)
            meanJ = np.tile((np.mean(resJ, 0)),(num_trajectories, 1))
            jdelta = resJ - meanJ
            magnitudeJ = math.sqrt(mean_dot(jdelta, jdelta, num_trajectories))

            meanDotIJ = mean_dot(idelta, jdelta, num_trajectories)
            magProd = magnitudeI * magnitudeJ

            correlation[a,b] = meanDotIJ/magProd
    return correlation



# ===========================N3cov&prs===========================
# Perform PRS calculations given and MD trajectory and a final state
# Author: David Penkler
# Update: Yi He
def superpose3D(ref, target, weights=None,refmask=None,targetmask=None,returnRotMat=False):
    """superpose3D performs 3d superposition using a weighted Kabsch algorithm : http://dx.doi.org/10.1107%2FS0567739476001873 & doi: 10.1529/biophysj.105.066654
    definition : superpose3D(ref, target, weights,refmask,targetmask)
    @parameter 1 :  ref - xyz coordinates of the reference structure (the ligand for instance)
    @type 1 :       float64 numpy array (nx3)
    ---
    @parameter 2 :  target - theoretical target positions to which we should move (does not need to be physically relevant.
    @type 2 :       float 64 numpy array (nx3)
    ---
    @parameter 3:   weights - numpy array of atom weights (usuallly between 0 and 1)
    @type 3 :       float 64 numpy array (n)
    @parameter 4:   mask - a numpy boolean mask for designating atoms to include
    Note ref and target positions must have the same dimensions -> n*3 numpy arrays where n is the number of points (or atoms)
    Returns a set of new coordinates, aligned to the target state as well as the rmsd
    """
    if weights == None :
        weights=1.0
    if refmask == None :
        refmask=np.ones(len(ref),"bool")
    if targetmask == None :
        targetmask=np.ones(len(target),"bool")
    #first get the centroid of both states
    ref_centroid = np.mean(ref[refmask]*weights,axis=0)
    #print ref_centroid
    refCenteredCoords=ref-ref_centroid
    #print refCenteredCoords
    target_centroid=np.mean(target[targetmask]*weights,axis=0)
    targetCenteredCoords=target-target_centroid
    #print targetCenteredCoords
    #the following steps come from : http://www.pymolwiki.org/index.php/OptAlign#The_Code and http://en.wikipedia.org/wiki/Kabsch_algorithm
    # Initial residual, see Kabsch.
    E0 = np.sum( np.sum(refCenteredCoords[refmask] * refCenteredCoords[refmask]*weights,axis=0),axis=0) + np.sum( np.sum(targetCenteredCoords[targetmask] * targetCenteredCoords[targetmask]*weights,axis=0),axis=0)
    reftmp=np.copy(refCenteredCoords[refmask])
    targettmp=np.copy(targetCenteredCoords[targetmask])
    #print refCenteredCoords[refmask]
    #single value decomposition of the dotProduct of both position vectors
    try:
        dotProd = np.dot( np.transpose(reftmp), targettmp* weights)
        V, S, Wt = np.linalg.svd(dotProd )
    except Exception:
        try:
            dotProd = np.dot( np.transpose(reftmp), targettmp)
            V, S, Wt = np.linalg.svd(dotProd )
        except Exception:
            print >> sys.stderr,"Couldn't perform the Single Value Decomposition, skipping alignment"
        return ref, 0
    # we already have our solution, in the results from SVD.
    # we just need to check for reflections and then produce
    # the rotation.  V and Wt are orthonormal, so their det's
    # are +/-1.
    reflect = float(str(float(np.linalg.det(V) * np.linalg.det(Wt))))
    if reflect == -1.0:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]
    rmsd = E0 - (2.0 * sum(S))
    rmsd = np.sqrt(abs(rmsd / len(ref[refmask])))   #get the rmsd
    #U is simply V*Wt
    U = np.dot(V, Wt)  #get the rotation matrix
    # rotate and translate the molecule
    new_coords = np.dot((refCenteredCoords), U)+ target_centroid  #translate & rotate
    #new_coords=(refCenteredCoords + target_centroid)
    #print U
    if returnRotMat :
        return new_coords,rmsd, U
    return new_coords,rmsd     

def round_sig(x, sig=2):
    return round(x,sig-int(floor(log10(x)))-1)


def align_frame(reference_frame, alternative_frame, aln=False):
    totalres = reference_frame.shape[0]
    return superpose3D(alternative_frame.reshape(totalres, 3), reference_frame)[0].reshape(1, totalres*3)[0]


def calc_rmsd(reference_frame, alternative_frame):
    return superpose3D(alternative_frame, reference_frame)[1]


def calc_MDTASK_3Ncov(traj_coord, num_residues, num_frames):
    trajectory = np.array(traj_coord).reshape((-1,num_residues*3))
    totalres = num_residues
    totalframes = num_frames

    aligned_mat = np.zeros((totalframes,3*totalres))
    frame_0 = trajectory[0].reshape(totalres, 3)

    for frame in range(0, totalframes):
        aligned_mat[frame] = align_frame(frame_0, trajectory[frame], False)

    del trajectory

    average_structure_1 = np.mean(aligned_mat, axis=0).reshape(totalres, 3)

    for i in range(0, 10):
        for frame in range(0, totalframes):
            aligned_mat[frame] = align_frame(average_structure_1, aligned_mat[frame], False)

        average_structure_2 = np.average(aligned_mat, axis=0).reshape(totalres, 3)

        rmsd = calc_rmsd(average_structure_1, average_structure_2)

        average_structure_1 = average_structure_2
        del average_structure_2

        if rmsd <= 0.000001:
            for frame in range(0, totalframes):
                aligned_mat[frame] = align_frame(average_structure_1, aligned_mat[frame], False)
            break

    meanstructure = average_structure_1.reshape(totalres*3)

    del average_structure_1

    R_mat = np.zeros((totalframes, totalres*3))
    for frame in range(0, totalframes):
        R_mat[frame,:] = (aligned_mat[frame,:]) - meanstructure


    RT_mat = np.transpose(R_mat)

    RT_mat = np.mat(RT_mat)
    R_mat = np.mat(R_mat)

    corr_mat = (RT_mat * R_mat)/ (totalframes-1)
    del aligned_mat
    del meanstructure
    del R_mat
    del RT_mat
    return corr_mat

def MDTASK_prs(traj_coord, N3cov, num_residues, perturbations):

    totalres=num_residues
    corr_mat = N3cov

    frame_initial = traj_coord[0]
    frame_final = traj_coord[-1]

    initial = np.zeros((totalres, 3))
    final = np.zeros((totalres, 3))

    res_index = 0
    for line_index, initial_line in enumerate(frame_initial):
        final_line = frame_final[line_index]
        if line_index >= 2 and res_index < totalres:
            initial_res = initial_line
            final_res = final_line
            initial[res_index,] = initial_res[2:]
            final[res_index,] = final_res[2:]
            res_index += 1


    final_alg = superpose3D(final, initial)[0]
    diffE = (final_alg-initial).reshape(totalres*3, 1)

    del final
    del final_alg

    perturbations = int(perturbations)
    diffP = np.zeros((totalres, totalres*3, perturbations))
    initial_trans = initial.reshape(1, totalres*3)

    for s in range(0, perturbations):
        for i in range(0, totalres):
            delF = np.zeros((totalres*3))
            f = 2 * np.random.random((3, 1)) - 1
            j = (i + 1) * 3

            delF[j-3] = round_sig(abs(f[0,0]), 5)* -1 if f[0,0]< 0 else round_sig(abs(f[0,0]), 5)
            delF[j-2] = round_sig(abs(f[1,0]), 5)* -1 if f[1,0]< 0 else round_sig(abs(f[1,0]), 5)
            delF[j-1] = round_sig(abs(f[2,0]), 5)* -1 if f[2,0]< 0 else round_sig(abs(f[2,0]), 5)

            diffP[i,:,s] = np.dot((delF), (corr_mat))
            diffP[i,:,s] = diffP[i,:,s] + initial_trans[0]

            diffP[i,:,s] = ((superpose3D(diffP[i,:,s].reshape(totalres, 3), initial)[0].reshape(1, totalres*3))[0]) - initial_trans[0]
            del delF

    del initial_trans
    del initial
    del corr_mat

    DTarget = np.zeros(totalres)
    DIFF = np.zeros((totalres, totalres, perturbations))
    RHO = np.zeros((totalres, perturbations))

    for i in range(0, totalres):
        DTarget[i] = sqrt(diffE[3*(i+1)-3]**2 + diffE[3*(i+1)-2]**2 + diffE[3*(i+1)-1]**2)

    for j in range(0, perturbations):
        for i in range(0, totalres):
            for k in range(0, totalres):
                DIFF[k,i,j] = sqrt((diffP[i, 3*(k+1)-3, j]**2) + (diffP[i, 3*(k+1)-2, j]**2) + (diffP[i, 3*(k+1)-1, j]**2))

    del diffP

    for i in range(0, perturbations):
        for j in range(0, totalres):
            RHO[j,i] = np.corrcoef(np.transpose(DIFF[:,j,i]), DTarget)[0,1]

    del DIFF
    del DTarget

    maxRHO = np.zeros(totalres)
    for i in range(0, totalres):
        maxRHO[i] = np.amax(abs(RHO[i,:]))

    return maxRHO




