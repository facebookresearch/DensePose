# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""DensePose utilities."""

from scipy.io import loadmat
import os.path as osp
import numpy as np
import scipy.spatial.distance as ssd
import pycocotools.mask as mask_util


def GetDensePoseMask(Polys, num_parts=14):
    """Get dense masks from the encoded masks."""
    MaskGen = np.zeros((256, 256), dtype=np.int32)
    for i in range(1, num_parts + 1):
        if Polys[i - 1]:
            current_mask = mask_util.decode(Polys[i - 1])
            MaskGen[current_mask > 0] = i
    return MaskGen


class DensePoseMethods:
    def __init__(self):
        ALP_UV = loadmat(
            osp.join(osp.dirname(__file__), '../../DensePoseData/UV_data/UV_Processed.mat')
        )
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]
        self.SemanticMaskSymmetries = [
            0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14
        ]
        self.Index_Symmetry_List = [
            1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23
        ]
        self.UV_symmetry_transformations = loadmat(
            osp.join(osp.dirname(__file__), '../../DensePoseData/UV_data/UV_symmetry_transforms.mat')
        )

    def get_symmetric_densepose(self, I, U, V, x, y, mask):
        """Get the mirror symmetric UV annotations"""
        symm_I = np.zeros_like(I)
        symm_U = np.zeros_like(U)
        symm_V = np.zeros_like(V)
        for i in range(24):
            inds = np.where(I == (i + 1))[0]
            if len(inds) > 0:
                symm_I[inds] = self.Index_Symmetry_List[i]
                loc_U = (U[inds] * 255).astype(np.int32)
                loc_V = (V[inds] * 255).astype(np.int32)
                symm_U[inds] = self.UV_symmetry_transformations['U_transforms'][0, i][loc_V, loc_U]
                symm_V[inds] = self.UV_symmetry_transformations['V_transforms'][0, i][loc_V, loc_U]

        flip_mask = np.fliplr(mask)
        symm_mask = np.zeros_like(mask)
        for i in range(1, 15):
            symm_mask[flip_mask == i] = self.SemanticMaskSymmetries[i]
        x_max = flip_mask.shape[1]
        symm_x = x_max - x
        symm_y = y
        return symm_I, symm_U, symm_V, symm_x, symm_y, symm_mask

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        u = P1 - P0; v = P2 - P0; w = P - P0
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if np.dot(vCrossW, vCrossU) < 0:
            return False

        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        if (np.dot(uCrossW, uCrossV) < 0):
            return False

        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        return ((r <= 1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self, P0, P1, P2, P):
        u = P1 - P0; v = P2 - P0; w = P - P0
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if np.dot(vCrossW, vCrossU) < 0:
            return -1, -1, -1
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        if np.dot(uCrossW, uCrossV) < 0:
            return -1, -1, -1
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        if ((r <= 1) & (t <= 1) & (r + t <= 1)):
            return 1 - (r + t), r, t
        else:
            return -1, -1, -1

    def IUV2FBC(self, I_point, U_point, V_point):
        """Convert IUV to FBC (faceIndex and barycentric coordinates)."""
        P = [U_point, V_point, 0]
        faceIndicesNow = np.where(self.FaceIndices == I_point)[0]
        FacesNow = self.FacesDensePose[faceIndicesNow]
        v0 = np.zeros_like(self.U_norm[FacesNow][:, 0])
        P_0 = np.vstack((self.U_norm[FacesNow][:, 0], self.V_norm[FacesNow][:, 0], v0)).transpose()
        P_1 = np.vstack((self.U_norm[FacesNow][:, 1], self.V_norm[FacesNow][:, 1], v0)).transpose()
        P_2 = np.vstack((self.U_norm[FacesNow][:, 2], self.V_norm[FacesNow][:, 2], v0)).transpose()

        for i, [P0, P1, P2] in enumerate(zip(P_0, P_1, P_2)) :
            bc1, bc2, bc3 = self.barycentric_coordinates(P0, P1, P2, P)
            if bc1 != -1:
                return faceIndicesNow[i], bc1, bc2, bc3

        # If the found UV is not inside any faces, select the vertex that is closest!
        D1 = ssd.cdist(np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = ssd.cdist(np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = ssd.cdist(np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()
        minD1 = D1.min(); minD2 = D2.min(); minD3 = D3.min()
        if ((minD1 < minD2) & (minD1 < minD3)):
            return faceIndicesNow[np.argmin(D1)], 1., 0., 0.
        elif ((minD2 < minD1) & (minD2 < minD3)):
            return faceIndicesNow[np.argmin(D2)], 0., 1., 0.
        else:
            return faceIndicesNow[np.argmin(D3)], 0., 0., 1.

    def FBC2PointOnSurface(self, face_ind, bc1, bc2, bc3, vertices):
        """Use FBC to get 3D coordinates on the surface."""
        Vert_indices = self.All_vertices[self.FacesDensePose[face_ind]] - 1
        # p = vertices[Vert_indices[0], :] * bc1 +  \
        #     vertices[Vert_indices[1], :] * bc2 +  \
        #     vertices[Vert_indices[2], :] * bc3 
        p = np.matmul(np.array([[bc1, bc2, bc3]]), vertices[Vert_indices]).squeeze()
        return p
