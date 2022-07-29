import numpy as np
import os

def lowerTriFromMat(matrix):
    # gets the lower triangular matrix as vector from a square matrix
    if matrix.shape[0]!=matrix.shape[1]:
        os.error("This method can only use square matrices.")
    
    sideLen = matrix.shape[0]

    outVec = np.array([]) # create an empty array

    for j in range(sideLen):
        startIndex = 0
        endIndex = j

        outVec = np.append(outVec,matrix[j,startIndex:endIndex])

    return outVec

def listOfMatsto2D(listOfMats):
    # converts a list of matrices to a 2D-object

    numCols = len(listOfMats)
    numNodes = listOfMats[0].shape[0]
    numRows = int((numNodes**2-numNodes)/2)
    outObj = np.empty((numRows,numCols)) # create an empty array

    colNum = 0
    for mat in listOfMats:
        vecForMat = lowerTriFromMat(mat)
        outObj[:,colNum] = vecForMat
        colNum+=1

    return outObj

def matrixFromVec(vec):
    # build adjacency matrix from vector. Only works for 10-node graphs for now

    adj_mat = np.zeros((10,10))

    currentInd = 0

    for row in range(10):
        adj_mat[row,0:row] = vec[currentInd:currentInd+row]
        currentInd+=row

    # add symmetric counterpart
    adj_mat = adj_mat + adj_mat.T

    return adj_mat

def listOfMatsFrom2DObj(inputMat):

    listOfMats = []
    for j in range(inputMat.shape[1]):
        listOfMats.append(matrixFromVec(inputMat[:,j]))

    return listOfMats