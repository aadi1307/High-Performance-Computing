#include "NonLinearElasticity.cuh"
#include <cmath>
#include <vector>
#include <iostream>
#include <utility>
#include <functional>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <set>
#include <iostream>
//#include <cuda_runtime.h>
#include "Eigen/Dense"
//#include "eigen-3.4.0/Eigen/Dense"
//#include "eigen/3.3.9/Eigen/Sparse"
//#include "eigen-3.3.9/Eigen/Dense"
//#include "eigen-3.3.9/Eigen/IterativeLinearSolvers"

// Constructor
NLElasticity::NLElasticity() {
            
}
 
__global__ void print_kernel() {
    printf("Inside Print:\n");
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    printf("Thread index: %d\n", threadId);
}



// Constructor for NLElasticity class
NLElasticity::NLElasticity(const std::string& parameter1, int parameter2, int parameter3, int parameter4, 
                           double parameter5, double parameter6, double parameter7) {
    if (parameter1 == "HexMeshGridGeneral") {
        std::cout << "inside HexMeshGridGeneral: " << std::endl;
        
        // Create HexGridMeshGeneral
        nElemX = parameter2; 
        nElemY = parameter3; 
        nElemZ = parameter4;
        sizeX = parameter5; 
        sizeY = parameter6; 
        sizeZ = parameter7;
        createHexGridMeshGeneral(nElemX, nElemY, nElemZ, sizeX, sizeY, sizeZ);
    } 

    // Initialize other parameters
   

    int numGQ = 2;
    int N = numGQ;
     // Initialize xi_GQ matrix
    myXi.resize(3, std::vector<double>(N * N * N));

    // Initialize wt_GQ vector
    myWt.resize(N * N * N);

    GaussQuad3dHex(numGQ, myXi, myWt);

    print_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    myE = 1.0;
    myNu = 0.3;
    computeDMatrix();

    myBodyForce = std::vector<double>(myNumDOF, 0.0);


    myfC = std::vector<double>(myNumDOF, 0.0);
    myNodalDirichlet = std::vector<double>(myNumDOF, 0.0);
    myNodesPerSurface = 4;
    
        // Initialize boundary conditions if available
        if (!mesh.e.empty()) {
            int maxSurfaceNum = 0;
            for (size_t i = 0; i < mesh.e[mesh.e.size() - 1].size(); ++i) {
                maxSurfaceNum = std::max(maxSurfaceNum, mesh.e[mesh.e.size() - 1][i]);
            }

            // Resize myBCtype and myBCvalue
            myBCtype.resize(maxSurfaceNum+1, std::vector<int>(3, 0));
            myBCvalue.resize(maxSurfaceNum+1, std::vector<double>(3, 0.0));
            myNumBoundaries = mesh.e[1].size();

        }
    

    energyErrNorm = 0.0;
    L2ErrNorm = 0.0;

    RelativeL2ErrNorm = 0.0;
    RelativeEnergyErrNorm = 0.0;

    myU = Eigen::VectorXd::Zero(myNumDOF/3);
    myV = Eigen::VectorXd::Zero(myNumDOF/3);
    myW = Eigen::VectorXd::Zero(myNumDOF/3);
    myDeformation = Eigen::VectorXd::Zero(myNumDOF);
    myMaxDelta = 0.0;
    myF = Eigen::VectorXd::Zero(myNumDOF);
    myNeumanForce = Eigen::VectorXd::Zero(myNumDOF);

    myProblem = 0;
    myApplyFiniteStrain = 0;
    myVerbose = 0;
    myNonlinearSteps = 1;

    myElemVol = Eigen::VectorXd::Zero(myNumElems);

}

void NLElasticity::createHexGridMeshGeneral(int nElemX, int nElemY, int nElemZ, double sizeX, double sizeY, double sizeZ)
{
    double elemSizeX = sizeX/nElemX;  //length of each element in X direction
    double  elemSizeY = sizeY/nElemY; //length of each element in Y direction
    double  elemSizeZ = sizeZ/nElemZ; //length of each element in Y direction
    
    // Calculate the number of elements in each direction
    int numNodesX = static_cast<int>(sizeX / elemSizeX) + 1;
    int numNodesY = static_cast<int>(sizeY / elemSizeY) + 1;
    int numNodesZ = static_cast<int>(sizeZ / elemSizeZ) + 1;

    // Create vectors to store node locations
    std::vector<double> x(numNodesX);
    std::vector<double> y(numNodesY);
    std::vector<double> z(numNodesZ);

    // Generate x, y, and z arrays
    for (int i = 0; i < numNodesX; ++i) {
        x[i] = i * elemSizeX;
    }

    for (int j = 0; j < numNodesY; ++j) {
        y[j] = j * elemSizeY;
    }

    for (int k = 0; k < numNodesZ; ++k) {
        z[k] = k * elemSizeZ;
    }

    std::vector<std::vector<double> > p(3);

    // Fill the vector with x, y, z coordinates
    for (int k = 0; k < numNodesZ; ++k) {
        for (int j = 0; j < numNodesY; ++j) {
            for (int i = 0; i < numNodesX; ++i) {
                p[0].push_back(x[i]);
                p[1].push_back(y[j]);
                p[2].push_back(z[k]);
            }
        }
    }
    //  // Create a Mesh object
   // Update myMesh.p with nodal coordinates
    mesh.p = p;

    // Initialize element connectivity matrix
    std::vector<std::vector<int> > q(8, std::vector<int>(nElemX * nElemY * nElemZ, 0));

    // Fill element connectivity matrix (similar to the previous code)
    for (int i = 0; i < nElemZ; ++i) {
        for (int j = 0; j < nElemY; ++j) {
            for (int m = 0; m < nElemX; ++m) {
                int k = (i * nElemX * nElemY) + (j * nElemX) + m;
                q[0][k] = (i) * (nElemX + nElemY + 1) + k + j;
                q[1][k] = (i) * (nElemX + nElemY + 1) + k + j + 1;
                q[2][k] = (i) * (nElemX + nElemY + 1) + k + j + nElemX + 2;
                q[3][k] = (i) * (nElemX + nElemY + 1) + k + j + nElemX + 1;
                q[4][k] = (i + 1) * (nElemX + nElemY + 1) + k + j + nElemX * nElemY;
                q[5][k] = (i + 1) * (nElemX + nElemY + 1) + k + j + 1 + nElemX * nElemY;
                q[6][k] = (i + 1) * (nElemX + nElemY + 1) + k + j + nElemX * nElemY + nElemX + 2;
                q[7][k] = (i + 1) * (nElemX + nElemY + 1) + k + j + nElemX * nElemY + nElemX + 1;
            }
        }
    }

    // Update myMesh.q with the element connectivity matrix
    mesh.q = q;

    myNCoord = mesh.p.size();
    myNumElems = mesh.q[0].size();  // Assuming myMesh.q is a 2D vector
    myNumNodes = mesh.p[0].size();  // Assuming myMesh.p is a 2D vector
    
    myNodesPerElement = mesh.q.size();  // Assuming myMesh.q is a 2D vector
    myDOFPerNode = 3;
    myDOFPerElem = myDOFPerNode * myNodesPerElement;
    myNumDOF = myDOFPerNode * myNumNodes;

    std::cout << "myNumNodesCreateHex: " << myNumNodes << std::endl;

    if (myNodesPerElement == 8) {
        myElemShape = "hex";
    }

    int count = 0;
    std::vector<int> b(5, 0);
    std::vector<std::vector<int> > boundary(6 * nElemX * nElemX, std::vector<int>(5, 0));

    // y = 0
    for (int j = 0; j < nElemZ; ++j) {
        for (int i = 0; i < nElemX; ++i) {
            b[0] = (j) * (nElemX + 1) * (nElemY + 1) + i;
            b[1] = (j) * (nElemX + 1) * (nElemY + 1) + i + 1;
            b[2] = (j + 1) * (nElemX + 1) * (nElemY + 1) + i + 1;
            b[3] = (j + 1) * (nElemX + 1) * (nElemY + 1) + i;
            b[4] = 4;
            ++count;
            boundary[count - 1] = b;
        }
    }

    // y = 1
    for (int j = 0; j < nElemZ; ++j) {
        for (int i = 0; i < nElemX; ++i) {
            b[0] = (nElemX) * (nElemY + 1) + (j+1) * (nElemX + 1) * (nElemY + 1) + i;
            b[1] = (nElemX) * (nElemY + 1) + (j+1) * (nElemX + 1) * (nElemY + 1) + i + 1;
            b[2] = (nElemX) * (nElemY + 1) + (j) * (nElemX + 1) * (nElemY + 1) + i + 1;
            b[3] = (nElemX) * (nElemY + 1) + (j) * (nElemX + 1) * (nElemY + 1) + i;
            b[4] = 5;
            ++count;
            boundary[count - 1] = b;
        }
    }

    // z = 0
    for (int j = 0; j < nElemY; ++j) {
        for (int i = 0; i < nElemX; ++i) {
            b[0] = (j) * (nElemX + 1) + i;
            b[1] = (j) * (nElemX + 1) + i + 1;
            b[2] = (j+1) * (nElemX + 1) + i + 1;
            b[3] = (j+1) * (nElemX + 1) + i;
            b[4] = 0;
            ++count;
            boundary[count - 1] = b;
        }
    }

    // z = 1
    int start = (nElemX + 1) * (nElemY + 1) * nElemZ;
    for (int j = 0; j < nElemY; ++j) {
        for (int i = 0; i < nElemX; ++i) {
            b[0] = start + (j) * (nElemX + 1) + i;
            b[1] = start + (j) * (nElemX + 1) + i + 1;
            b[2] = start + (j+1) * (nElemX + 1) + i + 1;
            b[3] = start + (j+1) * (nElemX + 1) + i;
            b[4] = 2;
            ++count;
            boundary[count - 1] = b;
        }
    }

    // x = 0
    for (int j = 0; j < nElemZ; ++j) {
        for (int i = 0; i < nElemY; ++i) {
            b[0] = (j) * (nElemX + 1) * (nElemY + 1) + (i) * (nElemX + 1);
            b[1] = (j) * (nElemX + 1) * (nElemY + 1) + (i+1) * (nElemX + 1);
            b[2] = (j+1) * (nElemX + 1) * (nElemY + 1) + (i+1) * (nElemX + 1);
            b[3] = (j+1) * (nElemX + 1) * (nElemY + 1) + (i) * (nElemX + 1);
            b[4] = 3;
            ++count;
            boundary[count - 1] = b;
        }
    }

    // x = 1
    start = nElemX;
    for (int j = 0; j < nElemZ; ++j) {
        for (int i = 0; i < nElemY; ++i) {
            b[0] = start + (j) * (nElemX + 1) * (nElemY + 1) + (i) * (nElemX + 1);
            b[1] = start + (j) * (nElemX + 1) * (nElemY + 1) + (i+1) * (nElemX + 1);
            b[2] = start + (j+1) * (nElemX + 1) * (nElemY + 1) + (i+1) * (nElemX + 1);
            b[3] = start + (j+1) * (nElemX + 1) * (nElemY + 1) + (i) * (nElemX + 1);
            b[4] = 1;
            ++count;
            boundary[count - 1] = b;
        }
    }

    
    size_t numRows = boundary.size();
    size_t numCols = boundary[0].size();
     // Transpose the vector
    std::vector<std::vector<int>> transposed_boundary(numCols, std::vector<int>(numRows, 0));

    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            transposed_boundary[j][i] = boundary[i][j];
        }
    }

    // Update myMesh.e with the boundary matrix
    mesh.e = transposed_boundary;

}
void NLElasticity::ShapeFunction2D(const std::vector<double>& xi,
                       std::vector<double>& N,
                       std::vector<std::vector<double>>& gradN) {
    if (xi.size() == 2)
    {
        N.resize(4);
        gradN.resize(2, std::vector<double>(4));

        N[0] = 0.25 * ((1 - xi[0]) * (1 - xi[1]));
        N[1] = 0.25 * ((1 + xi[0]) * (1 - xi[1]));
        N[2] = 0.25 * ((1 + xi[0]) * (1 + xi[1]));
        N[3] = 0.25 * ((1 - xi[0]) * (1 + xi[1]));

        gradN[0][0] = 0.25 * (xi[1] - 1);
        gradN[0][1] = 0.25 * (1 - xi[1]);
        gradN[0][2] = 0.25 * (xi[1] + 1);
        gradN[0][3] = 0.25 * (-xi[1] - 1);

        gradN[1][0] = 0.25 * (xi[0] - 1);
        gradN[1][1] = 0.25 * (-xi[0] - 1);
        gradN[1][2] = 0.25 * (xi[0] + 1);
        gradN[1][3] = 0.25 * (1 - xi[0]);
    }
}

//for three-dimensional case
std::pair<std::vector<std::vector<double> >, std::vector<std::vector<double> > > NLElasticity::ShapeFunction(const std::vector<double>& xi) {
     std::vector<std::vector<double> > N(8, std::vector<double>(3, 0.0));
        std::vector<std::vector<double> > gradN(3, std::vector<double>(8, 0.0));

        // Calculate N
        N[0][0] = 0.125 * (1 - xi[0]) * (1 - xi[1]) * (1 - xi[2]);
        N[1][0] = 0.125 * (1 + xi[0]) * (1 - xi[1]) * (1 - xi[2]);
        N[2][0] = 0.125 * (1 + xi[0]) * (1 + xi[1]) * (1 - xi[2]);
        N[3][0] = 0.125 * (1 - xi[0]) * (1 + xi[1]) * (1 - xi[2]);
        N[4][0] = 0.125 * (1 - xi[0]) * (1 - xi[1]) * (1 + xi[2]);
        N[5][0] = 0.125 * (1 + xi[0]) * (1 - xi[1]) * (1 + xi[2]);
        N[6][0] = 0.125 * (1 + xi[0]) * (1 + xi[1]) * (1 + xi[2]);
        N[7][0] = 0.125 * (1 - xi[0]) * (1 + xi[1]) * (1 + xi[2]);

        // Add gradN terms
        gradN[0][0] = 0.25 / 2 * (xi[1] - 1) * (1 - xi[2]);
        gradN[0][1] = 0.25 / 2 * (1 - xi[1]) * (1 - xi[2]);
        gradN[0][2] = 0.25 / 2 * (xi[1] + 1) * (1 - xi[2]);
        gradN[0][3] = 0.25 / 2 * (-xi[1] - 1) * (1 - xi[2]);
        gradN[0][4] = 0.25 / 2 * (xi[1] - 1) * (1 + xi[2]);
        gradN[0][5] = 0.25 / 2 * (1 - xi[1]) * (1 + xi[2]);
        gradN[0][6] = 0.25 / 2 * (xi[1] + 1) * (1 + xi[2]);
        gradN[0][7] = 0.25 / 2 * (-xi[1] - 1) * (1 + xi[2]);

        gradN[1][0] = 0.25 / 2 * (xi[0] - 1) * (1 - xi[2]);
        gradN[1][1] = 0.25 / 2 * (-xi[0] - 1) * (1 - xi[2]);
        gradN[1][2] = 0.25 / 2 * (xi[0] + 1) * (1 - xi[2]);
        gradN[1][3] = 0.25 / 2 * (1 - xi[0]) * (1 - xi[2]);
        gradN[1][4] = 0.25 / 2 * (xi[0] - 1) * (1 + xi[2]);
        gradN[1][5] = 0.25 / 2 * (-xi[0] - 1) * (1 + xi[2]);
        gradN[1][6] = 0.25 / 2 * (xi[0] + 1) * (1 + xi[2]);
        gradN[1][7] = 0.25 / 2 * (1 - xi[0]) * (1 + xi[2]);

        gradN[2][0] = - 0.25 / 2 * (1 - xi[0]) * (1 - xi[1]);
        gradN[2][1] = - 0.25 / 2 * (1 + xi[0]) * (1 - xi[1]);
        gradN[2][2] = - 0.25 / 2 * (1 + xi[0]) * (1 + xi[1]);
        gradN[2][3] = - 0.25 / 2 * (1 - xi[0]) * (1 + xi[1]);
        gradN[2][4] = 0.25 / 2 * (1 - xi[0]) * (1 - xi[1]);
        gradN[2][5] = 0.25 / 2 * (1 + xi[0]) * (1 - xi[1]);
        gradN[2][6] = 0.25 / 2 * (1 + xi[0]) * (1 + xi[1]);
        gradN[2][7] = 0.25 / 2 * (1 - xi[0]) * (1 + xi[1]);
     
    auto p1 = std::make_pair(N, gradN);
    return p1;
}

// Function to print Eigen::MatrixXd matrix with a given name
void NLElasticity::printEigenMatrix(const Eigen::MatrixXd& matrix, const std::string& matrixName) {
    std::cout << "Matrix " << matrixName << ":" << std::endl;
    std::cout << matrix << std::endl;
}

// Function to compute the Jacobian for an element
std::vector<std::vector<double> > NLElasticity::Jacobian(int elem, const std::vector<double> xi) {
     
   
    auto result = ShapeFunction(xi);
    std::vector<std::vector<double> > N = result.first;
    std::vector<std::vector<double> > gradN = result.second;
    std::vector<int> nodes;

    for (int i = 0; i < myNodesPerElement; ++i) {
        nodes.push_back(mesh.q[i][elem]);
    }
    std::vector<std::vector<double>> positionNodes(3, std::vector<double>(8, 0.0));

    for (size_t i = 0; i < nodes.size(); ++i) {
        for (int j = 0; j < myNCoord; ++j) {
            positionNodes[j][i] = mesh.p[j][nodes[i]];
        }
    }
    // Compute Jacobian
    std::vector<std::vector<double> > J(3, std::vector<double>(3, 0.0));
    
         for (int i = 0; i < J.size(); ++i) {
            for (int j = 0; j < positionNodes.size(); ++j) {
                for (int k = 0; k < positionNodes[0].size(); ++k) {
                    J[i][j] += gradN[i][k] * positionNodes[j][k];
                }
            }
        }


    return J;
}

    // Function to compute the Jacobian with provided nodes
std::vector<std::vector<double> > NLElasticity::JacobianWithNodes(const std::vector<std::vector<double> >& positionNodes, const std::vector<double> xi) {
    
    auto result = ShapeFunction(xi);
    std::vector<std::vector<double> > N = result.first;
    std::vector<std::vector<double> > gradN = result.second;
    // Compute Jacobian
    std::vector<std::vector<double> > J(2, std::vector<double>(2, 0.0));
    for (size_t i = 0; i < gradN.size(); ++i) {
        J[0][0] += gradN[0][i] * positionNodes[0][i];
        J[0][1] += gradN[0][i] * positionNodes[1][i];
        J[1][0] += gradN[1][i] * positionNodes[0][i];
        J[1][1] += gradN[1][i] * positionNodes[1][i];
    }

    return J;
}

//FEA Solver Begins
void NLElasticity::setMaterialParameter(double E) {
    myE = E;
    myD.resize(myNCoord, std::vector<double>(myNCoord, 0));

    for (int i = 0; i < myNCoord; ++i) {
        myD[i][i] = E;
    }
}

void NLElasticity::setYoungsModulus(double E) {
    myE = E;
    computeDMatrix();
}

void NLElasticity::setPoissonsRatio(double nu) {
    myNu = nu;
    computeDMatrix();
}

void NLElasticity::setMaterialModel(const std::string& materialModel) {
    myMaterialModel = materialModel;
}

void NLElasticity::setBulkModulus(double K1) {
    myBulkModulus = K1;
}

void NLElasticity::setShearModulus(double mu1) {
    myMu = mu1;
}

void NLElasticity::computeDMatrix() {
    double E = myE;
    double nu = myNu;

    myD.resize(6, std::vector<double>(6, 0));

    double factor = E / ((1 + nu) * (1 - 2 * nu));

    myD[0][0] = myD[1][1] = myD[2][2] = (1 - nu) * factor;
    myD[0][1] = myD[0][2] = myD[1][0] = myD[1][2] = myD[2][0] = myD[2][1] = nu * factor;
    myD[3][3] = myD[4][4] = myD[5][5] = (0.5 - nu) * factor;
}

//apply Dirichlet BC On specified DOFs
void NLElasticity::applyDirichletOnDOF(double value, int dof) {
    // Ensure myNodalDirichlet is large enough
    if (dof >= myNodalDirichlet.size()) {
        myNodalDirichlet.resize(dof + 1, 0.0);
    }

    myNodalDirichlet[dof] = value;

    // Add dof to myFixedDOF and myFixedNodes if not already present
    if (std::find(myFixedDOF.begin(), myFixedDOF.end(), dof) == myFixedDOF.end()) {
        myFixedDOF.push_back(dof);
    }

    // Assuming 'node' is the same as 'dof' in this context
    if (std::find(myFixedNodes.begin(), myFixedNodes.end(), dof) == myFixedNodes.end()) {
        myFixedNodes.push_back(dof);
    }
} 

//apply Dirichlet BC On Edge
void NLElasticity::applyDirichletOnSurface(double value, int SurfaceNum) {
    std::vector<int> boundaryNodes;

    // Loop through the last row of `myMesh.e` to find the surface number
    for (size_t i = 0; i < mesh.e.back().size(); ++i) {
        if (mesh.e.back()[i] == SurfaceNum) {
            // Add the nodes from the corresponding column
            for (size_t j = 0; j < mesh.e.size() - 1; ++j) {
                int node = mesh.e[j][i];
                boundaryNodes.push_back(node);
            }
        }
    }

    // Remove duplicates
    std::sort(boundaryNodes.begin(), boundaryNodes.end());
    boundaryNodes.erase(std::unique(boundaryNodes.begin(), boundaryNodes.end()), boundaryNodes.end());

    // Apply the Dirichlet boundary condition and update myFixedDOF and myFixedNodes
    for (int node : boundaryNodes) {
        if (node >= myNodalDirichlet.size()) {
            myNodalDirichlet.resize(node + 1, 0.0);
        }
        myNodalDirichlet[node] = value;

        if (std::find(myFixedDOF.begin(), myFixedDOF.end(), node) == myFixedDOF.end()) {
            myFixedDOF.push_back(node);
        }

        if (std::find(myFixedNodes.begin(), myFixedNodes.end(), node) == myFixedNodes.end()) {
            myFixedNodes.push_back(node);
        }
    }

}

// Function to apply Dirichlet boundary conditions on specified edges
void NLElasticity::applyDirichletOnSurfaceFunction(const std::string& uString,
                                        const std::string& vString,
                                        const std::string& wString,
                                        int EdgeNum) {
    // Find indices of edges specified by EdgeNum
    std::vector<int> index;
   
    for (size_t i = 0; i < mesh.e[0].size(); ++i) {
        if (mesh.e[mesh.e.size() - 1][i] == EdgeNum) {
            index.push_back(i);
        }
    }

    // Extract boundary nodes
    std::vector<std::vector<double> > boundary;
    size_t numRows = mesh.e.size() - 1;
    for (size_t j = 0; j < numRows; ++j) {
        std::vector<double> boundaryColVec;

         for (size_t i = 0; i < index.size(); ++i) {
            {
                boundaryColVec.push_back(mesh.e[j][index[i]]);
            }
        }
        boundary.push_back(boundaryColVec);
    }

     // Flatten the 2D vector into a 1D vector
    std::vector<int> flattened;
    for (const auto& row : boundary) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }

    // Convert the 1D vector to a set to remove duplicates
    std::set<int> uniqueSet(flattened.begin(), flattened.end());
    std::vector<int> node;
        for (const auto& elem : uniqueSet) {
            node.push_back(elem);
        }


    // Extract unique nodes and corresponding DOFs
    std::vector<int> udof, vdof, wdof;
    for (size_t i = 0; i < node.size(); ++i) {
        int currentNode = node[i];
        udof.push_back(3 * currentNode);
        vdof.push_back(3 * currentNode+1);
        wdof.push_back(3 * currentNode+2);
    }

    // Apply Dirichlet conditions based on the provided functions
    for (size_t i = 0; i < node.size(); ++i) {
        double x = mesh.p[0][node[i]];
        double y = mesh.p[1][node[i]];
        double z = mesh.p[2][node[i]];

        // Evaluate the expression
    
        double result = 0;
        myNodalDirichlet[udof[i]] = result;
        myNodalDirichlet[vdof[i]] = result;
        myNodalDirichlet[wdof[i]] = result;

 
    // Set the expression to the parser
   
 
    // Evaluate the expression
   

        myFixedDOF.push_back(udof[i]);
        myFixedDOF.push_back(vdof[i]);
        myFixedDOF.push_back(wdof[i]);
        myFixedNodes.push_back(node[i]);
    }
}

void NLElasticity::applyNeumanOnSurface(double forceU, double forceV, double forceW, size_t SurfaceNum) {
    // Ensure that myBCtype and myBCvalue have enough rows
    // Set values for the specified surface
    myBCtype[SurfaceNum][0] = 0;
    myBCvalue[SurfaceNum][0] = forceU;

    myBCtype[SurfaceNum][1] = 0;
    myBCvalue[SurfaceNum][1] = forceV;

    myBCtype[SurfaceNum][2] = 0;
    myBCvalue[SurfaceNum][2] = forceW;

}

// Function to calculate Gauss quadrature points for 2D quadrilateral
void NLElasticity::GaussQuad2dQuad(int numGQ, std::vector<double>& xi_GQ, std::vector<double>& wt_GQ) {
    // Gauss quadrature pts for a line (-1 to 1)
    if (numGQ == 0) {
        numGQ = 4;
    }
    
    int N = static_cast<int>(std::sqrt(numGQ));
    std::vector<double> x, w;

    // Call lgwt function
    lgwt(N, -1.0, 1.0, x, w);

    xi_GQ.resize(2 * N * N);
    wt_GQ.resize(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            xi_GQ[2 * (i * N + j)] = x[i];
            xi_GQ[2 * (i * N + j) + 1] = x[j];
            wt_GQ[i * N + j] = w[i] * w[j];
        }
    }
}

void NLElasticity::GaussQuad3dHex(int numGQ, std::vector<std::vector<double>>& xi_GQ, std::vector<double>& wt_GQ) {
    int N = numGQ;
    
    // Initialize xi_GQ with the correct size

    Eigen::VectorXd x, w;
    x.resize(N);
    w.resize(N);
 // Set the desired Gauss points
    std::vector<double> gaussPoints = {0.577350269189626, -0.577350269189626};
    
    // Populate xi values at quadrature points
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                xi_GQ[0][i * N * N + j * N + k] = gaussPoints[k];
                xi_GQ[1][i * N * N + j * N + k] = gaussPoints[j];
                xi_GQ[2][i * N * N + j * N + k] = gaussPoints[i];
            }
        }
    }

    // Populate weight values at quadrature points (uniform weight for now)
    wt_GQ.resize(N * N * N);
    std::fill(wt_GQ.begin(), wt_GQ.end(), 1.0);
}

// Function to print a vector of vectors
void NLElasticity::printVectorOfVectors(const std::vector<std::vector<double>>& vec) {
    std::cout << "Vector of Vectors:" << std::endl;
    for (const auto& row : vec) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

void NLElasticity::lgwt(int N, double a, double b, std::vector<double>& x, std::vector<double>& w) {
    N = N - 1;
    int N1 = N + 1;
    int N2 = N + 2;

    x.resize(N1);
    w.resize(N1);

    std::vector<double> xu(N1);

    // Initialize xu
    for (int i = 0; i < N1; ++i) {
        xu[i] = -1.0 + 2.0 * static_cast<double>(i) / N;
    }

    // Initial guess
    std::vector<double> y(N1);
    for (int i = 0; i < N1; ++i) {
        y[i] = std::cos((2.0 * i + 1.0) * M_PI / (2.0 * N + 2.0)) + (0.27 / N1) * std::sin(M_PI * xu[i] * N / N2);
    }

    // Legendre-Gauss Vandermonde Matrix
    std::vector<std::vector<double>> L(N1, std::vector<double>(N2, 0.0));
    // Derivative of LGVM
    std::vector<std::vector<double>> Lp(N1, std::vector<double>(N2, 0.0));

    double eps = 1e-15;  // Adjust epsilon as needed

    double y0 = 2.0;

    // Iterate until new points are uniformly within epsilon of old points
    while (true) {
        for (int i = 0; i < N1; ++i) {
            L[i][0] = 1.0;
            Lp[i][0] = 0.0;

            L[i][1] = y[i];
            Lp[i][1] = 1.0;

            for (int k = 2; k < N1; ++k) {
                L[i][k] = ((2 * k - 1) * y[i] * L[i][k - 1] - (k - 1) * L[i][k - 2]) / k;
            }
        }

        for (int i = 0; i < N1; ++i) {
            Lp[i][N1 - 1] = (N2) * (L[i][N1 - 1] - y[i] * L[i][N2 - 1]) / (1 - y[i] * y[i]);
        }

        y0 = y[0];
        y[0] = y0 - L[0][N2 - 1] / Lp[0][N2 - 1];

        // Check convergence
        bool converged = true;
        for (int i = 0; i < N1; ++i) {
            if (std::abs(y[i] - y0) > eps) {
                converged = false;
                break;
            }
        }

        if (converged) {
            break;
        }
    }

    // Copy results to output vectors x and w
    for (int i = 0; i < N1; ++i) {
        x[i] = 0.5 * (a * (1 - y[i]) + b * (1 + y[i]));
        w[i] = 0.5 * (b - a) / ((1 - y[i] * y[i]) * Lp[i][N2 - 1] * Lp[i][N2 - 1]);
    }
}

// Function to transpose a matrix
std::vector<std::vector<double>> NLElasticity::transpose(const std::vector<std::vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

// Function to multiply two matrices
std::vector<std::vector<double>> NLElasticity::multiplyMatrix(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    int rows1 = matrix1.size();
    int cols1 = matrix1[0].size();
    int rows2 = matrix2.size();
    int cols2 = matrix2[0].size();

    std::vector<std::vector<double>> result(rows1, std::vector<double>(cols2, 0.0));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

// Function to calculate the determinant of a matrix
double NLElasticity::determinant(const std::vector<std::vector<double>>& matrix) {
    int n = matrix.size();

    // Base case: If the matrix is 1x1, return its only element
    if (n == 1) {
        return matrix[0][0];
    }

    double det = 0.0;

    for (int i = 0; i < n; ++i) {
        // Calculate the cofactor (submatrix without the current row and column)
        std::vector<std::vector<double>> submatrix;
        for (int j = 1; j < n; ++j) {
            std::vector<double> row;
            for (int k = 0; k < n; ++k) {
                if (k != i) {
                    row.push_back(matrix[j][k]);
                }
            }
            submatrix.push_back(row);
        }

        // Recursive call to calculate the determinant of the cofactor
        double cofactor = matrix[0][i] * determinant(submatrix);

        // Alternating signs
        if (i % 2 == 0) {
            det += cofactor;
        } else {
            det -= cofactor;
        }
    }

    return det;
}

std::vector<double> NLElasticity::integrateOverBoundary(int geomEdge, int seg, const std::vector<double>& wt_GQ,
                                 const std::vector<Eigen::MatrixXd>& N2D,
                                 const std::vector<Eigen::MatrixXd>& gradN2D, int dof) {
        // assumption: surface is parallel to the x,y, or z planes.
        // Inclined surfaces not considered
        
        int nNodeVecSize = mesh.e.size()-1;
        std::vector<double> xNodes(nNodeVecSize);
        std::vector<double> yNodes(nNodeVecSize);
        std::vector<double> zNodes(nNodeVecSize);
        std::vector<int> nodes(nNodeVecSize);
        for (size_t i = 0; i < nNodeVecSize; ++i) {
            int nodeID = mesh.e[i][seg];
            nodes[i]=nodeID;
            xNodes[i] = mesh.p[0][nodeID];
            yNodes[i] = mesh.p[1][nodeID];
            zNodes[i] = mesh.p[2][nodeID];
        }
        

        std::vector<double> rangeXYZ(3);
        rangeXYZ[0] = *std::max_element(xNodes.begin(), xNodes.end()) - *std::min_element(xNodes.begin(), xNodes.end());
        rangeXYZ[1] = *std::max_element(yNodes.begin(), yNodes.end()) - *std::min_element(yNodes.begin(), yNodes.end());
        rangeXYZ[2] = *std::max_element(zNodes.begin(), zNodes.end()) - *std::min_element(zNodes.begin(), zNodes.end());

        auto minRangeInd = std::min_element(rangeXYZ.begin(), rangeXYZ.end()) - rangeXYZ.begin();
        std::vector<int> remainingIndVec = {0, 1, 2};
        remainingIndVec.erase(std::remove(remainingIndVec.begin(), remainingIndVec.end(), minRangeInd), remainingIndVec.end());

        std::vector<double> fBoundaryElem(N2D[0].size(), 0.0);

        for (size_t g = 0; g < wt_GQ.size(); ++g) {
            const Eigen::MatrixXd N = N2D[g];
            const Eigen::MatrixXd gradN = gradN2D[g];
            std::vector<std::vector<double>> positionNodes(remainingIndVec.size(), std::vector<double>(nodes.size()));

            for (size_t i = 0; i < remainingIndVec.size(); ++i) {
                std::transform(nodes.begin(), nodes.end(), positionNodes[i].begin(),
                               [&, i](int node) { return mesh.p[remainingIndVec[i]][node]; });
            }

            Eigen::MatrixXd positionNodesEigen(positionNodes.size(), positionNodes[0].size());
            for (int i = 0; i < positionNodes.size(); ++i) {
                for (int j = 0; j < positionNodes[0].size(); ++j) {
                    positionNodesEigen(i, j) = positionNodes[i][j];
                }
            }
            
            // Perform matrix multiplication
            Eigen::MatrixXd J = gradN * positionNodesEigen.transpose();

            double detJ = J.determinant();

            double f = myBCvalue[geomEdge][dof];  // flux per unit length

            for (size_t i = 0; i < fBoundaryElem.size(); ++i) {
                fBoundaryElem[i] += wt_GQ[g] * detJ * N(0,i) * f;
            }
        }

        return fBoundaryElem;
    }

void NLElasticity::assembleBC() {
    // Dirichlet boundary conditions
    myfC = myNodalDirichlet;
    myShape = "quad";
    
    //GaussQuad2dQuad(4, xi_GQ, wt_GQ); // need to check lgwt function to get correct quadrature and wts
    //for now directly adding the values
    std::vector<std::vector<double>> xi_GQ = {
        {0.5774, 0.5774, -0.5774, -0.5774},
        {0.5774, -0.5774, 0.5774, -0.5774}
    };

    // Initialize wt_GQ
    std::vector<double> wt_GQ = {1.0000, 1.0000, 1.0000, 1.0000};

   // Assuming xi_GQ.size() is the correct size
     std::vector<Eigen::MatrixXd> N2DCell;
     std::vector<Eigen::MatrixXd> gradN2DCell;


    for (size_t i = 0; i < xi_GQ[0].size(); ++i) {
        std::vector<double> N2D;
        std::vector<std::vector<double>> gradN2D;
        std::vector<double> xi(xi_GQ.size());
        for (size_t j = 0; j < xi_GQ.size(); ++j) {
            xi[j] = xi_GQ[j][i];
        }
        ShapeFunction2D(xi, N2D, gradN2D);
       
        Eigen::Map<Eigen::MatrixXd> N2DEigen(N2D.data(), N2D.size(), 1);
        Eigen::MatrixXd N2DEigenTransposed = N2DEigen.transpose();
        N2DCell.push_back(N2DEigenTransposed);
       
        int rows = gradN2D.size();
        int cols = gradN2D[0].size();
        Eigen::MatrixXd gradN2DEigen(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                gradN2DEigen(i, j) = gradN2D[i][j];
            }
        }

        //Eigen::MatrixXd gradNEigenTransposed = gradN2DEigen.transpose();
        gradN2DCell.push_back(gradN2DEigen);
       
    }

    std::vector<double> fBoundary(myNumDOF, 0.0);
    myForcedNodes.clear();

    // Neumann boundary conditions
    int numEdges = *std::max_element(mesh.e.back().begin(), mesh.e.back().end());
    for (int geomEdge = 0; geomEdge <= numEdges; ++geomEdge) {
        int typeu = myBCtype[geomEdge][0];
        int typev = myBCtype[geomEdge][1];
        int typew = myBCtype[geomEdge][2];
        double valueu = myBCvalue[geomEdge][0];
        double valuev = myBCvalue[geomEdge][1];
        double valuew = myBCvalue[geomEdge][2];
        if (valuew == -8000)
        {
            int temp = 1;
        }
        if (typeu == 0 && std::abs(valueu) > 0) {
           size_t boundarySegments = 0;

            // Iterate through the columns of mesh.e
            for (const auto& column : mesh.e) {
                // Check if the vector is not empty and the last element of the vector matches geomEdge
                if (!column.empty() && column.back() == geomEdge) {
                    // The last element matches, increment the count
                    boundarySegments++;
                }
            }

            for (size_t seg = 0; seg < boundarySegments; ++seg) {
                auto nodes = std::vector<int>(mesh.e[seg].begin(), mesh.e[seg].end() - 1);
                
                // Transform nodes to udof using std::transform and a lambda function
                int nodeSize = nodes.size();
                std::vector<int> udof(nodeSize);
                for (size_t i = 0; i < nodeSize; ++i) {
                    udof[i]=(3*nodes[i]);
                }
                std::vector<double> fBoundaryElem = integrateOverBoundary(geomEdge, seg, wt_GQ, N2DCell, gradN2DCell, 0);

                for (auto dof : udof) {
                    fBoundary[dof] += fBoundaryElem[dof];
                }

                std::set<int> uniqueSet(nodes.begin(), nodes.end());
                for (const auto& elem : uniqueSet) {
                    myForcedNodes.push_back(elem);
                }
            }

        }

        // Similar blocks for typev and typew
        if (typev == 0 && std::abs(valuev) > 0) {
           size_t boundarySegments = 0;

            // Iterate through the columns of mesh.e
            for (const auto& column : mesh.e) {
                // Check if the vector is not empty and the last element of the vector matches geomEdge
                if (!column.empty() && column.back() == geomEdge) {
                    // The last element matches, increment the count
                    boundarySegments++;
                }
            }

            for (size_t seg = 0; seg < boundarySegments; ++seg) {
                auto nodes = std::vector<int>(mesh.e[seg].begin(), mesh.e[seg].end() - 1);
                
                // Transform nodes to udof using std::transform and a lambda function
                int nodeSize = nodes.size();
                std::vector<int> vdof(nodeSize);
                for (size_t i = 0; i < nodeSize; ++i) {
                    vdof[i]=(3*nodes[i]+1);
                }
                std::vector<double> fBoundaryElem = integrateOverBoundary(geomEdge, seg, wt_GQ, N2DCell, gradN2DCell, 1);

                for (auto dof : vdof) {
                    fBoundary[dof] += fBoundaryElem[dof];
                }

                // Convert the 1D vector to a set to remove duplicates
                std::set<int> uniqueSet(nodes.begin(), nodes.end());
                for (const auto& elem : uniqueSet) {
                    myForcedNodes.push_back(elem);
                }
            }

        }

        std::set<int> uniqueSet(myForcedNodes.begin(), myForcedNodes.end());
        for (const auto& elem : uniqueSet) {
            myForcedNodes.push_back(elem);
        }

        if (typew == 0 && std::abs(valuew) > 0) {
           std::vector<int>  boundarySegments;
            int nMeshElastColSize = mesh.e[mesh.e.size() - 1].size();
             for (size_t i = 0; i < nMeshElastColSize; ++i) {
                if ( mesh.e[mesh.e.size() - 1][i] == geomEdge){
                    // The last element matches, increment the count
                    boundarySegments.push_back(i);
                }
            }
            for (size_t seg = 0; seg < boundarySegments.size(); ++seg) {
                std::vector<int> nodes;
                for (size_t i = 0; i < mesh.e.size()-1; ++i) {
                    nodes.push_back(mesh.e[i][boundarySegments[seg]]);
                }
                // Transform nodes to udof using std::transform and a lambda function
                int nodeSize = nodes.size();
                std::vector<int> wdof(nodeSize);
                for (size_t i = 0; i < nodeSize; ++i) {
                    wdof[i]=(3*nodes[i]+2);
                }
                std::vector<double> fBoundaryElem = integrateOverBoundary(geomEdge, boundarySegments[seg], wt_GQ, N2DCell, gradN2DCell, 2);

                for (int i = 0; i < wdof.size(); ++i) {
                    fBoundary[wdof[i]] += fBoundaryElem[i];
                }

                std::set<int> uniqueSet(nodes.begin(), nodes.end());
                for (const auto& elem : uniqueSet) {
                    myForcedNodes.push_back(elem);
                }
            }

        }

    }

    
    Eigen::VectorXd fBoundaryVec(fBoundary.size());
    for (size_t i = 0; i < fBoundary.size(); ++i) {
        fBoundaryVec(i) = fBoundary[i];
    }

    myNeumanForce += fBoundaryVec;

}

// Function to convert Eigen::MatrixXd to std::vector<std::vector<double>>
std::vector<std::vector<double>> NLElasticity::eigenMatrixToStdVector(const Eigen::MatrixXd& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = matrix(i, j);
        }
    }

    return result;
}

std::vector<std::vector<std::vector<std::vector<double>>>> NLElasticity::computeElasticityTensorGeneralizedNeoHookean(
        const std::string& materialModel, int NCoords, double K1, double mu1, const std::vector<std::vector<double>>& B, double J) {
 
        std::vector<std::vector<std::vector<std::vector<double>>>> C(NCoords, std::vector<std::vector<std::vector<double>>>(NCoords, std::vector<std::vector<double>>(NCoords, std::vector<double>(NCoords, 0.0))));
 
        std::vector<std::vector<double>> delta(NCoords, std::vector<double>(NCoords, 0.0));
        for (int i = 0; i < NCoords; ++i) {
            delta[i][i] = 1.0;
        }
 
        double Bqq = (NCoords == 2) ? B[0][0] + B[1][1] + 1.0 : trace(B);
        if (materialModel == "GeneralizedNeoHookean1") {
            for (int i = 0; i < NCoords; ++i) {
                for (int j = 0; j < NCoords; ++j) {
                    for (int k = 0; k < NCoords; ++k) {
                        for (int l = 0; l < NCoords; ++l) {
                            C[i][j][k][l] = mu1 * (delta[i][k] * B[j][l] + B[i][l] * delta[j][k]
                                - (2.0 / 3.0) * (B[i][j] * delta[k][l] + delta[i][j] * B[k][l])
                                + (2.0 / 3.0) * Bqq * delta[i][j] * delta[k][l] / 3) / pow(J, 2.0 / 3.0)
                                + K1 * (2 * J - 1) * J * delta[i][j] * delta[k][l];
                        }
                    }
                }
            }
        } else if (materialModel == "GeneralizedNeoHookean2") {
            for (int i = 0; i < NCoords; ++i) {
                for (int j = 0; j < NCoords; ++j) {
                    for (int k = 0; k < NCoords; ++k) {
                        for (int l = 0; l < NCoords; ++l) {
                            C[i][j][k][l] = mu1 * (delta[i][k] * B[j][l] + B[i][l] * delta[j][k]
                                - (2.0 / 3.0) * (B[i][j] * delta[k][l] + delta[i][j] * B[k][l])
                                + (2.0 / 3.0) * Bqq * delta[i][j] * delta[k][l] / 3) / pow(J, 2.0 / 3.0)
                                + K1 * J * J * delta[i][j] * delta[k][l];
                        }
                    }
                }
            }
        }
 
        return C;
    }
 
    double NLElasticity::trace(const std::vector<std::vector<double>>& matrix) {
        double result = 0.0;
        for (int i = 0; i < matrix.size(); ++i) {
            result += matrix[i][i];
        }
        return result;
    }

Eigen::MatrixXd NLElasticity::KirchhoffStress(const std::string& materialModel, int NCoords, double K1, double mu1, const Eigen::MatrixXd& B, double J) {
    Eigen::MatrixXd stress = Eigen::MatrixXd::Zero(NCoords, NCoords);
    Eigen::MatrixXd delta = Eigen::MatrixXd::Identity(3, 3);

    double Bkk = B.trace();
    if (NCoords == 2) {
        Bkk = Bkk + 1;
    }

    if (materialModel == "GeneralizedNeoHookean1") {
        for (int i = 0; i < NCoords; ++i) {
            for (int j = 0; j < NCoords; ++j) {
                stress(i, j) = mu1 * (B(i, j) - Bkk * delta(i, j) / 3.) / std::pow(J, 2. / 3.) + K1 * J * (J - 1) * delta(i, j);
            }
        }
    } else if (materialModel == "GeneralizedNeoHookean2") {
        for (int i = 0; i < NCoords; ++i) {
            for (int j = 0; j < NCoords; ++j) {
                stress(i, j) = mu1 * (B(i, j) - Bkk * delta(i, j) / 3.) / std::pow(J, 2. / 3.) + 0.5 * K1 * J * (J - 1 / J) * delta(i, j);
            }
        }
    }

    return stress;
}


void NLElasticity::computeElementStiffnessFiniteStrainSpatialConf(const int elem, Eigen::MatrixXd& KElem, Eigen::VectorXd& fElem) {
   
    // Access elemNodes in C++
    // Get the number of rows in the matrix
    size_t numRows = mesh.q.size();

    // Store the elements from the first column
    std::vector<int> elemNodes;

    for (size_t i = 0; i < numRows; ++i) {
        // Check if the row has at least one element
        if (!mesh.q[i].empty()) {
            // Store the first element of each row (first column)
            elemNodes.push_back(mesh.q[i][elem]);
        }
    }

    // Access other parameters
    int nodes = myNodesPerElement;
    int NCoords = myNCoord;

    // Access gradNCell, xi_GQ, wt_GQ
    std::vector<Eigen::MatrixXd> gradNCell;

    for (const auto& gradN : myGradN) {
        // Convert to Eigen::MatrixXd
    
        gradNCell.push_back(gradN);
    }

    Eigen::MatrixXd Kmaterial = Eigen::MatrixXd::Zero(myDOFPerElem, myDOFPerElem);
    Eigen::MatrixXd Kgeometric = Eigen::MatrixXd::Zero(myDOFPerElem, myDOFPerElem);
    fElem = Eigen::VectorXd::Zero(myDOFPerElem);
    int numGQ = 2;

    std::vector<std::vector<double>> xi_GQ;
    std::vector<double> wt_GQ;

    // Assuming obj is an instance of a class with myXi and myWt member variables
    xi_GQ = myXi;
    wt_GQ = myWt;

    numGQ = wt_GQ.size();
    // Replace 'zeros' with Eigen matrix initialization
    KElem = Eigen::MatrixXd::Zero(myDOFPerElem, myDOFPerElem);
     // Replace 'sol' initialization with Eigen vector
    Eigen::MatrixXd sol;

    if (NCoords == 2) {
        sol = mySol.segment(2 * elemNodes[0] - 1, 2);
    } 
    else 
    {
        Eigen::MatrixXd solMat(8, 3);
      
        // Fill the Eigen::MatrixXd sol
        for (int i = 0; i < 8; ++i) {
            int index = 3 * elemNodes[i];
            solMat(i, 0) = mySol[index];
            solMat(i, 1) = mySol[index + 1];
            solMat(i, 2) = mySol[index + 2];
        }
        sol = solMat;
    }

    // Replace 'disp' with Eigen matrix initialization
    Eigen::MatrixXd gradNdxs = Eigen::MatrixXd::Zero(NCoords, nodes);

   for (int g = 0; g < numGQ; ++g) {
        Eigen::MatrixXd gradNAll = gradNCell[g]; // shape function gradient values at current quadrature point
        
        // Get the g-th column from xi_GQ
        size_t numRows = xi_GQ.size();

        // Store the elements from the first column
        std::vector<double> xi_GQColVec;

        for (size_t i = 0; i < numRows; ++i) {
            // Check if the row has at least one element
            if (!xi_GQ[i].empty()) {
                // Store the first element of each row (first column)
                xi_GQColVec.push_back(xi_GQ[i][g]);
            }
        }
        // Convert Eigen column vector to std::vector<double>
          std::vector<std::vector<double>> JacobianResult = Jacobian(elem, xi_GQColVec);

        // Convert std::vector<std::vector<double>> to Eigen::MatrixXd
        Eigen::MatrixXd JTotal(JacobianResult.size(), JacobianResult[0].size());
        for (int i = 0; i < JacobianResult.size(); ++i) {
            for (int j = 0; j < JacobianResult[0].size(); ++j) {
                JTotal(i, j) = JacobianResult[i][j];
            }
        }

        // compute Jacobian Matrix
        Eigen::MatrixXd gradNdx = JTotal.fullPivLu().solve(gradNAll);
        //sol should be (3X8)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(NCoords, NCoords) + sol.transpose() * gradNdx.transpose();
        Eigen::MatrixXd b = F * F.transpose();
        Eigen::MatrixXd Finv = F.inverse();
        Eigen::MatrixXd gradNdxs = Eigen::MatrixXd::Zero(NCoords, nodes);
    
        // Calculate gradNdxs using Eigen
        for (int k = 0; k < nodes; ++k) {
            for (int i = 0; i < NCoords; ++i) {
                gradNdxs(i, k) = 0;
                for (int j = 0; j < NCoords; ++j) {
                    gradNdxs(i, k) += gradNdx(j, k) * Finv(j, i);
                }
            }
        }

        double JF = F.determinant();
        if (JF < 0) {
            std::cerr << "Determinant of F negative" << std::endl;
            break;
        }

        double K1 = myBulkModulus;
        double mu1 = myMu;
        const std::string materialModel = myMaterialModel;
        Eigen::MatrixXd stress = KirchhoffStress(materialModel, NCoords, K1, mu1, b, JF);

        std::vector<std::vector<double>> bVector = eigenMatrixToStdVector(b);
        std::vector<std::vector<std::vector<std::vector<double>>>> C = computeElasticityTensorGeneralizedNeoHookean(materialModel, NCoords, K1, mu1, bVector, JF);
        
        double dJ = std::abs(JTotal.determinant()); // determinant of Jacobian

        // Update the loop for Kmaterial computation
        for (int A = 0; A < nodes; ++A) {
            for (int i = 0; i < NCoords; ++i) {
                for (int B = 0; B < nodes; ++B) {
                    for (int k = 0; k < NCoords; ++k) {
                        for (int j = 0; j < NCoords; ++j) {
                            for (int l = 0; l < NCoords; ++l) {
                                Kmaterial(NCoords * A + i, NCoords * B + k) += wt_GQ[g] * dJ * gradNdxs(j, A) * C[i][j][k][l] * gradNdxs(l, B);                            
                            }
                            Kgeometric(NCoords * A + i, NCoords * B + k) += wt_GQ[g] * dJ * gradNdxs(k, A) * gradNdxs(j, B) * stress(i, j);
                        }
                    }
                }
            }
        }
        for (int A = 0; A < nodes; ++A) {
            for (int i = 0; i < NCoords; ++i) {
                for (int J = 0; J < NCoords; ++J) {
                    fElem(NCoords * A + i) += wt_GQ[g] * dJ * stress(i, J) * gradNdxs(J, A);
                }
            }
        }
    }

    KElem = Kmaterial + Kgeometric;


}

void NLElasticity::assembleK() {
    int nElements = myNumElems;

    // Initialize Sparse Matrix
    Eigen::SparseMatrix<double> K(myNumDOF, myNumDOF);
    std::vector<Eigen::Triplet<double>> triplets;

    Eigen::VectorXd f = Eigen::VectorXd::Zero(myNumDOF);

    std::vector<std::vector<double>> N;
    std::vector<std::vector<double>> gradN;

    // Define myGradN as a vector of matrices
    std::vector<Eigen::MatrixXd> gradNCell;
     // Get the number of rows
    size_t numRows = myXi.size();

    // Create a vector to store the first column of each row
    std::vector<double> xiRows(numRows);

    // Iterate over columns (assumes all rows have the same number of columns)
    for (size_t i = 0; i < myXi[0].size(); ++i) {
        // Iterate over rows
        for (size_t j = 0; j < numRows; ++j) {
            // Check if the row has at least one element and if the column index is within bounds
                // Store the first element of each row (first column)
                xiRows[j] = myXi[j][i];
        }
    

        auto result = ShapeFunction(xiRows);
        N = result.first;
        gradN = result.second;
        
        Eigen::Map<Eigen::MatrixXd> gradNEigen(gradN.data()->data(), gradN[0].size(), gradN.size());
        Eigen::MatrixXd gradNEigenTransposed = gradNEigen.transpose();

        // Create matrices and push them into the vector
        gradNCell.push_back(gradNEigenTransposed); // First matrix

    }

    myN = N;
    myGradN = gradNCell;
    
    int index = 0;

    for (int elem = 0; elem < nElements; ++elem) {
        size_t nodesSize = mesh.q.size();
        // Create a vector to store the first column of each row
        std::vector<int> nodes(nodesSize);
        // Iterate over rows
        for (size_t j = 0; j < nodesSize; ++j) {            
            nodes[j] = mesh.q[j][elem];
        }
        Eigen::MatrixXd KElem;
        Eigen::VectorXd fElem;
        computeElementStiffnessFiniteStrainSpatialConf(elem, KElem, fElem);
        
        Eigen::VectorXd dof;

        if (myDOFPerNode == 1) {
            dof.resize(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                dof[i] = nodes[i];
            }
        } else if (myDOFPerNode == 3) {
            size_t numRows = nodes.size();
            dof.resize(3 * numRows);

            for (size_t i = 0; i < numRows; ++i) {
                dof[3 * i]     = 3 * nodes[i];
                dof[3 * i + 1] = 3 * nodes[i]+1;
                dof[3 * i + 2] = 3 * nodes[i]+2;
            }
        } 
        else {
            // Handle other cases if needed
            std::cerr << "Unsupported value for myDOFPerNode: " << myDOFPerNode << std::endl;
            // You may want to throw an exception or handle it differently based on your needs.
        }

        for (int i = 0; i < myDOFPerElem; ++i) {
            for (int j = 0; j < myDOFPerElem; ++j) {
                triplets.push_back(Eigen::Triplet<double>(dof(i), dof(j), KElem(i, j)));
            }
        }

        f.segment(dof.minCoeff(), myDOFPerElem) += fElem;
    }


    K.setFromTriplets(triplets.begin(), triplets.end());

    myK = K;
    myF = f;

}

//typedef Eigen::SparseMatrix<double> SparseMatrix;
//typedef Eigen::VectorXd VectorXd;
template <typename data_type>

__global__ void csr_spmv_kernel(
    unsigned int n_rows,
    const int *col_ids,
    const int *row_ptr,
    const data_type *data,
    const data_type *x,
    data_type *y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    printf("row: %d",row);
    if (row < n_rows) {
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row + 1];
        data_type sum = 0;
        for (unsigned int element = row_start; element < row_end; element++) {
            sum += data[element] * x[col_ids[element]];
        }
        y[row] = sum;
    }
}


template <class T>
__device__ T warp_reduce (T val)
{
for (int offset = warpSize / 2; offset > 0; offset /= 2)
 val += __shfl_down_sync (FULL_WARP_MASK, val, offset);
 
return val;
}

template <typename data_type>
__global__ void csr_spmv_vector_kernel (
unsigned int n_rows, 
const int *col_ids, 
const int *row_ptr, 
const data_type *data, 
const data_type *x, 
data_type *y)
{
 printf("Inside csr_spmv_vector_kernel:\n");
unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
const int warp_id = thread_id / 32;
const int lane = thread_id % 32;
const int row = warp_id;
data_type sum = 0;
if (row < n_rows)
{
const int row_start = row_ptr[row];
const int row_end = row_ptr[row + 1];

for (unsigned int element = row_start + lane; element < row_end; element += 32)
sum += data[element] * x[col_ids[element]] ;
 printf("data[element]:%d\n",data[element]);

}
sum = warp_reduce (sum);
if (lane == 0 && row < n_rows)
y [row] = sum;
}

//
// Created by egi on 11/3/19.
//


#define NNZ_PER_WG 64u ///< Should be power of two

template <typename data_type>
__global__ void fill_vector (unsigned int n, data_type *vec, data_type value)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    vec[i] = value;
}

__device__ unsigned int prev_power_of_2 (unsigned int n)
{
  while (n & n - 1)
    n = n & n - 1;
  return n;
}

template <typename data_type>
__global__ void csr_adaptive_spmv_kernel (
    const unsigned int n_rows,
    const int *col_ids,
    const int *row_ptr,
    const int *row_blocks,
    const data_type *data,
    const data_type *x,
    data_type *y)
{
 printf("Inside csr_adaptive_spmv_kernel:\n");
  const unsigned int block_row_begin = row_blocks[blockIdx.x];
  const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
  const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin];

  __shared__ data_type cache[NNZ_PER_WG];

  if (block_row_end - block_row_begin > 1)
  {
    /// CSR-Stream case
    const unsigned int i = threadIdx.x;
    const unsigned int block_data_begin = row_ptr[block_row_begin];
    const unsigned int thread_data_begin = block_data_begin + i;

    if (i < nnz)
      cache[i] = data[thread_data_begin] * x[col_ids[thread_data_begin]];
    __syncthreads ();

    const unsigned int threads_for_reduction = prev_power_of_2 (blockDim.x / (block_row_end - block_row_begin));

    if (threads_for_reduction > 1)
      {
        /// Reduce all non zeroes of row by multiple thread
        const unsigned int thread_in_block = i % threads_for_reduction;
        const unsigned int local_row = block_row_begin + i / threads_for_reduction;

        data_type dot = 0.0;

        if (local_row < block_row_end)
          {
            const unsigned int local_first_element = row_ptr[local_row] - row_ptr[block_row_begin];
            const unsigned int local_last_element = row_ptr[local_row + 1] - row_ptr[block_row_begin];

            for (unsigned int local_element = local_first_element + thread_in_block;
                 local_element < local_last_element;
                 local_element += threads_for_reduction)
              {
                dot += cache[local_element];
              }
          }
        __syncthreads ();
        cache[i] = dot;

        /// Now each row has threads_for_reduction values in cache
        for (int j = threads_for_reduction / 2; j > 0; j /= 2)
          {
            /// Reduce for each row
            __syncthreads ();

            const bool use_result = thread_in_block < j && i + j < NNZ_PER_WG;

            if (use_result)
              dot += cache[i + j];
            __syncthreads ();

            if (use_result)
              cache[i] = dot;
          }

        if (thread_in_block == 0 && local_row < block_row_end)
          y[local_row] = dot;
      }
    else
      {
        /// Reduce all non zeroes of row by single thread
        unsigned int local_row = block_row_begin + i;
        while (local_row < block_row_end)
          {
            data_type dot = 0.0;

            for (unsigned int j = row_ptr[local_row] - block_data_begin;
                 j < row_ptr[local_row + 1] - block_data_begin;
                 j++)
              {
                dot += cache[j];
              }

            y[local_row] = dot;
            local_row += NNZ_PER_WG;
          }
      }
  }
  else
  {
    const unsigned int row = block_row_begin;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;

    data_type dot = 0;

    if (nnz <= 64 || NNZ_PER_WG <= 32)
    {
      /// CSR-Vector case
      if (row < n_rows)
      {
        const unsigned int row_start = row_ptr[row];
        const unsigned int row_end = row_ptr[row + 1];

        for (unsigned int element = row_start + lane; element < row_end; element += 32)
          dot += data[element] * x[col_ids[element]];
      }

      dot = warp_reduce (dot);

      if (lane == 0 && warp_id == 0 && row < n_rows)
      {
        y[row] = dot;
      }
    }
    else
    {
      /// CSR-VectorL case
      if (row < n_rows)
      {
        const unsigned int row_start = row_ptr[row];
        const unsigned int row_end = row_ptr[row + 1];

        for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
          dot += data[element] * x[col_ids[element]];
      }

      dot = warp_reduce (dot);

      if (lane == 0)
        cache[warp_id] = dot;
      __syncthreads ();

      if (warp_id == 0)
      {
        dot = 0.0;

        for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
          dot += cache[element];

        dot = warp_reduce (dot);

        if (lane == 0 && row < n_rows)
        {
          y[row] = dot;
        }
      }
    }
  }
}

unsigned int
fill_row_blocks (
    bool fill,
    int rows_count,
    const int *row_ptr,
    int *row_blocks
)
{
  if (fill)
    row_blocks[0] = 0;

  int last_i = 0;
  int current_wg = 1;
  unsigned int nnz_sum = 0;
  for (int i = 1; i <= rows_count; i++)
  {
    nnz_sum += row_ptr[i] - row_ptr[i - 1];

    if (nnz_sum == NNZ_PER_WG)
    {
      last_i = i;

      if (fill)
        row_blocks[current_wg] = i;
      current_wg++;
      nnz_sum = 0;
    }
    else if (nnz_sum > NNZ_PER_WG)
    {
      if (i - last_i > 1)
      {
        if (fill)
          row_blocks[current_wg] = i - 1;
        current_wg++;
        i--;
      }
      else
      {
        if (fill)
          row_blocks[current_wg] = i;
        current_wg++;
      }

      last_i = i;
      nnz_sum = 0;
    }
    else if (i - last_i > NNZ_PER_WG)
    {
      last_i = i;
      if (fill)
        row_blocks[current_wg] = i;
      current_wg++;
      nnz_sum = 0;
    }
  }

  if (fill)
    row_blocks[current_wg] = rows_count;

  return current_wg;
}

Eigen::VectorXd NLElasticity::conjugateGradientSolver(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &B) {
    double TOLERANCE = 1.0e-10;

    int n = A.rows();
    Eigen::VectorXd X = Eigen::VectorXd::Zero(n);

    Eigen::VectorXd R = B;
    Eigen::VectorXd P = R;
    int k = 0;

    while (k < n) {
        Eigen::VectorXd Rold = R;  // Store previous residual
        Eigen::VectorXd AP = A * P;  // Matrix-vector multiplication

        int n_rows = A.rows();
        const int *col_ids = A.innerIndexPtr();
        const int *row_ptr = A.outerIndexPtr();
        const double *data = A.valuePtr();
        const double *x = B.data();
        
        // Allocate memory on the device for y
        double *y;
        cudaMalloc(&y, n_rows * sizeof(double));
 
        // Call the kernel (assuming grid and block dimensions are already defined)
        unsigned int threads_per_block = 512;
        dim3 threadsPerBlock(4);
        unsigned int numBlocks = (n_rows + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
        dim3 blocks(3);
        // std::cout << "threadsPerBlock: " << threadsPerBlock.x << std::endl;
        //std::cout << "blocks: " << blocks.x << std::endl;


        // fill delimiters
        const int blocks_count = fill_row_blocks (false, n_rows, row_ptr, nullptr);
        //std::unique_ptr< int[]> row_blocks(new int[blocks_count + 1]);
        int* row_blocks = new int[blocks_count + 1];
        fill_row_blocks (true,n_rows, row_ptr, row_blocks);
        
        int *d_row_blocks {};
        cudaMalloc (&d_row_blocks, (blocks_count + 1) * sizeof (unsigned int));
        cudaMemcpy (d_row_blocks, row_blocks, sizeof (unsigned int) * (blocks_count + 1), cudaMemcpyHostToDevice);

        //csr_spmv_kernel<<<blocks, threadsPerBlock>>>(n_rows, col_ids, row_ptr, data, x, y);
        csr_adaptive_spmv_kernel<<<blocks, threadsPerBlock>>>(n_rows, col_ids, row_ptr, d_row_blocks, data, x, y);
        //cudaDeviceSynchronize();
        
        // Assuming n_rows is the number of rows in your matrix A
        // and y is the pointer to your result in device memory
        
        // Allocate host memory for the result
        double *host_y = new double[n_rows];
        
        // Copy result from device to host
        cudaMemcpy(host_y, y, n_rows * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Initialize Eigen::VectorXd with the data from host_y
        AP = Eigen::Map<Eigen::VectorXd>(host_y, n_rows);
    
        // Now AP contains the results of your kernel computation
        
        // Clean up
        delete[] row_blocks;
        delete[] host_y; // Free host memory
        cudaFree(y);     // Free device memory

        double alpha = R.dot(R) / std::max(P.dot(AP), TOLERANCE);
        X = X + alpha * P;  // Next estimate of solution
        R = R - alpha * AP;  // Residual

        if (R.norm() < TOLERANCE)
            break;  // Convergence test

        double beta = R.dot(R) / std::max(Rold.dot(Rold), TOLERANCE);
        P = R + beta * P;  // Next gradient
        k++;
    }

    return X;
}

// Eigen::VectorXd NLElasticity::conjugateGradientSolver(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &B) {

//     double TOLERANCE = 1.0e-10;
 
//     int n = A.rows();

//     Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
 
//     Eigen::VectorXd R = B;

//     Eigen::VectorXd P = R;

//     int k = 0;
 
//     while (k < n) {

//         Eigen::VectorXd Rold = R;  // Store previous residual

//           // Timing using CUDA events
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//         Eigen::VectorXd AP = A * P;  // Matrix-vector multiplication
 
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     if (k==0)
//     {
//         std::cout << "Time for  A * P;Matrix-vector multiplication " << milliseconds << std::endl;
//     }
//         double alpha = R.dot(R) / std::max(P.dot(AP), TOLERANCE);

//         X = X + alpha * P;  // Next estimate of solution

//         R = R - alpha * AP;  // Residual
 
//         if (R.norm() < TOLERANCE)

//             break;  // Convergence test
 
//         double beta = R.dot(R) / std::max(Rold.dot(Rold), TOLERANCE);

//         P = R + beta * P;  // Next gradient

//         k++;

//     }
 
//     return X;

// }
 

void NLElasticity::solveNonlinearFEM() {
    bool verbose = myVerbose;
    int nSteps = myNonlinearSteps;
    int maxIter = 1;
    double tol = 1e-9;

    assembleBC();

    mySol = Eigen::VectorXd::Zero(myNumDOF);
    double err = 1.0;
    Eigen::VectorXd b;
     Eigen::VectorXd dSol;
     Eigen::VectorXd dSolSp;

    int NCoord = myNCoord;
    double loadFactor;
    int iter = 0;
    //typedef Eigen::SparseMatrix<double> SpMatmyK; 
    Eigen::SparseMatrix<double> SpmyK(myNumDOF, myNumDOF);

    for (int step = 1; step <= nSteps; ++step) {
        if (verbose) {
            std::cout << "Load Step " << step << "/" << nSteps << std::endl;
        }

        loadFactor = static_cast<double>(step) / nSteps;
        iter = 0;
        err = 1.0;
        //while (iter < maxIter && err > tol) {
        while (iter < maxIter) {
            iter++;
            std::cout << "iter: "<< iter << std::endl;
            assembleK();
            
             // Get all degrees of freedom
            Eigen::VectorXd allDOF = Eigen::VectorXd::LinSpaced(myNumDOF, 0, myNumDOF-1);
            // Convert Eigen vector to std::vector<int>
            std::vector<int> allDOF_std(allDOF.data(), allDOF.data() + allDOF.size());
  
            // Initialize DeltaSol and FreeDOF
            myDeltaSol.setZero();
            myFreeDOF = allDOF_std;

            for (int i = 0; i < myFixedDOF.size(); ++i) {
                myFreeDOF.erase(std::remove(myFreeDOF.begin(), myFreeDOF.end(), myFixedDOF[i]), myFreeDOF.end());
            }

            
            
            b.setZero();
            b = myF - loadFactor * myNeumanForce;

            // Fix constrained nodes.
            for (int i = 0; i < myFixedDOF.size(); ++i) {
                int rw = myFixedDOF[i];
                myK.row(rw).setZero();
                myK(rw, rw) = 1;
                b(rw) = -loadFactor * myNodalDirichlet[rw] + mySol(rw);
            }
            
            // Convert the dense matrix to a sparse matrix
            std::vector<Eigen::Triplet<double>> triplets;

            // Iterate through the dense matrix and extract non-zero entries into triplets
            for (int i = 0; i < myK.rows(); ++i) {
                for (int j = 0; j < myK.cols(); ++j) {
                    if (myK(i, j) != 0.0) {
                        triplets.push_back(Eigen::Triplet<double>(i, j, myK(i, j)));
                    }
                }
            }

            // Set the sparse matrix from triplets
            SpmyK.setFromTriplets(triplets.begin(), triplets.end());
            Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;
            // Compute the ordering permutation vector from the structural pattern of A
            solver.analyzePattern(SpmyK); 
            // Compute the numerical factorization 
            solver.factorize(SpmyK); 
            //Use the factors to solve the linear system 
            //dSolSp = solver.solve(-b); 

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            dSolSp = conjugateGradientSolver(SpmyK, -b);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Time for conjugateGradientSolver "<< milliseconds << std::endl;


            // Solving:
            //dSol = myK.inverse() * (-b);
            
            myDeltaSol = dSolSp.head(myNumDOF);
            mySol += myDeltaSol;

            for (int i = 0; i < myNumDOF/3; ++i) {
                myU(i) = mySol(3*i);
                myV(i) = mySol((3*i)+1);
                myW(i) = mySol((3*i)+2);
            }
            std::cout << "mySolnorm" << mySol.norm() << std::endl;
            std::cout << "myDeltaSol.norm()" << myDeltaSol.norm() << std::endl;

            err = myDeltaSol.norm() / mySol.norm();

            if (verbose) {
                std::cout << iter << "\t" << err << "\t" << b.norm() / (NCoord * myNumNodes) << std::endl;
            }
        }
    }

    if (err < tol) {
        std::cout << "Convergence achieved." << std::endl;
    } else {
        std::cout << "Convergence not achieved within maximum iterations." << std::endl;
    }

    myDeformation = (myU.array().square() + myV.array().square() + myW.array().square()).sqrt();
    myMaxDelta = myDeformation.maxCoeff();
}

double NLElasticity::getConditionNumber(const Eigen::MatrixXd& matrix) {
    // Calculate singular value decomposition
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);

    // Get singular values
    Eigen::VectorXd singularValues = svd.singularValues();

    // Compute the condition number
    double conditionNumber = singularValues.maxCoeff() / singularValues.minCoeff();

    return conditionNumber;
}

void NLElasticity::solveFEMProblem() {
    if (myApplyFiniteStrain) {
        solveNonlinearFEM();
    } 
    // else {
    //     assembleKMF();
    //     assembleBC();
    //     solveLinearSystem();
    // }
}
