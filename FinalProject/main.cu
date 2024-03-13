#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "NonLinearElasticity.cuh"
#include <utility>
#include <functional>

//#include <Eigen/Dense>

int main() {
    // Simulation parameters
 int nEYLoop = 3;
    int nElemY, nElemZ, nElemX;
    double Xlength = 10.0, Ylength = 2.0, Zlength = 2.0;
    int applyFiniteStrain = 1, nSteps = 1, verbose = 1, solMethod = 0;
    double E = 3e7, nu = 0.3, K = E / (3 * (1 - 2 * nu)), mu = E / (2 * (1 + nu));
    std::string materialModel = "GeneralizedNeoHookean1";
    // Display parameters (optional)
    if (verbose) {
        std::cout << "applyFiniteStrain: " << applyFiniteStrain << std::endl;
        std::cout << "nSteps: " << nSteps << std::endl;
        std::cout << "verbose: " << verbose << std::endl;
        std::cout << "solMethod: " << solMethod << std::endl;
        std::cout << "E: " << E << std::endl;
        std::cout << "nu: " << nu << std::endl;
        std::cout << "Xlength: " << Xlength << std::endl;
        std::cout << "Ylength: " << Ylength << std::endl;
        std::cout << "Zlength: " << Zlength << std::endl;
        std::cout << "K: " << K << std::endl;
        std::cout << "mu: " << mu << std::endl;
        std::cout << "materialModel: " << materialModel << std::endl;
        std::cout << "nEYLoop: " << nEYLoop << std::endl;
    }

   

    for (nElemY = nEYLoop; nElemY <= nEYLoop; ++nElemY) {
        nElemZ = nElemY;
        nElemX = 3 * nElemY;

        NLElasticity fem("HexMeshGridGeneral", nElemX, nElemY, nElemZ, Xlength, Ylength, Zlength);
         fem.meshType = "HexMeshGridGeneral";
        fem.myApplyFiniteStrain = applyFiniteStrain;
        fem.myVerbose = verbose;
        fem.myNonlinearSteps = nSteps;
        fem.setMaterialModel(materialModel);
        fem.setYoungsModulus(E);
        fem.setPoissonsRatio(nu);
        fem.setBulkModulus(K);
        fem.setShearModulus(mu);
        fem.applyDirichletOnSurfaceFunction("0", "0", "0", 3);
        fem.applyNeumanOnSurface(0, 0, -8000, 2);

        
            
        cudaEvent_t start, stop;    
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start); 

        fem.solveFEMProblem();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Time for solveFEMProblem in main: " << elapsedTime << " ms" << std::endl;

        int dofTip = 3 * (nElemZ + 1) * (nElemY + 1) * (nElemX + 1);
        double CN_FEM = fem.getConditionNumber(fem.myK(fem.myFreeDOF,fem.myFreeDOF));
        double yplot = fem.mySol(dofTip);

        std::cout << applyFiniteStrain << "\t" << solMethod << "\t" << nElemY << "\t" << fem.myNumDOF
                  << "\t" << yplot << "\t" << CN_FEM << std::endl;
    }

    return 0;

};
