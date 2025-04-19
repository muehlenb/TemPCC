#pragma once

#include "pdflow_cudalib.h"
#include "src/util/OrganizedPointCloud.h"

class PDFlow {
public:

    unsigned int ctf_levels;//Number of levels used in the coarse-to-fine scheme (always dividing by two)
    unsigned int num_max_iter[6];  //Number of iterations at every pyramid level (primal-dual solver)
    float g_mask[25];

    unsigned int integratedPointClouds = 0;

    //Motion field
    float4* D;

    //Max resolution of the coarse-to-fine scheme.
    unsigned int rows;
    unsigned int cols;

    //Optimization Parameters
    float mu, lambda_i, lambda_d;

    //Cuda
    CSF_cuda csf_host, *csf_device;



    PDFlow(unsigned int pCols, unsigned int pRows){
        rows = pRows;
        cols = pCols;
        ctf_levels = static_cast<unsigned int>(log2(float(rows/18))) + 1;

        //Iterations of the primal-dual solver at each pyramid level.
        //Maximum value set to 100 at the finest level
        for (int i=5; i>=0; i--)
        {
            if (i >= int(ctf_levels - 1))
                num_max_iter[i] = 100;
            else
                num_max_iter[i] = num_max_iter[i+1]-15; //Replaced 15 by 18, is this depending on img height??
        }

        //Compute gaussian mask
        int v_mask[5] = {1,4,6,4,1};
        for (unsigned int i=0; i<5; i++)
            for (unsigned int j=0; j<5; j++)
                g_mask[i+5*j] = float(v_mask[i]*v_mask[j])/256.f;


        //Reserve memory for the scene flow estimate (the finest)
        cudaMalloc(&D, sizeof(float4)*rows*cols);

        //Parameters of the variational method
        lambda_i = 0.04f;
        lambda_d = 0.35f;
        mu = 75.f;
    }

    ~PDFlow(){
        freeGPUMemory();
    }


    //Methods
    bool integratePointCloud(std::shared_ptr<OrganizedPointCloud> pc){
        createImagePyramidGPU(pc);
        ++integratedPointClouds;

        if(integratedPointClouds > 1){
            solveSceneFlowGPU(pc);
        }

        return 1;
    }


    void createImagePyramidGPU(std::shared_ptr<OrganizedPointCloud> pc){
        //Copy new frames to the scene flow object
        csf_host.copyNewFrames(pc->gpuColors, pc->gpuPositions, pc->gpuLookupImageTo3D);

        //Copy scene flow object to device
        csf_device = ObjectToDevice(&csf_host);

        unsigned int pyr_levels = static_cast<unsigned int>(log2(float(pc->width/cols))) + ctf_levels;
        GaussianPyramidBridge(csf_device, pyr_levels);

        //Copy scene flow object back to host
        BridgeBack(&csf_host, csf_device);
    }

    void solveSceneFlowGPU(std::shared_ptr<OrganizedPointCloud> pc)
    {
        unsigned int s;
        unsigned int cols_i, rows_i;
        unsigned int level_image;
        unsigned int num_iter;

        //For every level (coarse-to-fine)
        for (unsigned int i=0; i<ctf_levels; i++)
        {
            s = static_cast<unsigned int>(pow(2.f,int(ctf_levels-(i+1))));
            cols_i = cols/s;
            rows_i = rows/s;
            level_image = ctf_levels - i + static_cast<unsigned int>(log2(float(pc->width/cols))) - 1;

            //=========================================================================
            //                              Cuda - Begin
            //=========================================================================

            //Cuda allocate memory
            csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

            //Cuda copy object to device
            csf_device = ObjectToDevice(&csf_host);

            //Assign zeros to the corresponding variables
            AssignZerosBridge(csf_device);

            //Upsample previous solution
            if (i>0)
                UpsampleBridge(csf_device);

            //Compute connectivity (Rij)
            RijBridge(csf_device);

            //Compute colour and depth derivatives
            ImageGradientsBridge(csf_device);
            WarpingBridge(csf_device);

            //Compute mu_uv and step sizes for the primal-dual algorithm
            MuAndStepSizesBridge(csf_device);

            //Primal-Dual solver
            for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
            {
                GradientBridge(csf_device);
                DualVariablesBridge(csf_device);
                DivergenceBridge(csf_device);
                PrimalVariablesBridge(csf_device);
            }

            //Filter solution
            FilterBridge(csf_device);

            //Compute the motion field
            MotionFieldBridge(csf_device);

            //BridgeBack to host
            BridgeBack(&csf_host, csf_device);

            //Free memory of variables associated to this level
            csf_host.freeLevelVariables();

            //Copy motion field to CPU
            //csf_host.copyMotionField(D);

            D = csf_host.D_dev;

            //For debugging
            //DebugBridge(csf_device);

            //=========================================================================
            //                              Cuda - end
            //=========================================================================
        }
    }


    void freeGPUMemory(){
        csf_host.freeDeviceMemory();
    }

    void initializeCUDA(){
        //Read parameters
        csf_host.readParameters(rows, cols, 576, 640, lambda_i, lambda_d, mu, g_mask, ctf_levels);

        //Allocate memory
        csf_host.allocateDevMemory();
    }
};
