/*****************************************************************************
**				Primal-Dual Scene Flow for RGB-D cameras					**
**				----------------------------------------					**
**																			**
**	Copyright(c) 2015, Mariano Jaimez Tarifa, University of Malaga			**
**	Copyright(c) 2015, Mohamed Souiai, Technical University of Munich		**
**	Copyright(c) 2015, MAPIR group, University of Malaga					**
**	Copyright(c) 2015, Computer Vision group, Tech. University of Munich	**
**																			**
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/

#include "pdflow.h"


//                         Memory allocation - device
//=============================================================================
__host__ void CSF_cuda::allocateDevMemory()
{
    const unsigned int width = 640;
    const unsigned int height = 576;
    unsigned int s;

    //Allocate the unfiltered depth and colour images on GPU
    cudaError_t err = cudaMalloc(&position_wf_dev, width*height*sizeof(float4) );
    cudaError_t err2 = cudaMalloc(&color_wf_dev, width*height*sizeof(uchar4) );

    //Resize pyramid. Allocate memory for the different levels
    const unsigned int pyr_levels = roundf(log2f(width/cols)) + ctf_levels;

    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        s = static_cast<unsigned int>(powf(2,i));

        cudaMalloc((void**)&color_dev[i], width*height*sizeof(float)/(s*s) );
        cudaMalloc((void**)&color_old_dev[i], width*height*sizeof(float)/(s*s) );
        cudaMalloc((void**)&position_dev[i], width*height*sizeof(float4)/(s*s) );
        cudaMalloc((void**)&position_old_dev[i], width*height*sizeof(float4)/(s*s) );
    }

    //Allocate dx, dy, dz on GPU
    cudaMalloc((void**)&D_dev, sizeof(float4) );

    //Allocate final solutions at the biggest resolution only once
    cudaMalloc((void**)&du_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&dv_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&dw_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&pd_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&puu_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&puv_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&pvu_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&pvv_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&pwu_l_dev, width*height*sizeof(float) );
    cudaMalloc((void**)&pwv_l_dev, width*height*sizeof(float) );
}

__host__ void CSF_cuda::allocateMemoryNewLevel(unsigned int rows_loc, unsigned int cols_loc, unsigned int level_i, unsigned int level_image_i)
{
    local_level = level_i;
    level_image = level_image_i;
    rows_i = rows_loc;
    cols_i = cols_loc;

    //Allocate derivatives on GPU
    cudaMalloc((void**)&dct_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dcu_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dcv_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&ddt_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&ddu_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&ddv_dev, rows_i*cols_i*sizeof(float) );

    cudaMalloc((void**)&dcu_aux_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dcv_aux_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&ddu_aux_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&ddv_aux_dev, rows_i*cols_i*sizeof(float) );

    //Allocate gradients on GPU
    cudaMalloc((void**)&gradu1_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&gradu2_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&gradv1_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&gradv2_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&gradw1_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&gradw2_dev, rows_i*cols_i*sizeof(float) );

    //Allocate divergence on GPU
    cudaMalloc((void**)&divpu_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&divpv_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&divpw_dev, rows_i*cols_i*sizeof(float) );

    //Allocate step sizes on GPU
    cudaMalloc((void**)&sigma_pd_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&sigma_puvx_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&sigma_puvy_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&sigma_pwx_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&sigma_pwy_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&tau_u_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&tau_v_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&tau_w_dev, rows_i*cols_i*sizeof(float) );

    //Allocate mu_uv on GPU
    cudaMalloc((void**)&mu_uv_dev, rows_i*cols_i*sizeof(float) );

    //Allocate du_acc, dv_acc, dw_acc on GPU
    cudaMalloc((void**)&du_acc_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dv_acc_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dw_acc_dev, rows_i*cols_i*sizeof(float) );

    //Allocate ri, rj, ri_2, rj_2, du_prev, dv_prev on GPU
    cudaMalloc((void**)&ri_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&rj_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&ri_2_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&rj_2_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&du_prev_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dv_prev_dev, rows_i*cols_i*sizeof(float) );

    //Allocate values of previous level on GPU
    cudaMalloc((void**)&du_new_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dv_new_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dw_new_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pd_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&puu_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&puv_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pvu_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pvv_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pwu_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pwv_dev, rows_i*cols_i*sizeof(float) );

    //Allocate memory for the upsampling variables
    cudaMalloc((void**)&du_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dv_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&dw_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pd_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&puu_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&puv_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pvu_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pvv_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pwu_upsamp_dev, rows_i*cols_i*sizeof(float) );
    cudaMalloc((void**)&pwv_upsamp_dev, rows_i*cols_i*sizeof(float) );

    //Allocate dx, dy, dz on GPU
    cudaFree(D_dev);
    cudaMalloc((void**)&D_dev, rows_i*cols_i*sizeof(float4));
}

//                          Copy object to device
//=============================================================================
CSF_cuda *ObjectToDevice(CSF_cuda *csf_host)
{
    CSF_cuda *csf_device;
    cudaMalloc((void**)&csf_device, sizeof(CSF_cuda) );
    cudaMemcpy(csf_device, csf_host, sizeof(CSF_cuda), cudaMemcpyHostToDevice);
    return csf_device;
}

//                Copy data from host to device and viceversa
//=============================================================================
__host__ void CSF_cuda::readParameters(unsigned int rows_host, unsigned int cols_host, unsigned int height_host, unsigned int width_host,
                                       float lambda_i_host, float lambda_d_host, float mu_host,
                                       float *g_mask, unsigned int levels_host)
{
    height = height_host;
    width = width_host;
    rows = rows_host;
    cols = cols_host;
    lambda_i = lambda_i_host;
    lambda_d = lambda_d_host;
    mu = mu_host;
    ctf_levels = levels_host;

    //Allocate  and copy gaussian mask
    cudaError_t err = cudaMalloc((void**)&g_mask_dev, 5*5*sizeof(float));
    //printf("%s", cudaGetErrorString(err));
    cudaMemcpy(g_mask_dev, g_mask, 5*5*sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void CSF_cuda::copyNewFrames(uchar4 *color_wf, float4 *position_wf, float2* pDF2CS)
{
    const unsigned int width = 640;
    const unsigned int height = 576;

    cudaError_t err = cudaMemcpy(position_wf_dev, position_wf, width*height*sizeof(float4), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaError_t err2 = cudaMemcpy(color_wf_dev, color_wf, width*height*sizeof(uchar4), cudaMemcpyDeviceToDevice);
    if (err2 != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err2));
    }

    DF2CS = pDF2CS;

    //Swap pointers of old and new images of the pyramid (equivalent to pushing the new frames to the old ones)
    for (unsigned int i=0; i<8; i++)
    {
        float* tempColor = color_old_dev[i];
        color_old_dev[i] = color_dev[i];
        color_dev[i] = tempColor;

        float4* tempPos = position_old_dev[i];
        position_old_dev[i] = position_dev[i];
        position_dev[i] = tempPos;
    }
}

/*
__host__ void CSF_cuda::copyAllSolutions(float *dx, float *dy, float *dz, float *depth, float *depth_old, float *colour, float *colour_old, float *xx, float *xx_old, float *yy, float *yy_old)
{
    cudaMemcpy(dx, dx_dev, rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dy, dy_dev, rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dz, dz_dev, rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(depth, depth_dev[level_image], rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(depth_old, depth_old_dev[level_image], rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(colour, colour_dev[level_image], rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(colour_old, colour_old_dev[level_image], rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xx, xx_dev[level_image], rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xx_old, xx_old_dev[level_image], rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(yy, yy_dev[level_image], rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(yy_old, yy_old_dev[level_image], rows_i*cols_i*sizeof(float), cudaMemcpyDeviceToHost);
}*/

__host__ void CSF_cuda::copyMotionField(float4* D)
{
    cudaMemcpy(D, D_dev, rows_i*cols_i*sizeof(float4), cudaMemcpyDeviceToDevice);
}


//                              Free memory - device
//=============================================================================
__host__ void CSF_cuda::freeDeviceMemory()
{
    cudaFree(g_mask_dev);

    const unsigned int width = 640;
    const unsigned int pyr_levels = roundf(log2f(width/cols)) + ctf_levels;
    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        cudaFree(position_old_dev[i]);
        cudaFree(position_dev[i]);
        cudaFree(color_old_dev[i]);
        cudaFree(color_dev[i]);
    }

    //Free pointers to pointers
    cudaFree(position_old_dev);
    cudaFree(position_dev);
    cudaFree(color_old_dev);
    cudaFree(color_dev);

    cudaFree(du_l_dev); cudaFree(dv_l_dev); cudaFree(dw_l_dev);
    cudaFree(pd_l_dev);
    cudaFree(puu_l_dev); cudaFree(puv_l_dev);
    cudaFree(pvu_l_dev); cudaFree(pvv_l_dev);
    cudaFree(pwu_l_dev); cudaFree(pwv_l_dev);

    cudaFree(color_wf_dev);
    cudaFree(position_wf_dev);

    cudaFree(D_dev);
}

__host__ void CSF_cuda::freeLevelVariables()
{
    cudaFree(du_upsamp_dev); cudaFree(dv_upsamp_dev); cudaFree(dw_upsamp_dev);
    cudaFree(pd_upsamp_dev);
    cudaFree(puu_upsamp_dev); cudaFree(puv_upsamp_dev);
    cudaFree(pvu_upsamp_dev); cudaFree(pvv_upsamp_dev);
    cudaFree(pwu_upsamp_dev); cudaFree(pwv_upsamp_dev);

    cudaFree(du_new_dev);
    cudaFree(dv_new_dev);
    cudaFree(dw_new_dev);
    cudaFree(pd_dev);
    cudaFree(puu_dev); cudaFree(puv_dev);
    cudaFree(pvu_dev); cudaFree(pvv_dev);
    cudaFree(pwu_dev); cudaFree(pwv_dev);

    cudaFree(mu_uv_dev);
    cudaFree(ri_dev); cudaFree(rj_dev);
    cudaFree(ri_2_dev); cudaFree(rj_2_dev);
    cudaFree(du_acc_dev); cudaFree(dv_acc_dev); cudaFree(dw_acc_dev);
    cudaFree(du_prev_dev); cudaFree(dv_prev_dev);

    cudaFree(dct_dev); cudaFree(dcu_dev); cudaFree(dcv_dev);
    cudaFree(ddt_dev); cudaFree(ddu_dev); cudaFree(ddv_dev);
    cudaFree(dcu_aux_dev); cudaFree(dcv_aux_dev);
    cudaFree(ddu_aux_dev); cudaFree(ddv_aux_dev);

    cudaFree(gradu1_dev); cudaFree(gradu2_dev);
    cudaFree(gradv1_dev); cudaFree(gradv2_dev);
    cudaFree(gradw1_dev); cudaFree(gradw2_dev);

    cudaFree(divpu_dev); cudaFree(divpv_dev); cudaFree(divpw_dev);

    cudaFree(sigma_pd_dev); cudaFree(sigma_puvx_dev); cudaFree(sigma_puvy_dev);
    cudaFree(sigma_pwx_dev); cudaFree(sigma_pwy_dev);
    cudaFree(tau_u_dev); cudaFree(tau_v_dev); cudaFree(tau_w_dev);
}


//                  Create gaussian pyramid
//=============================================================================
__device__ void CSF_cuda::computePyramidLevel(unsigned int index, unsigned int level)
{
    //Shared memory for the gaussian mask
    __shared__ float mask_shared[25];
    if (threadIdx.x < 25)			//Warning!!!!!! Number of threads should be higher than 25
        mask_shared[threadIdx.x] = g_mask_dev[threadIdx.x];
    __syncthreads();


    const float max_depth_dif = 0.1f;

    //Calculate indices
    const unsigned int v = index%(rows_i);
    const unsigned int u = index/(rows_i);

    // Remark by Andre: Our gpuColors & gpuPositions are in Row-Major order, we need to convert
    // this when accessing the original pixels, since this algorithm works in Column-Major order:

    const unsigned int sY = index%height;
    const unsigned int sX = index/height;

    const unsigned int srcIndex = sX + sY * width;

    if (level == 0)
    {
        //Copy intensity image
        color_dev[level][index] = (0.299f * color_wf_dev[srcIndex].x + 0.587f * color_wf_dev[srcIndex].y + 0.114f * color_wf_dev[srcIndex].z) / 255.f;

        //Copy depth image
        position_dev[level][index] = position_wf_dev[srcIndex];
    }

    //                              Downsampling
    //-----------------------------------------------------------------------------
    else
    {
        float acu_weights_d = 0.f, acu_weights_c = 0.f;

        float3 sumPos = {0.f, 0.f, 0.f};
        float sumColor = 0.f;

        const unsigned int ind_cent_prev = 2*v + 4*u*rows_i;
        const float dcenter = position_dev[level-1][ind_cent_prev].z;

        //Inner pixels
        if ((v>0)&&(v<rows_i-1)&&(u>0)&&(u<cols_i-1))
        {
            for (int k=-2; k<3; k++)
                for (int l=-2; l<3; l++)
                {
                    const unsigned int ind_loop_prev = 2*v+k + 2*(2*u+l)*rows_i;
                    const unsigned int ind_mask = 12+k+5*l;	//2+k+(2+l)*5

                    //Colour
                    sumColor += mask_shared[ind_mask]*color_dev[level-1][ind_loop_prev];

                    //Depth
                    if ((position_dev[level-1][ind_loop_prev].z > 0.f)&&(fabsf(position_dev[level-1][ind_loop_prev].z-dcenter) < max_depth_dif))
                    {
                        const float aux_w = mask_shared[ind_mask]*(max_depth_dif - fabsf(position_dev[level-1][ind_loop_prev].z - dcenter));
                        acu_weights_d += aux_w;
                        sumPos.x += aux_w*position_dev[level-1][ind_loop_prev].x;
                        sumPos.y += aux_w*position_dev[level-1][ind_loop_prev].y;
                        sumPos.z += aux_w*position_dev[level-1][ind_loop_prev].z;
                    }
                }

            if (sumPos.z > 0.f){
                position_dev[level][index].x = sumPos.x/acu_weights_d;
                position_dev[level][index].y = sumPos.y/acu_weights_d;
                position_dev[level][index].z = sumPos.z/acu_weights_d;
                //position_dev[level][index].w = 1;
            } else {
                position_dev[level][index].z = 0.f;
            }

            color_dev[level][index] = sumColor;
        }

        //Boundary
        else
        {
            for (int k=-2; k<3; k++)
                for (int l=-2; l<3; l++)
                {
                    const int indv = 2*v+k, indu = 2*u+l;
                    if ((indv>=0)&&(indv<2*rows_i)&&(indu>=0)&&(indu<2*cols_i))
                    {
                        const unsigned int ind_loop_prev = 2*v+k + 2*(2*u+l)*rows_i;
                        const unsigned int ind_mask = 12+k+5*l;	//2+k+(2+l)*5

                        //Colour
                        sumColor += mask_shared[ind_mask]*color_dev[level-1][ind_loop_prev];
                        acu_weights_c += mask_shared[ind_mask];

                        //Depth
                        if ((position_dev[level-1][ind_loop_prev].z > 0.f)&&(fabsf(position_dev[level-1][ind_loop_prev].z-position_dev[level-1][ind_cent_prev].z) < max_depth_dif))
                        {
                            const float aux_w = mask_shared[ind_mask]*(max_depth_dif - fabsf(position_dev[level-1][ind_loop_prev].z - position_dev[level-1][ind_cent_prev].z));
                            acu_weights_d += aux_w;
                            sumPos.x += aux_w*position_dev[level-1][ind_loop_prev].x;
                            sumPos.y += aux_w*position_dev[level-1][ind_loop_prev].y;
                            sumPos.z += aux_w*position_dev[level-1][ind_loop_prev].z;
                        }
                    }
                }

            color_dev[level][index] = sumColor/acu_weights_c;

            if (sumPos.z > 0.f){
                position_dev[level][index].x = sumPos.x/acu_weights_d;
                position_dev[level][index].y = sumPos.y/acu_weights_d;
                position_dev[level][index].z = sumPos.z/acu_weights_d;
            } else {
                position_dev[level][index].x = 0.f;
                position_dev[level][index].y = 0.f;
                position_dev[level][index].z = 0.f;
            }
        }
    }


    // TODO ANDRE: X and Y must probably be calculated for all levels of this gaussian pyramid!?

    /*
    //Calculate coordinates "xy" of the points
    const float inv_f_i = 2.f*tan(0.5f*fovh)/float(cols_i);
    const float disp_u_i = 0.5f*(cols_i-1);
    const float disp_v_i = 0.5f*(rows_i-1);

    xx_dev[level][index] = (u - disp_u_i)*depth_dev[level][index]*inv_f_i;
    yy_dev[level][index] = (v - disp_v_i)*depth_dev[level][index]*inv_f_i;
    */
}


//                  Initiallize some variables
//=============================================================================
__device__ void CSF_cuda::assignZeros(unsigned int index)
{
    du_upsamp_dev[index] = 0.f; dv_upsamp_dev[index] = 0.f; dw_upsamp_dev[index] = 0.f;
    pd_upsamp_dev[index] = 0.f;
    puu_upsamp_dev[index] = 0.f; puv_upsamp_dev[index] = 0.f;
    pvu_upsamp_dev[index] = 0.f; pvv_upsamp_dev[index] = 0.f;
    pwu_upsamp_dev[index] = 0.f; pwv_upsamp_dev[index] = 0.f;

    du_prev_dev[index] = 0.f; dv_prev_dev[index] = 0.f;
    du_new_dev[index] = 0.f; dv_new_dev[index] = 0.f; dw_new_dev[index] = 0.f;
    pd_dev[index] = 0.f;
    puu_dev[index] = 0.f; puv_dev[index] = 0.f;
    pvu_dev[index] = 0.f; pvv_dev[index] = 0.f;
    pwu_dev[index] = 0.f; pwv_dev[index] = 0.f;

    du_acc_dev[index] = 0.f; dv_acc_dev[index] = 0.f; dw_acc_dev[index] = 0.f;
}


//                  Upsample previous solution
//=============================================================================
__device__ void CSF_cuda::upsampleCopyPrevSolution(unsigned int index)
{
    //Calculate (v,u)
    const unsigned int v = index%(rows_i/2);
    const unsigned int u = 2*index/(rows_i);
    const unsigned int index_big = 2*(v + u*rows_i);

    du_upsamp_dev[index_big] = 2.f*du_l_dev[index];
    dv_upsamp_dev[index_big] = 2.f*dv_l_dev[index];
    dw_upsamp_dev[index_big] = dw_l_dev[index];
    pd_upsamp_dev[index_big] = pd_l_dev[index];
    puu_upsamp_dev[index_big] = puu_l_dev[index];
    puv_upsamp_dev[index_big] = puv_l_dev[index];
    pvu_upsamp_dev[index_big] = pvu_l_dev[index];
    pvv_upsamp_dev[index_big] = pvv_l_dev[index];
    pwu_upsamp_dev[index_big] = pwu_l_dev[index];
    pwv_upsamp_dev[index_big] = pwv_l_dev[index];
}

__device__ void CSF_cuda::upsampleFilterPrevSolution(unsigned int index)
{
    const unsigned int v = index%rows_i;
    const unsigned int u = index/rows_i;

    //Shared memory for the gaussian mask - Warning!! The number of threads should be higher than 25
    __shared__ float mask_shared[25];
    if (threadIdx.x < 25)
        mask_shared[threadIdx.x] = 4.f*g_mask_dev[threadIdx.x];
    __syncthreads();

    float du = 0.f, dv = 0.f, dw = 0.f, pd = 0.f, puu = 0.f, puv = 0.f, pvu = 0.f, pvv = 0.f, pwu = 0.f, pwv = 0.f;

    //Inner pixels
    if ((v>1)&&(v<rows_i-2)&&(u>1)&&(u<cols_i-2))
    {
        for (int k=-2; k<3; k++)
            for (int l=-2; l<3; l++)
            {
                const unsigned incr_index = v+k+(u+l)*rows_i;
                const float gmask_factor = mask_shared[12 + k + 5*l];	//[2+k + (2+l)*5]
                du += gmask_factor*du_upsamp_dev[incr_index];
                dv += gmask_factor*dv_upsamp_dev[incr_index];
                dw += gmask_factor*dw_upsamp_dev[incr_index];
                pd  += gmask_factor*pd_upsamp_dev[incr_index];
                puu += gmask_factor*puu_upsamp_dev[incr_index];
                puv += gmask_factor*puv_upsamp_dev[incr_index];
                pvu += gmask_factor*pvu_upsamp_dev[incr_index];
                pvv += gmask_factor*pvv_upsamp_dev[incr_index];
                pwu += gmask_factor*pwu_upsamp_dev[incr_index];
                pwv += gmask_factor*pwv_upsamp_dev[incr_index];
            }
    }
    //Boundary
    else
    {
        float acu_weight = 1.f;
        for (int k=-2; k<3; k++)
            for (int l=-2; l<3; l++)
            {
                const int indv = v+k, indu = u+l;
                if ((indv<0)||(indv>=rows_i)||(indu<0)||(indu>=cols_i))
                {
                    acu_weight -= 0.25f*mask_shared[12 + k + 5*l];		//[2+k + (2+l)*5]
                    continue;
                }
                else
                {
                    const unsigned incr_index = v+k+(u+l)*rows_i;
                    const float gmask_factor = mask_shared[12 + k + 5*l];	//[2+k + (2+l)*5]
                    du += gmask_factor*du_upsamp_dev[incr_index];
                    dv += gmask_factor*dv_upsamp_dev[incr_index];
                    dw += gmask_factor*dw_upsamp_dev[incr_index];
                    pd  += gmask_factor*pd_upsamp_dev[incr_index];
                    puu += gmask_factor*puu_upsamp_dev[incr_index];
                    puv += gmask_factor*puv_upsamp_dev[incr_index];
                    pvu += gmask_factor*pvu_upsamp_dev[incr_index];
                    pvv += gmask_factor*pvv_upsamp_dev[incr_index];
                    pwu += gmask_factor*pwu_upsamp_dev[incr_index];
                    pwv += gmask_factor*pwv_upsamp_dev[incr_index];
                }
            }

        const float inv_acu_weight = fdividef(1.f, acu_weight);
        du *= inv_acu_weight;
        dv *= inv_acu_weight;
        dw *= inv_acu_weight;
        pd  *= inv_acu_weight;
        puu *= inv_acu_weight;
        puv *= inv_acu_weight;
        pvu *= inv_acu_weight;
        pvv *= inv_acu_weight;
        pwu *= inv_acu_weight;
        pwv *= inv_acu_weight;
    }

    //Write results to global memory
    du_prev_dev[index] = du;
    dv_prev_dev[index] = dv;
    dw_new_dev[index]  = dw;
    pd_dev[index]  = pd;
    puu_dev[index] = puu;
    puv_dev[index] = puv;
    pvu_dev[index] = pvu;
    pvv_dev[index] = pvv;
    pwu_dev[index] = pwu;
    pwv_dev[index] = pwv;

    //Last update, for dw_acc
    dw_acc_dev[index] = dw;
}


//                  Compute intensity and depth derivatives
//=============================================================================
__device__ void CSF_cuda::computeImGradients(unsigned int index)
{
    //Calculate (v,u)
    const unsigned int v = index%rows_i;
    const unsigned int u = index/rows_i;

    //Row gradients
    if (u == 0)
    {
        dcu_aux_dev[index] = color_dev[level_image][index+rows_i] - color_dev[level_image][index];
        ddu_aux_dev[index] = position_dev[level_image][index+rows_i].z - position_dev[level_image][index].z;
    }
    else if (u == cols_i-1)
    {
        dcu_aux_dev[index] = color_dev[level_image][index] - color_dev[level_image][index-rows_i];
        ddu_aux_dev[index] = position_dev[level_image][index].z - position_dev[level_image][index-rows_i].z;
    }
    else
    {
        dcu_aux_dev[index] = (ri_2_dev[index]*(color_dev[level_image][index+rows_i]-color_dev[level_image][index])
                            + ri_2_dev[index-rows_i]*(color_dev[level_image][index]-color_dev[level_image][index-rows_i]))
                            /(ri_2_dev[index]+ri_2_dev[index-rows_i]);
        if (position_dev[level_image][index].z > 0.f)
            ddu_aux_dev[index] = (ri_2_dev[index]*(position_dev[level_image][index+rows_i].z-position_dev[level_image][index].z)
                                + ri_2_dev[index-rows_i]*(position_dev[level_image][index].z-position_dev[level_image][index-rows_i].z))
                                /(ri_2_dev[index]+ri_2_dev[index-rows_i]);
        else
            ddu_aux_dev[index] = 0.f;
    }

    //Col gradients
    if (v == 0)
    {
        dcv_aux_dev[index] = color_dev[level_image][index+1] - color_dev[level_image][index];
        ddv_aux_dev[index] = position_dev[level_image][index+1].z - position_dev[level_image][index].z;
    }
    else if (v == rows_i-1)
    {
        dcv_aux_dev[index] = color_dev[level_image][index] - color_dev[level_image][index-1];
        ddv_aux_dev[index] = position_dev[level_image][index].z - position_dev[level_image][index-1].z;
    }
    else
    {
        dcv_aux_dev[index] = (rj_2_dev[index]*(color_dev[level_image][index+1]-color_dev[level_image][index])
                            + rj_2_dev[index-1]*(color_dev[level_image][index]-color_dev[level_image][index-1]))
                            /(rj_2_dev[index]+rj_2_dev[index-1]);
        if (position_dev[level_image][index].z > 0.f)
            ddv_aux_dev[index] = (rj_2_dev[index]*(position_dev[level_image][index+1].z-position_dev[level_image][index].z)
                                + rj_2_dev[index-1]*(position_dev[level_image][index].z-position_dev[level_image][index-1].z))
                                /(rj_2_dev[index]+rj_2_dev[index-1]);
        else
            ddv_aux_dev[index] = 0.f;
    }
}

__device__ void CSF_cuda::performWarping(unsigned int index)
{
    //Calculate (v,u)
    const unsigned int v = index%rows_i;
    const unsigned int u = index/rows_i;
    float warped_pixel;

    //Intensity images
    const float ind_uf = float(u) + du_prev_dev[index];
    const float ind_vf = float(v) + dv_prev_dev[index];
    warped_pixel = interpolatePixel(color_dev[level_image], ind_uf, ind_vf);
    dct_dev[index] = warped_pixel - color_old_dev[level_image][index];
    dcu_dev[index] = interpolatePixel(dcu_aux_dev, ind_uf, ind_vf);
    dcv_dev[index] = interpolatePixel(dcv_aux_dev, ind_uf, ind_vf);

    //Depth images
    warped_pixel = interpolatePixelDepth(position_dev[level_image], ind_uf, ind_vf);
    if (warped_pixel > 0.f)
        ddt_dev[index] = warped_pixel - position_old_dev[level_image][index].z;
    else
        ddt_dev[index] = 0.f;
    ddu_dev[index] = interpolatePixel(ddu_aux_dev, ind_uf, ind_vf);
    ddv_dev[index] = interpolatePixel(ddv_aux_dev, ind_uf, ind_vf);
}

__device__ float CSF_cuda::interpolatePixel(float *mat, float ind_u, float ind_v)
{
    if (ind_u < 0.f) { ind_u = 0.f;}
    else if (ind_u > cols_i - 1.f) { ind_u = cols_i - 1.f;}
    if (ind_v < 0.f) { ind_v = 0.f;}
    else if (ind_v > rows_i - 1.f) { ind_v = rows_i - 1.f;}

    const unsigned int sup_u = __float2int_ru(ind_u);
    const unsigned int inf_u = __float2int_rd(ind_u);
    const unsigned int sup_v = __float2int_ru(ind_v);
    const unsigned int inf_v = __float2int_rd(ind_v);

    if ((sup_u == inf_u)&&(sup_v == inf_v))
        return mat[lrintf(ind_v + rows_i*ind_u)];

    else if (sup_u == inf_u)
        return (sup_v - ind_v)*mat[inf_v + rows_i*lrintf(ind_u)] + (ind_v - inf_v)*mat[sup_v + rows_i*lrintf(ind_u)];

    else if (sup_v == inf_v)
        return (sup_u - ind_u)*mat[lrintf(ind_v) + rows_i*inf_u] + (ind_u - inf_u)*mat[lrintf(ind_v) + rows_i*sup_u];

    else
    {
        //First in u
        const float val_sup_v = (sup_u - ind_u)*mat[sup_v + rows_i*inf_u] + (ind_u - inf_u)*mat[sup_v + rows_i*sup_u];
        const float val_inf_v = (sup_u - ind_u)*mat[inf_v + rows_i*inf_u] + (ind_u - inf_u)*mat[inf_v + rows_i*sup_u];
        return (sup_v - ind_v)*val_inf_v + (ind_v - inf_v)*val_sup_v;
    }
}

__device__ float CSF_cuda::interpolatePixelDepth(float4 *mat, float ind_u, float ind_v)
{
    if (ind_u < 0.f) { ind_u = 0.f;}
    else if (ind_u > cols_i - 1.f) { ind_u = cols_i - 1.f;}
    if (ind_v < 0.f) { ind_v = 0.f;}
    else if (ind_v > rows_i - 1.f) { ind_v = rows_i - 1.f;}

    const unsigned int sup_u = __float2int_ru(ind_u);
    const unsigned int inf_u = __float2int_rd(ind_u);
    const unsigned int sup_v = __float2int_ru(ind_v);
    const unsigned int inf_v = __float2int_rd(ind_v);

    if ((mat[sup_v + rows_i*sup_u].z == 0.f)||(mat[sup_v + rows_i*inf_u].z == 0.f)||(mat[inf_v + rows_i*sup_u].z == 0.f)||(mat[inf_v + rows_i*inf_u].z==0.f))
    {
        const unsigned int rind_u = __float2int_rn(ind_u);
        const unsigned int rind_v = __float2int_rn(ind_v);
        return mat[rind_v + rows_i*rind_u].z;
    }
    else
    {
        if ((sup_u == inf_u)&&(sup_v == inf_v))
            return mat[lrintf(ind_v + rows_i*ind_u)].z;

        else if (sup_u == inf_u)
            return (sup_v - ind_v)*mat[inf_v + rows_i*lroundf(ind_u)].z + (ind_v - inf_v)*mat[sup_v + rows_i*lroundf(ind_u)].z;

        else if (sup_v == inf_v)
            return (sup_u - ind_u)*mat[lroundf(ind_v) + rows_i*inf_u].z + (ind_u - inf_u)*mat[lroundf(ind_v) + rows_i*sup_u].z;

        else
        {
            //First in u
            const float val_sup_v = (sup_u - ind_u)*mat[sup_v + rows_i*inf_u].z + (ind_u - inf_u)*mat[sup_v + rows_i*sup_u].z;
            const float val_inf_v = (sup_u - ind_u)*mat[inf_v + rows_i*inf_u].z + (ind_u - inf_u)*mat[inf_v + rows_i*sup_u].z;
            return (sup_v - ind_v)*val_inf_v + (ind_v - inf_v)*val_sup_v;
        }
    }
}

//                          Preliminary computations
//=============================================================================
__device__ void CSF_cuda::computeRij(unsigned int index)
{
    //Calculate (v,u)
    const unsigned int v = index%rows_i;
    const unsigned int u = index/rows_i;

    float dxu, dzu, dxu_2, dzu_2;
    float dyv, dzv, dyv_2, dzv_2;

    if (u == cols_i-1)
    {
        dxu = 0.f; dzu = 0.f;
        dxu_2 = 0.f; dzu_2 = 0.f;
    }
    else
    {
        dxu = position_old_dev[level_image][index + rows_i].x - position_old_dev[level_image][index].x;
        dzu = position_old_dev[level_image][index + rows_i].z - position_old_dev[level_image][index].z;
        dxu_2 = position_dev[level_image][index + rows_i].x - position_dev[level_image][index].x;
        dzu_2 = position_dev[level_image][index + rows_i].z - position_dev[level_image][index].z;
    }

    if (v == rows_i-1)
    {
        dyv = 0.f; dzv = 0.f;
        dyv_2 = 0.f; dzv_2 = 0.f;
    }
    else
    {
        dyv = position_old_dev[level_image][index+1].y - position_old_dev[level_image][index].y;
        dzv = position_old_dev[level_image][index+1].z - position_old_dev[level_image][index].z;
        dyv_2 = position_dev[level_image][index+1].y - position_dev[level_image][index].y;
        dzv_2 = position_dev[level_image][index+1].z - position_dev[level_image][index].z;
    }

    if (fabsf(dxu) + fabsf(dzu) > 0.f)
        ri_dev[index] = 2.f*rhypotf(dxu,dzu);	//2.f/sqrtf(dxu*dxu + dzu*dzu);
    else
        ri_dev[index] = 1.f;

    if (fabsf(dyv) + fabsf(dzv) > 0.f)
        rj_dev[index] = 2.f*rhypotf(dyv,dzv);	//2.f/sqrtf(dyv*dyv + dzv*dzv);
    else
        rj_dev[index] = 1.f;

    if (fabsf(dxu_2) + fabsf(dzu_2) > 0.f)
        ri_2_dev[index] = 2.f*rhypotf(dxu_2,dzu_2);	//2.f/sqrtf(dxu*dxu + dzu*dzu);
    else
        ri_2_dev[index] = 1.f;

    if (fabsf(dyv_2) + fabsf(dzv_2) > 0.f)
        rj_2_dev[index] = 2.f*rhypotf(dyv_2,dzv_2);	//2.f/sqrtf(dyv*dyv + dzv*dzv);
    else
        rj_2_dev[index] = 1.f;
}

__device__ void CSF_cuda::computeMu(unsigned int index)
{
    mu_uv_dev[index] = fdividef(mu, 1.f + 1000.f*(ddu_dev[index]*ddu_dev[index] + ddv_dev[index]*ddv_dev[index] + ddt_dev[index]*ddt_dev[index]));
}

__device__ void CSF_cuda::computeStepSizes(unsigned int index)
{
    //Load lambda from global memory
    const float lambdai = lambda_i, lambdad = lambda_d;

    sigma_pd_dev[index] = fdividef(1.f, mu_uv_dev[index]*(1.f + abs(ddu_dev[index]) + abs(ddv_dev[index])) + 1e-10f);
    sigma_puvx_dev[index] = fdividef(0.5f, lambdai*ri_dev[index] + 1e-10f);
    sigma_puvy_dev[index] = fdividef(0.5f, lambdai*rj_dev[index] + 1e-10f);
    sigma_pwx_dev[index] = fdividef(0.5f, ri_dev[index]*lambdad + 1e-10f);
    sigma_pwy_dev[index] = fdividef(0.5f, rj_dev[index]*lambdad + 1e-10f);

    //Calculate (v,u)
    const unsigned int v = index%rows_i;
    const unsigned int u = index/rows_i;

    float acu_r = ri_dev[index] + rj_dev[index];
    if (u > 0) acu_r += ri_dev[index-rows_i];
    if (v > 0) acu_r += rj_dev[index-1];

    tau_u_dev[index] = fdividef(1.f, mu_uv_dev[index]*abs(ddu_dev[index]) + lambdai*acu_r + 1e-10f);
    tau_v_dev[index] = fdividef(1.f, mu_uv_dev[index]*abs(ddv_dev[index]) + lambdai*acu_r + 1e-10f);
    tau_w_dev[index] = fdividef(1.f, mu_uv_dev[index] + lambdad*acu_r + 1e-10f);
}


//                              Main iteration
//=============================================================================
__device__ void CSF_cuda::updateDualVariables(unsigned int index)
{
    //Create aux variables to avoid repetitive global memory access
    float module_p;
    float pd = pd_dev[index], puu = puu_dev[index], puv = puv_dev[index];
    float pvu = pvu_dev[index], pvv = pvv_dev[index], pwu = pwu_dev[index], pwv = pwv_dev[index];

    //Update dual variables
    //Solve pd
    pd += sigma_pd_dev[index]*mu_uv_dev[index]*(-dw_acc_dev[index] + ddt_dev[index] + ddu_dev[index]*du_acc_dev[index] + ddv_dev[index]*dv_acc_dev[index]);

    //Solve pu
    puu += sigma_puvx_dev[index]*lambda_i*gradu1_dev[index];
    puv += sigma_puvy_dev[index]*lambda_i*gradu2_dev[index];

    //Solve pv
    pvu += sigma_puvx_dev[index]*lambda_i*gradv1_dev[index];
    pvv += sigma_puvy_dev[index]*lambda_i*gradv2_dev[index];

    //Solve pw
    pwu += sigma_pwx_dev[index]*lambda_d*gradw1_dev[index];
    pwv += sigma_pwy_dev[index]*lambda_d*gradw2_dev[index];

    //Constrain pd
    module_p = fabsf(pd);
    if (module_p > 1.f)
    {
        if (pd > 1.f)
            pd_dev[index] = 1.f;
        else
            pd_dev[index] = -1.f;
    }
    else
        pd_dev[index] = pd;

    //Constrain pu
    module_p = rhypotf(puu, puv);	//1.f/sqrtf(puu*puu + puv*puv);
    if (module_p < 1.f)
    {
        puu_dev[index] = puu*module_p;
        puv_dev[index] = puv*module_p;
    }
    else
    {
        puu_dev[index] = puu;
        puv_dev[index] = puv;
    }

    //Constrain pv
    module_p = rhypotf(pvu, pvv);	//1.f/sqrtf(pvu*pvu + pvv*pvv);
    if (module_p < 1.f)
    {
        pvu_dev[index] = pvu*module_p;
        pvv_dev[index] = pvv*module_p;
    }
    else
    {
        pvu_dev[index] = pvu;
        pvv_dev[index] = pvv;
    }

    //Constrain pw
    module_p = rhypotf(pwu, pwv);	//1.f/sqrt(pwu*pwu + pwv*pwv);
    if (module_p < 1.f)
    {
        pwu_dev[index] = pwu*module_p;
        pwv_dev[index] = pwv*module_p;
    }
    else
    {
        pwu_dev[index] = pwu;
        pwv_dev[index] = pwv;
    }

}

__device__ void CSF_cuda::updatePrimalVariables(unsigned int index)
{
    float du = du_new_dev[index], dv = dv_new_dev[index], dw = dw_new_dev[index];
    const float du_old = du, dv_old = dv, dw_old = dw;

    //Compute du, dv and dw
    //Solve du
    du += - tau_u_dev[index]*(mu_uv_dev[index]*ddu_dev[index]*pd_dev[index] - lambda_i*divpu_dev[index]);

    //Solve dv
    dv += - tau_v_dev[index]*(mu_uv_dev[index]*ddv_dev[index]*pd_dev[index] - lambda_i*divpv_dev[index]);

    //Solve dw
    dw += - tau_w_dev[index]*(-mu_uv_dev[index]*pd_dev[index] - lambda_d*divpw_dev[index]);

    //shrink du, dv and dw
    //-----------------------------------------------------------
    const float optflow = dct_dev[index] + dcu_dev[index]*du + dcv_dev[index]*dv;
    const float of_threshold = tau_u_dev[index]*dcu_dev[index]*dcu_dev[index] + tau_v_dev[index]*dcv_dev[index]*dcv_dev[index];
    if (optflow < -of_threshold)
    {
        du += tau_u_dev[index]*dcu_dev[index];
        dv += tau_v_dev[index]*dcv_dev[index];
    }
    else if (optflow > of_threshold)
    {
        du -= tau_u_dev[index]*dcu_dev[index];
        dv -= tau_v_dev[index]*dcv_dev[index];
    }
    else
    {
        const float den = tau_u_dev[index]*dcu_dev[index]*dcu_dev[index] + tau_v_dev[index]*dcv_dev[index]*dcv_dev[index] + 1e-10f;
        du -= tau_u_dev[index]*dcu_dev[index]*optflow/den;
        dv -= tau_v_dev[index]*dcv_dev[index]*optflow/den;
    }

    //Update du, dv
    du_acc_dev[index] = 2.f*du - du_old;
    dv_acc_dev[index] = 2.f*dv - dv_old;
    dw_acc_dev[index] = 2.f*dw - dw_old;

    du_new_dev[index] = du;
    dv_new_dev[index] = dv;
    dw_new_dev[index] = dw;
}

__device__ void CSF_cuda::computeDivergence(unsigned int index)
{
    //Calculate (v,u)
    const unsigned int v = index%rows_i;
    const unsigned int u = index/rows_i;

    //First terms
    if (u == 0)
    {
        divpu_dev[index] = ri_dev[index]*puu_dev[index];
        divpv_dev[index] = ri_dev[index]*pvu_dev[index];
        divpw_dev[index] = ri_dev[index]*pwu_dev[index];
    }
    else if (u == cols_i-1)
    {
        divpu_dev[index] = -ri_dev[index-rows_i]*puu_dev[index-rows_i];
        divpv_dev[index] = -ri_dev[index-rows_i]*pvu_dev[index-rows_i];
        divpw_dev[index] = -ri_dev[index-rows_i]*pwu_dev[index-rows_i];
    }
    else
    {
        divpu_dev[index] = ri_dev[index]*puu_dev[index] - ri_dev[index-rows_i]*puu_dev[index-rows_i];
        divpv_dev[index] = ri_dev[index]*pvu_dev[index] - ri_dev[index-rows_i]*pvu_dev[index-rows_i];
        divpw_dev[index] = ri_dev[index]*pwu_dev[index] - ri_dev[index-rows_i]*pwu_dev[index-rows_i];
    }

    //Second term
    if (v == 0)
    {
        divpu_dev[index] += rj_dev[index]*puv_dev[index];
        divpv_dev[index] += rj_dev[index]*pvv_dev[index];
        divpw_dev[index] += rj_dev[index]*pwv_dev[index];
    }
    else if (v == rows_i-1)
    {
        divpu_dev[index] += -rj_dev[index-1]*puv_dev[index-1];
        divpv_dev[index] += -rj_dev[index-1]*pvv_dev[index-1];
        divpw_dev[index] += -rj_dev[index-1]*pwv_dev[index-1];
    }
    else
    {
        divpu_dev[index] += rj_dev[index]*puv_dev[index] - rj_dev[index-1]*puv_dev[index-1];
        divpv_dev[index] += rj_dev[index]*pvv_dev[index] - rj_dev[index-1]*pvv_dev[index-1];
        divpw_dev[index] += rj_dev[index]*pwv_dev[index] - rj_dev[index-1]*pwv_dev[index-1];
    }
}

__device__ void CSF_cuda::computeGradient(unsigned int index)
{
    //Calculate (v,u)
    const unsigned int v = index%rows_i;
    const unsigned int u = index/rows_i;

    if (u == cols_i-1)
    {
        gradu1_dev[index] = 0.f;
        gradv1_dev[index] = 0.f;
        gradw1_dev[index] = 0.f;
    }
    else
    {
        gradu1_dev[index] = ri_dev[index]*((du_acc_dev[index+rows_i] + du_prev_dev[index+rows_i]) - (du_acc_dev[index] + du_prev_dev[index]));
        gradv1_dev[index] = ri_dev[index]*((dv_acc_dev[index+rows_i] + dv_prev_dev[index+rows_i]) - (dv_acc_dev[index] + dv_prev_dev[index]));
        gradw1_dev[index] = ri_dev[index]*(dw_acc_dev[index+rows_i] - dw_acc_dev[index]);
    }

    if (v == rows_i-1)
    {
        gradu2_dev[index] = 0.f;
        gradv2_dev[index] = 0.f;
        gradw2_dev[index] = 0.f;
    }
    else
    {
        gradu2_dev[index] = rj_dev[index]*((du_acc_dev[index+1] + du_prev_dev[index+1]) - (du_acc_dev[index] + du_prev_dev[index]));
        gradv2_dev[index] = rj_dev[index]*((dv_acc_dev[index+1] + dv_prev_dev[index+1]) - (dv_acc_dev[index] + dv_prev_dev[index]));
        gradw2_dev[index] = rj_dev[index]*(dw_acc_dev[index+1] - dw_acc_dev[index]);
    }
}


//                              Filter
//=============================================================================
__device__ void CSF_cuda::saturateVariables(unsigned int index)
{
    float du = du_new_dev[index], dv = dv_new_dev[index], dw = dw_new_dev[index];
    if (du > 1.f)
        du = 1.f;
    else if (du < -1.f)
        du = -1.f;

    if (dv > 1.f)
        dv = 1.f;
    else if (dv < -1.f)
        dv = -1.f;

    if (position_old_dev[level_image][index].z == 0.f)
        dw = 0.f;

    //Add previous solution to filter all together
    du_new_dev[index] = du + du_prev_dev[index];
    dv_new_dev[index] = dv + dv_prev_dev[index];
    dw_new_dev[index] = dw;
}

__device__ void CSF_cuda::filterSolution(unsigned int index)
{
    const float depth_old = position_old_dev[level_image][index].z;

    //Calculate (v,u)
    const unsigned int v = index%rows_i;
    const unsigned int u = index/rows_i;

    //								Weighted median filter
    //----------------------------------------------------------------------------------------
    fieldAndPresence up[9], vp[9], wp[9];
    float pres_cum_u[9], pres_cum_v[9], pres_cum_w[9], pres_med;
    int indr, indc, ind_loop;
    unsigned int point_count, v_index;
    const float kd = 5.f;
    const float kddt = 10.f;

    if (depth_old > 0.f)
    {
        point_count = 9;
        v_index = 0;

        for (int k=-1; k<2; k++)
            for (int l=-1; l<2; l++)
            {
                indr = v+k;
                indc = u+l;
                ind_loop = index + l*rows_i + k;
                if ((indr < 0)||(indr >= rows_i)||(indc < 0)||(indc >= cols_i))
                {
                    point_count--;
                    continue;
                }

                //Compute weights
                const float pres = 1.f/(1.f + kd*powf(depth_old - position_old_dev[level_image][ind_loop].z,2.f) + kddt*powf(ddt_dev[ind_loop],2.f));

                up[v_index].field = du_new_dev[ind_loop]; up[v_index].pres = pres;
                vp[v_index].field = dv_new_dev[ind_loop]; vp[v_index].pres = pres;
                wp[v_index].field = dw_new_dev[ind_loop]; wp[v_index].pres = pres;
                v_index++;
            }

        //Sort vectors (both the solution and the weights)
        bubbleSortDev(up, point_count);
        bubbleSortDev(vp, point_count);
        bubbleSortDev(wp, point_count);

        //Compute cumulative weight
        pres_cum_u[0] = up[0].pres; pres_cum_v[0] = vp[0].pres; pres_cum_w[0] = wp[0].pres;
        for (unsigned int i=1; i<point_count; i++)
        {
            pres_cum_u[i] = pres_cum_u[i-1] + up[i].pres;
            pres_cum_v[i] = pres_cum_v[i-1] + vp[i].pres;
            pres_cum_w[i] = pres_cum_w[i-1] + wp[i].pres;
        }

        pres_med = 0.5f*pres_cum_u[point_count-1];

        //Look for the indices comprising pres_med and get the filtered value
        unsigned int cont = 0, ind_l, ind_r;

        //For u
        while (pres_med > pres_cum_u[cont]) {cont++;}
        if (cont == 0)
            du_l_dev[index] = up[0].field;
        else
        {
            ind_r = cont; ind_l = cont-1;
            du_l_dev[index] = ((pres_cum_u[ind_r] - pres_med)*up[ind_l].field + (pres_med - pres_cum_u[ind_l])*up[ind_r].field)/(pres_cum_u[ind_r] - pres_cum_u[ind_l]);
        }

        //For v
        cont = 0;
        while (pres_med > pres_cum_v[cont]) {cont++;}
        if (cont == 0)
            dv_l_dev[index] = vp[0].field;
        else
        {
            ind_r = cont; ind_l = cont-1;
            dv_l_dev[index] = ((pres_cum_v[ind_r] - pres_med)*vp[ind_l].field + (pres_med - pres_cum_v[ind_l])*vp[ind_r].field)/(pres_cum_v[ind_r] - pres_cum_v[ind_l]);
        }

        //For w
        cont = 0;
        while (pres_med > pres_cum_w[cont]) {cont++;}
        if (cont == 0)
            dw_l_dev[index] = wp[0].field;
        else
        {
            ind_r = cont; ind_l = cont-1;
            dw_l_dev[index] = ((pres_cum_w[ind_r] - pres_med)*wp[ind_l].field + (pres_med - pres_cum_w[ind_l])*wp[ind_r].field)/(pres_cum_w[ind_r] - pres_cum_w[ind_l]);
        }
    }
    else
    {
        du_l_dev[index] = du_new_dev[index];
        dv_l_dev[index] = dv_new_dev[index];
        dw_l_dev[index] = dw_new_dev[index];
    }

    pd_l_dev[index] = pd_dev[index];
    puu_l_dev[index] = puu_dev[index];
    puv_l_dev[index] = puv_dev[index];
    pvu_l_dev[index] = pvu_dev[index];
    pvv_l_dev[index] = pvv_dev[index];
    pwu_l_dev[index] = pwu_dev[index];
    pwv_l_dev[index] = pwv_dev[index];
}

__device__ void CSF_cuda::computeMotionField(unsigned int index)
{
    const unsigned int v = index%(rows_i);
    const unsigned int u = index/(rows_i);
    const unsigned int srcIndex = u + v * cols_i;

    //Fill the matrices dx,dy,dz with the scene flow estimate
    if (position_old_dev[level_image][index].z > 0)
    {
        D_dev[srcIndex].z = dw_l_dev[index];

        // TODO Andre: In our camera model, we don't use the pinhole model
        // and do not rectify the image. So we get a bigger error / deviation
        // when simply using the inverse f. We have to replace the du_l_dev[index]*inv_f
        // term by the difference in our DF2CS map for that specific pixels.
        // Probably I shoud give the pointers to the DF2CS-map in the beginning,
        // and I should also give the position- and color-array which we already
        // have uploaded to the gpu to avoid performance loss:

        const unsigned int factor = height/rows;

        const unsigned int sY = (index%rows) * factor;
        const unsigned int sX = (index/rows) * factor;

        const float sDXf = du_l_dev[index] * factor;
        const float sDYf = dv_l_dev[index] * factor;

        const int sDXa = int(sDXf);
        const int sDYa = int(sDYf);

        const int sDXb = sDXa < 0 ? sDXa - 1 : sDXa + 1;
        const int sDYb = sDYa < 0 ? sDYa - 1 : sDYa + 1;

        float decPartX = abs(sDXf - sDXa);
        float decPartY = abs(sDYf - sDYa);


        //
        // Bilinear interpolate the TARGET DF2CS value for more precision in xy-direction
        // START ==>
        //
        const unsigned int dSrcIndex = sX + sY * width;
        const unsigned int dTgt00Index = max(min(int((sX + sDXa) + (sY + sDYa) * width), int(width*height-1)), 0);
        const unsigned int dTgt10Index = max(min(int((sX + sDXb) + (sY + sDYa) * width), int(width*height-1)), 0);
        const unsigned int dTgt01Index = max(min(int((sX + sDXa) + (sY + sDYb) * width), int(width*height-1)), 0);
        const unsigned int dTgt11Index = max(min(int((sX + sDXb) + (sY + sDYb) * width), int(width*height-1)), 0);

        const float2 DF2CS_LinX0Interp = {
            DF2CS[dTgt00Index].x * (1-decPartX) + DF2CS[dTgt10Index].x * decPartX,
            DF2CS[dTgt00Index].y * (1-decPartX) + DF2CS[dTgt10Index].y * decPartX
        };

        const float2 DF2CS_LinX1Interp = {
            DF2CS[dTgt01Index].x * (1-decPartX) + DF2CS[dTgt11Index].x * decPartX,
            DF2CS[dTgt01Index].y * (1-decPartX) + DF2CS[dTgt11Index].y * decPartX
        };

        const float2 DF2CS_BiLinInterp = {
            DF2CS_LinX0Interp.x * (1-decPartY) + DF2CS_LinX1Interp.x * decPartY,
            DF2CS_LinX0Interp.y * (1-decPartY) + DF2CS_LinX1Interp.y * decPartY,
        };
        //
        // <== END
        //

        // Calculate Delta between the Target DF2CS value (where the flow vector
        // points to) and the SOURCE DF2CS value (where the flow vector starts):
        const float2 deltaDF2CS = {
            DF2CS_BiLinInterp.x - DF2CS[dSrcIndex].x,
            DF2CS_BiLinInterp.y - DF2CS[dSrcIndex].y
        };

        //dy_dev[index] = position_old_dev[level_image][index].z*du_l_dev[index]*inv_f + dw_l_dev[index]*position_old_dev[level_image][index].x/position_old_dev[level_image][index].z;
        //dz_dev[index] = position_old_dev[level_image][index].z*dv_l_dev[index]*inv_f + dw_l_dev[index]*position_old_dev[level_image][index].y/position_old_dev[level_image][index].z;
        D_dev[srcIndex].x = -position_old_dev[level_image][index].z * deltaDF2CS.x + dw_l_dev[index]*position_old_dev[level_image][index].x/position_old_dev[level_image][index].z;
        D_dev[srcIndex].y = position_old_dev[level_image][index].z * deltaDF2CS.y + dw_l_dev[index]*position_old_dev[level_image][index].y/position_old_dev[level_image][index].z;
    }
    else
    {
        D_dev[srcIndex].z = 0.f;
        D_dev[srcIndex].x = 0.f;
        D_dev[srcIndex].y = 0.f;
    }
    D_dev[srcIndex].w = 0.f;
}


//                              Bridges
//=================================================================================
void GaussianPyramidBridge(CSF_cuda *csf, unsigned int levels)
{
    for (unsigned int i=0; i<levels; i++)
    {
        const unsigned int cols_i_aux = 640/(powf(2,i));
        const unsigned int rows_i_aux = 576/(powf(2,i));

        cudaMemcpy(&csf->rows_i, &rows_i_aux, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(&csf->cols_i, &cols_i_aux, sizeof(unsigned int), cudaMemcpyHostToDevice);

        ComputePyramidLevelKernel <<<N_blocks, N_threads>>>(csf, i);
    }
}

void AssignZerosBridge(CSF_cuda *csf)
{
    AssignZerosKernel <<<N_blocks, N_threads>>>(csf);
}

void UpsampleBridge(CSF_cuda *csf)
{
    UpsampleCopyKernel <<<N_blocks, N_threads>>>(csf);
    UpsampleFilterKernel <<<N_blocks, N_threads>>>(csf);
}

void ImageGradientsBridge(CSF_cuda *csf)
{
    ComputeImGradients <<<N_blocks, N_threads>>>(csf);
}

void WarpingBridge(CSF_cuda *csf)
{
    PerformWarping <<<N_blocks, N_threads>>> (csf);
}

void RijBridge(CSF_cuda *csf)
{
    RijKernel <<<N_blocks, N_threads>>>(csf);
}

void MuAndStepSizesBridge(CSF_cuda *csf)
{
    MuAndStepSizesKernel <<<N_blocks, N_threads>>>(csf);
}

void DualVariablesBridge(CSF_cuda *csf)
{
    DualIteration <<<N_blocks, N_threads>>>(csf);
}

void PrimalVariablesBridge(CSF_cuda *csf)
{
    PrimalIteration <<<N_blocks, N_threads>>>(csf);
}

void DivergenceBridge(CSF_cuda *csf)
{
    DivergenceComputation <<<N_blocks, N_threads>>>(csf);
}

void GradientBridge(CSF_cuda *csf)
{
    GradientComputation <<<N_blocks, N_threads>>>(csf);
}

void FilterBridge(CSF_cuda *csf)
{
    SaturateSolution <<<N_blocks, N_threads>>>(csf);
    FilterSolution <<<N_blocks, N_threads>>>(csf);
}

void MotionFieldBridge(CSF_cuda *csf)
{
    MotionFieldKernel <<<N_blocks, N_threads>>>(csf);
}

void DebugBridge(CSF_cuda *csf_device)
{
    printf("Executing debug kernel");
    DebugKernel <<<1,1>>>(csf_device);
}

void BridgeBack(CSF_cuda *csf_host, CSF_cuda *csf_device)
{
    cudaMemcpy(csf_host, csf_device, sizeof(CSF_cuda), cudaMemcpyDeviceToHost);
    cudaFree(csf_device);
}


//                                  Kernels
//=============================================================================
__global__ void DebugKernel(CSF_cuda *csf)
{
    //Add here the code you want to use for debugging
    printf("\n dx: ");
    for (unsigned int i = 0; i< (csf->rows_i)*(csf->cols_i); i++)
        printf(" %f", csf->D_dev[i].x);

}

__global__ void ComputePyramidLevelKernel (CSF_cuda *csf, unsigned int level)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->computePyramidLevel(index, level);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void AssignZerosKernel (CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->assignZeros(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void UpsampleCopyKernel (CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i/4)
    {
        csf->upsampleCopyPrevSolution(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void UpsampleFilterKernel (CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->upsampleFilterPrevSolution(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void ComputeImGradients(CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->computeImGradients(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void PerformWarping(CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->performWarping(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void RijKernel(CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->computeRij(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void MuAndStepSizesKernel(CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->computeMu(index);
        csf->computeStepSizes(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void DualIteration(CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->updateDualVariables(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void PrimalIteration(CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->updatePrimalVariables(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void DivergenceComputation(CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->computeDivergence(index);
        index += blockDim.x*gridDim.x;
    }
}


__global__ void GradientComputation(CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->computeGradient(index);
        index += blockDim.x*gridDim.x;
    }
}


__global__ void SaturateSolution (CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->saturateVariables(index);
        index += blockDim.x*gridDim.x;
    }
}


__global__ void FilterSolution (CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->filterSolution(index);
        index += blockDim.x*gridDim.x;
    }
}

__global__ void MotionFieldKernel (CSF_cuda *csf)
{
    // detect pixel
    unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

    while (index < csf->rows_i*csf->cols_i)
    {
        csf->computeMotionField(index);
        index += blockDim.x*gridDim.x;
    }
}

//Naive implementations of bubbleSort (applied to very small arrays)
__device__ void bubbleSortDev(fieldAndPresence array[], unsigned int num_elem)
{
    bool go_on = true;
    while (go_on)
    {
        go_on = false;
        for (unsigned int i=1; i<num_elem; i++)
        {
            if (array[i-1].field > array[i].field)
            {
                ELEM_SWAP(array[i-1].field,array[i].field);
                ELEM_SWAP(array[i-1].pres,array[i].pres);
                go_on = true;
            }
        }
    }
}




