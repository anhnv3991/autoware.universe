#include <autoware/euclidean_cluster/cuda_utils/voxel_grid_search.hpp>
#include <autoware/euclidean_cluster/cuda_utils/cuda_utility.cuh>

namespace autoware::cuda
{

VoxelGridSearch::VoxelGridSearch(
    std::shared_ptr<CudaStream> stream,
    std::shared_ptr<CudaMempool> mempool
) :
    stream_(stream),
    mempool_(mempool),
    voxel_starting_pid_(stream_, mempool_),
    sorted_pid_(stream_, mempool_)
{
    res_ = 0;
}



void VoxelGridSearch::setResolution(float resolution)
{
    res_ = resolution;
}

__global__ void computeVoxelId(
    PointCloud2::Ptr cloud, float res, int * voxel_id
)
{

}

void VoxelGridSearch::setInputCloud(PointCloud2::ConstSharedPtr input)
{

}


}
