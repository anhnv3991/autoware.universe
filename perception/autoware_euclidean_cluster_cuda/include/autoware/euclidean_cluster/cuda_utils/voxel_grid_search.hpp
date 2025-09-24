#ifndef AUTOWARE_EUCLIDEAN_CLUSTER_CUDA_VOXEL_GRID_SEARCH_HPP_
#define AUTOWARE_EUCLIDEAN_CLUSTER_CUDA_VOXEL_GRID_SEARCH_HPP_

#include <memory>

#include "cuda_common.hpp"
#include "cuda_mempool_wrapper.hpp"
#include "cuda_stream_wrapper.hpp"
#include "cuda_point_cloud2.hpp"

namespace autoware::cuda
{

class VoxelGridSearch
{
public:
    using SharedPtr = std::shared_ptr<VoxelGridSearch>;
    using ConstSharedPtr = std::shared_ptr<const VoxelGridSearch>;
    
    VoxelGridSearch(
        std::shared_ptr<CudaStream> stream = std::make_shared<CudaStream>(true),
        std::shared_ptr<CudaMempool> mempool = nullptr
    );

    VoxelGridSearch(const VoxelGridSearch &);
    VoxelGridSearch(VoxelGridSearch &&);

    VoxelGridSearch& operator=(const VoxelGridSearch &);
    VoxelGridSearch& operator=(VoxelGridSearch &&);

    void setResolution(float resolution);
    void setInputCloud(PointCloud2::ConstSharedPtr input);
    size_t radiusSearch(
        PointCloud2 & query_cloud, float radius,
        device_vector<int> & starting_pid, 
        device_vector<int> & neighbor_ids
    );

private:
    std::shared_ptr<CudaStream> stream_;
    std::shared_ptr<CudaMempool> mempool_;

    float res_;
    PointCloud2::ConstSharedPtr cloud_;
    device_vector<int> voxel_starting_pid_;
    device_vector<int> sorted_pid_;
};

}

#endif
