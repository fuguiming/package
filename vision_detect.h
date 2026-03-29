#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include "trtyolo.hpp"
#include "checkbox.h"

struct BoundingBox
{
    int x1, y1, x2, y2;
    int cls;
    float score;
};

struct PCBRegion
{
    int x1, y1, x2, y2;
    
    PCBRegion(int _x1, int _y1, int _x2, int _y2)
    : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}
};

class VisionDetect
{
public:
    VisionDetect() = default;
    ~VisionDetect();

    VisionDetect(const VisionDetect &) = delete;
    VisionDetect &operator=(const VisionDetect &) = delete;

    bool Init(const std::string &engine_path);

    void Reset();

    void Run(const cv::Mat &img,
             double conf_thresh,
             std::vector<CheckBox> &out,
             uint64_t &id_base,
             bool append);

    bool IsInited() const { return model_ != nullptr; }

    int solder_width_thre = 20;
    int solder_height_thre = 20;

private:
    // trtyolo::DetectModel *model_ = nullptr;
    std::unique_ptr<trtyolo::DetectModel> model_;

private:

    std::vector<PCBRegion> detect_pcb_regions(const cv::Mat &img, float min_area_ratio=0.01);

    bool tile_overlaps_with_pcb(
        int x1, int y1, int x2, int y2,
        const std::vector<PCBRegion> &pcb_regions,
        float overlap_threshold = 0.1);

    std::vector<BoundingBox> NMS(std::vector<BoundingBox>& boxes, float iou_thresh);

    std::vector<BoundingBox> merge_boundary_boxes_by_distance(
        const std::vector<BoundingBox> &boxes,
        float distance_threshold);
};