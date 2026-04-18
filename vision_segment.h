/**
 * @file vision_segment.h
 * @brief 基于YOLO的分割模型封装类
 * @date 2025-06-07
 */
#ifndef VISION_SEGMENT_H
#define VISION_SEGMENT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include "trtyolo.hpp"
#include "checkbox.h"

#define MASK_AREA_THRE 100
#define MASK_SIZE_THRE 60
class VisionSegment {
public:
    VisionSegment();
    ~VisionSegment();

    // 禁止拷贝
    VisionSegment(const VisionSegment&) = delete;
    VisionSegment& operator=(const VisionSegment&) = delete;

    // 允许移动
    VisionSegment(VisionSegment&& other) noexcept;
    VisionSegment& operator=(VisionSegment&& other) noexcept;

    /** 从engine路径初始化；成功返回true */
    bool Init(const std::string& engine_path);

    /** 释放模型 */
    void Reset();

    /**
     * 对img做分割，结果写入out，id_base递增。
     * @param img 输入图像
     * @param conf_thresh 置信度阈值
     * @param out 输出检测框列表
     * @param id_base ID基数，每次调用后会自动递增
     * @param append 若false会先清空out；若true则追加
     */
    void Run(const cv::Mat& img,
             double conf_thresh,
             std::vector<CheckBox>& out,
             uint64_t& id_base,
             bool append = false);

    /**
     * 对图像中的指定区域进行分割（基于检测框）
     * @param img 输入图像
     * @param detections 检测框列表
     * @param conf_thresh 置信度阈值
     * @param out 输出检测框列表（包含分割掩码）
     * @param id_base ID基数，每次调用后会自动递增
     * @param crop_size 裁剪尺寸
     */
    void RunWithBoxes(const cv::Mat& img,
                      const std::vector<CheckBox>& detections,
                      double conf_thresh,
                      std::vector<std::vector<CheckBox>>& out,
                      uint64_t& id_base,
                      int crop_size = 1280);

    /** 检查模型是否已初始化 */
    bool IsInited() const { return model_ != nullptr; }

private:
    std::unique_ptr<trtyolo::SegmentModel> model_;  // 模型指针
};

#endif // VISION_SEGMENT_H