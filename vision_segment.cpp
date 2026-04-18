/**
 * @file vision_segment.cpp
 * @brief 基于YOLO的分割模型封装类实现
 * @date 2025-06-07
 */
#include "vision_segment.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

// 裁剪信息结构体
struct CropInfo {
    cv::Rect crop_rect;        // 在原图中的裁剪区域
    cv::Point box_offset;       // 检测框在裁剪图像中的偏移量
};

// 根据检测框计算裁剪区域
static CropInfo calculateCropRegion(const CheckBox& box, int image_width, int image_height, int target_size) {
    CropInfo info;
    int center_x = box.center_x();
    int center_y = box.center_y();
    
    // 计算裁剪区域的左上角坐标
    int crop_x = center_x - target_size / 2;
    int crop_y = center_y - target_size / 2;
    
    // 处理边界情况
    int crop_x1 = crop_x;
    int crop_y1 = crop_y;
    int crop_x2 = crop_x + target_size;
    int crop_y2 = crop_y + target_size;
    
    // 如果超出左边界
    if (crop_x1 < 0) {
        crop_x1 = 0;
        crop_x2 = target_size;
    }
    // 如果超出右边界
    else if (crop_x2 > image_width) {
        crop_x2 = image_width;
        crop_x1 = image_width - target_size;
    }
    
    // 如果超出上边界
    if (crop_y1 < 0) {
        crop_y1 = 0;
        crop_y2 = target_size;
    }
    // 如果超出下边界
    else if (crop_y2 > image_height) {
        crop_y2 = image_height;
        crop_y1 = image_height - target_size;
    }
    
    // 确保裁剪区域在图像范围内
    crop_x1 = std::max(0, std::min(crop_x1, image_width - target_size));
    crop_y1 = std::max(0, std::min(crop_y1, image_height - target_size));
    crop_x2 = crop_x1 + target_size;
    crop_y2 = crop_y1 + target_size;
    
    info.crop_rect = cv::Rect(crop_x1, crop_y1, target_size, target_size);
    
    // 计算原始检测框在裁剪图像中的偏移量
    info.box_offset = cv::Point(box.left - crop_x1, box.top - crop_y1);
    
    return info;
}

static void splitMaskToBoxes(const CheckBox& in_box,
                             std::vector<CheckBox>& out,
                             uint64_t& id_base,
                             float mask_thresh = 0.5,
                             int min_area = 20)
{
    if (in_box.mask.empty()) return;
    // 1. 转成 cv::Mat
    cv::Mat mask(in_box.mask_height, in_box.mask_width, CV_32F,
                 (void*)in_box.mask.data());
    // 调整掩码大小到边界框尺寸
    cv::Mat resized_mask;
    cv::resize(mask, resized_mask, cv::Size(in_box.width(), in_box.height()), 
            0, 0, cv::INTER_LINEAR);
    // 2. 二值化
    cv::Mat bin;
    cv::threshold(resized_mask, bin, mask_thresh, 255, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U);
    // 3. 连通域分析
    cv::Mat labels, stats, centroids;
    int num = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);
    for (int i = 1; i < num; ++i) // 0是背景
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        // std::cout<<"area: "<<area<<std::endl;
        if (area < min_area) continue;
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        // if (std::max(w,h) < MASK_SIZE_THRE) continue;
        // 4. 构建新mask
        cv::Mat sub_mask = (labels == i);
        sub_mask = sub_mask(cv::Rect(x, y, w, h)).clone();
        // 3. resize到 160x160（保持和YOLO一致）
        cv::Mat resized_mask;
        cv::resize(sub_mask, resized_mask, cv::Size(in_box.mask_width, in_box.mask_height), 0, 0, cv::INTER_NEAREST);
        // 转回 float mask（保持一致）
        resized_mask.convertTo(resized_mask, CV_32F, 1.0 / 255);
        // 5. 生成新 CheckBox
        CheckBox new_box;
        new_box.left   = in_box.left + x;
        new_box.top    = in_box.top + y;
        new_box.right  = new_box.left + w;
        new_box.bottom = new_box.top + h;
        new_box.cls    = in_box.cls;
        new_box.score  = in_box.score;
        new_box.id     = id_base++;

        new_box.mask_width  = in_box.mask_width;
        new_box.mask_height = in_box.mask_height;
        // std::cout<<"new mask w h: "<<w<<" "<<h<<std::endl;
        // std::cout<<"src mask w h: "<<in_box.mask_width<<" "<<in_box.mask_height<<std::endl;

        new_box.mask.assign((float*)resized_mask.datastart,
                            (float*)resized_mask.dataend);

        out.push_back(new_box);
    }
}

cv::Point2f computeMainDirection(const cv::Mat& mask)
{
    std::vector<cv::Point> pts;
    cv::findNonZero(mask, pts);

    if (pts.size() < 10) return {1, 0};

    cv::Mat data_pts(pts.size(), 2, CV_32F);
    for (int i = 0; i < pts.size(); i++)
    {
        data_pts.at<float>(i, 0) = pts[i].x;
        data_pts.at<float>(i, 1) = pts[i].y;
    }

    cv::PCA pca(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

    cv::Point2f dir(
        pca.eigenvectors.at<float>(0, 0),
        pca.eigenvectors.at<float>(0, 1)
    );

    return dir; // 主方向
}

float pointToLineDistance(cv::Point2f pt, cv::Point2f line_pt, cv::Point2f dir)
{
    cv::Point2f v = pt - line_pt;
    return std::abs(v.x * dir.y - v.y * dir.x) / (cv::norm(dir) + 1e-6);
}

bool isNearMainLine(const cv::Mat& comp_mask,
                    cv::Point2f line_pt,
                    cv::Point2f dir,
                    float dist_thre)
{
    std::vector<cv::Point> pts;
    cv::findNonZero(comp_mask, pts);

    for (auto& p : pts)
    {
        if (pointToLineDistance(p, line_pt, dir) < dist_thre)
            return true;
    }
    return false;
}

static void splitMaskToBoxes_v2(
    const CheckBox& in_box,
    std::vector<CheckBox>& out,
    uint64_t& id_base,
    float mask_thresh = 0.5,
    int min_area = 20)
{
    if (in_box.mask.empty()) return;

    cv::Mat mask(in_box.mask_height, in_box.mask_width, CV_32F,
                 (void*)in_box.mask.data());

    cv::Mat resized_mask;
    cv::resize(mask, resized_mask,
               cv::Size(in_box.width(), in_box.height()));

    cv::Mat bin;
    cv::threshold(resized_mask, bin, mask_thresh, 255, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U);

    cv::Mat labels, stats, centroids;
    int num = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);

    if (num <= 1) return;

    // ===== 找最大连通域 =====
    int max_idx = -1;
    int max_area = 0;

    for (int i = 1; i < num; ++i)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > max_area)
        {
            max_area = area;
            max_idx = i;
        }
    }

    if (max_idx < 0) return;

    cv::Mat main_mask = (labels == max_idx);

    // ===== 主方向 =====
    cv::Point2f main_dir = computeMainDirection(main_mask);

    // 主线参考点（用质心）
    cv::Point2f center(
        centroids.at<double>(max_idx, 0),
        centroids.at<double>(max_idx, 1));

    // ===== 合并mask =====
    cv::Mat final_mask = main_mask.clone();

    for (int i = 1; i < num; ++i)
    {
        if (i == max_idx) continue;

        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < min_area) continue;

        cv::Mat comp_mask = (labels == i);

        // 条件1：靠近主线
        if (!isNearMainLine(comp_mask, center, main_dir, 5))
            continue;

        // 合并
        final_mask |= comp_mask;
    }

    // ===== 输出一个box =====
    cv::Rect rect = cv::boundingRect(final_mask);

    CheckBox new_box;
    new_box.left   = in_box.left + rect.x;
    new_box.top    = in_box.top + rect.y;
    new_box.right  = new_box.left + rect.width;
    new_box.bottom = new_box.top + rect.height;

    new_box.cls = in_box.cls;
    new_box.score = in_box.score;
    new_box.id = id_base++;

    new_box.mask_width  = in_box.mask_width;
    new_box.mask_height = in_box.mask_height;

    // mask resize回原大小
    cv::Mat resized_final;
    cv::resize(final_mask(rect), resized_final,
               cv::Size(in_box.mask_width, in_box.mask_height),
               0, 0, cv::INTER_NEAREST);

    resized_final.convertTo(resized_final, CV_32F, 1.0 / 255);

    new_box.mask.assign((float*)resized_final.datastart,
                        (float*)resized_final.dataend);

    out.push_back(new_box);
}

// 转换内部结果到CheckBox格式
static void convertToCheckBox(const trtyolo::SegmentRes& result,
                              std::vector<CheckBox>& out,
                              uint64_t& id_base,
                              const cv::Rect& crop_rect = cv::Rect(),
                              bool is_cropped = false) {
    // 同一颜色不同连通域需要切分，去掉小面积连通域
    for (size_t i = 0; i < result.num; ++i) {
        CheckBox box;
        
        box.left = result.boxes[i].left + crop_rect.x;
        box.top = result.boxes[i].top + crop_rect.y;
        box.right = result.boxes[i].right + crop_rect.x;
        box.bottom = result.boxes[i].bottom + crop_rect.y;
        box.cls = result.classes[i];
        box.score = result.scores[i];
        box.id = id_base++;

        if (i < result.masks.size())
        {
            box.mask = result.masks[i].data;
            box.mask_width  = result.masks[i].width;
            box.mask_height = result.masks[i].height;
            // 拆分连通域
            splitMaskToBoxes_v2(box, out, id_base, 0.5, MASK_AREA_THRE);
        }
        else
        {
            out.push_back(box);
        }
    }
}

// VisionSegment实现
VisionSegment::VisionSegment() : model_(nullptr) {}

VisionSegment::~VisionSegment() {
    Reset();
}

VisionSegment::VisionSegment(VisionSegment&& other) noexcept
    : model_(std::move(other.model_)) {
}

VisionSegment& VisionSegment::operator=(VisionSegment&& other) noexcept {
    if (this != &other) {
        model_ = std::move(other.model_);
    }
    return *this;
}

bool VisionSegment::Init(const std::string& engine_path) {
    try {
        if (!fs::exists(engine_path)) {
            std::cerr << "Engine path does not exist: " << engine_path << std::endl;
            return false;
        }

        // 创建推理选项
        trtyolo::InferOption option;
        option.enableSwapRB();
        
        // 创建模型
        model_ = std::make_unique<trtyolo::SegmentModel>(engine_path, option);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize model: " << e.what() << std::endl;
        return false;
    }
}

void VisionSegment::Reset() {
    if (model_) {
        model_.reset();
    }
}

void VisionSegment::Run(const cv::Mat& img,
                        double conf_thresh,
                        std::vector<CheckBox>& out,
                        uint64_t& id_base,
                        bool append) {
    if (!IsInited()) {
        throw std::runtime_error("Model not initialized");
    }

    if (!append) {
        out.clear();
    }

    try {
        // 创建推理图像
        trtyolo::Image trt_img(const_cast<uchar*>(img.data), img.cols, img.rows);

        // 执行推理
        auto result = model_->predict(trt_img);

        // 可以根据置信度阈值过滤结果
        // 注意：如果模型内部已经做了过滤，这里可能不需要再做
        // 如果需要过滤，可以添加以下代码：
        /*
        trtyolo::SegmentRes filtered_result;
        filtered_result.num = 0;
        for (size_t i = 0; i < result.num; ++i) {
            if (result.scores[i] >= conf_thresh) {
                filtered_result.boxes.push_back(result.boxes[i]);
                filtered_result.classes.push_back(result.classes[i]);
                filtered_result.scores.push_back(result.scores[i]);
                filtered_result.masks.push_back(result.masks[i]);
                filtered_result.num++;
            }
        }
        convertToCheckBox(filtered_result, out, id_base);
        */
        
        // 默认使用所有结果
        convertToCheckBox(result, out, id_base);

    } catch (const std::exception& e) {
        std::cerr << "Run inference failed: " << e.what() << std::endl;
        throw;
    }
}

void VisionSegment::RunWithBoxes(const cv::Mat& img,
                                  const std::vector<CheckBox>& detections,
                                  double conf_thresh,
                                  std::vector<std::vector<CheckBox>>& out,
                                  uint64_t& id_base,
                                  int crop_size) {
    if (!IsInited()) {
        throw std::runtime_error("Model not initialized");
    }

    out.clear();

    if (detections.empty()) {
        return;
    }

    try {
        // 存储每个检测框的分割结果
        out.reserve(detections.size());

        for (size_t idx = 0; idx < detections.size(); ++idx) {
            const auto& det = detections[idx];
            
            // 计算裁剪区域
            CropInfo crop_info = calculateCropRegion(det, img.cols, img.rows, crop_size);
            
            // 裁剪图像
            cv::Mat cropped_image = img(crop_info.crop_rect).clone();
            
            // 对裁剪图像进行分割推理
            trtyolo::Image trt_img(cropped_image.data, cropped_image.cols, cropped_image.rows);
            auto result = model_->predict(trt_img);
            
            // 可以根据置信度阈值过滤结果
            // 如果需要过滤，取消下面的注释
            
            trtyolo::SegmentRes filtered_result;
            filtered_result.num = 0;
            for (size_t i = 0; i < result.num; ++i) {
                if (result.scores[i] >= conf_thresh) {
                    filtered_result.boxes.push_back(result.boxes[i]);
                    filtered_result.classes.push_back(result.classes[i]);
                    filtered_result.scores.push_back(result.scores[i]);
                    filtered_result.masks.push_back(result.masks[i]);
                    filtered_result.num++;
                }
            }
            
            // 默认使用所有结果
            std::vector<CheckBox> out_crop;
            std::cout<<"process ===================== "<<idx<<std::endl;
            convertToCheckBox(filtered_result, out_crop, id_base, crop_info.crop_rect, true);
            out.push_back(out_crop);
        }
    } catch (const std::exception& e) {
        std::cerr << "RunWithBoxes inference failed: " << e.what() << std::endl;
        throw;
    }
}
