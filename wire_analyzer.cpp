/**
 * @file wire_analyzer.cpp
 * @brief 导线分析模块实现
 * @date 2025-06-07
 */
#include "wire_analyzer.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <queue>
#include <algorithm>

// cv::Mat extractSkeleton(const cv::Mat& mask) {
//     cv::Mat skeleton = cv::Mat::zeros(mask.size(), CV_8UC1);
//     cv::Mat temp;
//     cv::Mat eroded;
//     cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    
//     mask.copyTo(temp);
    
//     bool done = false;
//     while (!done) {
//         cv::erode(temp, eroded, element);
//         cv::dilate(eroded, temp, element);
//         cv::subtract(temp, eroded, temp);
//         cv::bitwise_or(skeleton, temp, skeleton);
//         eroded.copyTo(temp);
        
//         done = (cv::countNonZero(temp) == 0);
//     }
    
//     return skeleton;
// }

std::vector<cv::Point> findEndpoints(const cv::Mat& mask)
{
    std::vector<cv::Point> points;

    for (int y = 0; y < mask.rows; ++y) {
        const uchar* ptr = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (ptr[x] > 0) {
                points.emplace_back(x, y);
            }
        }
    }

    if (points.size() < 2) return {};

    // 找最远的两个点（近似端点）
    double max_dist = 0;
    cv::Point p1, p2;

    for (size_t i = 0; i < points.size(); i += 10) {
        for (size_t j = i + 1; j < points.size(); j += 10) {
            double d = cv::norm(points[i] - points[j]);
            if (d > max_dist) {
                max_dist = d;
                p1 = points[i];
                p2 = points[j];
            }
        }
    }

    return {p1, p2};
}

double pointToBoxDistance(const cv::Point& p, const CheckBox& box)
{
    int cx = (box.left + box.right) / 2;
    int cy = (box.top + box.bottom) / 2;

    return cv::norm(p - cv::Point(cx, cy));
}

std::vector<cv::Point> getMaskPoints(const cv::Mat& mask)
{
    std::vector<cv::Point> pts;
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* ptr = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (ptr[x] > 0)
                pts.emplace_back(x, y);
        }
    }
    return pts;
}

std::vector<cv::Point> extractLocalSegment(
    const std::vector<cv::Point>& pts,
    const cv::Point& endpoint,
    int radius = 100)
{
    std::vector<cv::Point> local_pts;

    local_pts.clear();

    for (const auto& p : pts) {
        if (cv::norm(p - endpoint) < radius) {
            local_pts.push_back(p);
        }
    }

    return local_pts; // fallback
}

float computeDirectionFromEndpoint(const cv::Mat& mask, const cv::Point& endpoint)
{
    auto pts = getMaskPoints(mask);
    std::cout<<"pts: "<<pts.size()<<std::endl;
    // ===== 局部mask =====
    std::vector<cv::Point> local_pts = extractLocalSegment(pts, endpoint);
    std::cout<<"local_pts: "<<local_pts.size()<<std::endl;

    // ===== 最小外接矩形 =====
    cv::RotatedRect rect = cv::minAreaRect(local_pts);

    float angle = rect.angle;

    // OpenCV angle修正
    if (rect.size.width < rect.size.height)
        angle += 90.0f;

    return angle; // degree
}

bool isDirectionTowardsSolder(
    const cv::Point& endpoint,
    float angle_deg,
    const CheckBox& solder,
    float dist_thresh = 50  // 新增：距离阈值（像素）
)
{
    // ===== 1. 焊点中心 =====
    cv::Point2f center(
        (solder.left + solder.right) * 0.5f,
        (solder.top + solder.bottom) * 0.5f
    );

    // ===== 2. 方向向量（单位向量）=====
    float rad = angle_deg * CV_PI / 180.0f;
    cv::Point2f dir(std::cos(rad), std::sin(rad));

    // ===== 3. 向量：端点 -> 焊点 =====
    cv::Point2f v = center - cv::Point2f(endpoint);

    // ===== 4. 点到直线距离（叉积）=====
    float distance = std::abs(v.x * dir.y - v.y * dir.x);
    std::cout<<"distance: "<<distance<<" thre: "<<dist_thresh<<std::endl;

    return distance < dist_thresh;
}

bool isWireBelongToSolder_v2(
    const CheckBox& solder,
    const CheckBox& wire,
    const float min_dist)
{
    if (wire.mask.empty()) return false;

    // 创建掩码矩阵
    cv::Mat float_mask(wire.mask_height, wire.mask_width, CV_32FC1, 
                    const_cast<float*>(wire.mask.data()));
    
    // 计算边界框在原图中的位置
    int box_width = wire.right - wire.left + 1;
    int box_height = wire.bottom - wire.top + 1;

    // std::cout<<"mask box width: "<<box_width<<" "<<"mask box height: "<<box_height<<std::endl;
    
    // 调整掩码大小到边界框尺寸
    cv::Mat resized_mask;
    cv::resize(float_mask, resized_mask, cv::Size(box_width, box_height), 
            0, 0, cv::INTER_LINEAR);
    
    // 二值化掩码并转换为8位
    cv::Mat bool_mask;
    cv::threshold(resized_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);
    bool_mask.convertTo(bool_mask, CV_8UC1);
    // cv::imwrite("bool_mask.jpg", bool_mask);
    
    auto endpoints = findEndpoints(bool_mask);
    if (endpoints.size() < 2) return false;

    for (auto& ep : endpoints) {
        // 转换到图像坐标系
        cv::Point ep_img;
        ep_img.x = ep.x + wire.left;
        ep_img.y = ep.y + wire.top;
        // ===== 距离过滤 =====
        double dist = pointToBoxDistance(ep_img, solder);
        std::cout<<"solder: "<<solder.center_x()<<" "<<solder.center_y()<<" ep_img: "<<ep_img.x<<" "<<ep_img.y<<" dist: "<<dist<<" dist thre: "<<std::min(min_dist, SOLDER_WIRE_DIST_THRE)<<std::endl;
        if (dist > std::min(min_dist, SOLDER_WIRE_DIST_THRE)) continue;//当两个焊点比较近，阈值要相应调整
        return true;
        // // ===== 方向判断 =====方向判断留给对比模板有漏检时进行补救
        // float angle = computeDirectionFromEndpoint(bool_mask, ep);
        // std::cout<<"angle: "<<angle<<std::endl;

        // double pt_to_line_dist_thre = std::sqrt(solder.width() * solder.width() + solder.height() * solder.height()) / 2.0;

        // if (isDirectionTowardsSolder(ep_img, angle, solder, pt_to_line_dist_thre)) {
        //     return true;
        // }
    }

    return false;
}

float centerDistance(const CheckBox& a, const CheckBox& b)
{
    float dx = a.center_x() - b.center_x();
    float dy = a.center_y() - b.center_y();
    return std::sqrt(dx * dx + dy * dy);
}

float minCenterDistance(const std::vector<CheckBox>& boxes, int idx)
{
    float min_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < boxes.size(); ++i)
    {
        if (i == idx) continue;

        float d = centerDistance(boxes[idx], boxes[i]);
        if (d < min_dist)
            min_dist = d;
    }
    return min_dist;
}

void matchWiresToSolder(
    const std::vector<CheckBox>& solderings,
    const std::vector<std::vector<CheckBox>>& all_segmentations,
    std::vector<SolderWire> &results)
{
    std::cout<<"solderings size: "<<solderings.size()<<std::endl;
    for (size_t i = 0; i < solderings.size(); ++i) {

        SolderWire solderwire;
        const auto& solder = solderings[i];
        //计算当前soldering与其他soldering的最小距离，作为后续端点到焊点距离判断的依据之一
        float min_dist = minCenterDistance(solderings, i);
        solderwire.solder_info = solder;
        const auto& wires = all_segmentations[i];

        std::cout<<"wires size: "<<wires.size()<<std::endl;
        for (size_t j = 0; j < wires.size(); ++j) {
            if (isWireBelongToSolder_v2(solder, wires[j], min_dist)) {
                solderwire.wires.push_back(wires[j]);
            }
        }
        if (0 == solderwire.wires.size())
        {
            continue;//空焊点跳过
        }
        results.push_back(solderwire);
    }
}

void analyseBasedTruth(
    const cv::Mat& img,
    const std::vector<CheckBox>& solderings,
    const std::vector<std::vector<CheckBox>>& all_segmentations,
    const std::vector<SolderWire>& truth,
    const TemplateMatchResult& params,
    std::vector<SolderWire>& results)
{

}

float calcIoU(const CheckBox& a, const CheckBox& b)
{
    float inter_left   = std::max(a.left, b.left);
    float inter_top    = std::max(a.top, b.top);
    float inter_right  = std::min(a.right, b.right);
    float inter_bottom = std::min(a.bottom, b.bottom);

    float inter_w = std::max(0.0f, inter_right - inter_left);
    float inter_h = std::max(0.0f, inter_bottom - inter_top);
    float inter_area = inter_w * inter_h;

    float area_a = (a.right - a.left) * (a.bottom - a.top);
    float area_b = (b.right - b.left) * (b.bottom - b.top);

    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

// 应用变换到点
cv::Point2f applyTransform(const cv::Point2f& point, float angle, const cv::Point2f& src_center, const cv::Point2f& translation) {
    float rad = angle * CV_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);
    
    // 1. 平移到以中心为原点
    float x = point.x - src_center.x;
    float y = point.y - src_center.y;

    // 2. 旋转
    float x_new = x * cos_a - y * sin_a;
    float y_new = x * sin_a + y * cos_a;

    // 3. 平移回去 + translation
    cv::Point2f rotated_point(
        x_new + src_center.x,
        y_new + src_center.y
    );
    std::cout<<"solder rotate: "<<rotated_point.x<< " "<<rotated_point.y<<std::endl;
    return rotated_point + (translation - src_center);
}

// 应用变换到CheckBox
CheckBox transformCheckBox(const cv::Mat& template_img, const CheckBox& box, float angle, const cv::Point2f& translation) {
    CheckBox transformed_box;
    cv::Point2f src_center(template_img.cols / 2, template_img.rows / 2);
    // 转换中心点
    cv::Point2f center(box.center_x(), box.center_y());
    cv::Point2f transformed_center = applyTransform(center, angle, src_center, translation);
    
    // 在旋转后，宽度和高度可能会变化，这里简化为保持原尺寸
    // 如果需要更精确的变换，需要计算旋转后的包围盒
    transformed_box.left = static_cast<int>(transformed_center.x - box.width() / 2.0f);
    transformed_box.top = static_cast<int>(transformed_center.y - box.height() / 2.0f);
    transformed_box.right = static_cast<int>(transformed_center.x + box.width() / 2.0f);
    transformed_box.bottom = static_cast<int>(transformed_center.y + box.height() / 2.0f);
    transformed_box.cls = box.cls;
    
    return transformed_box;
}

bool checkWireColorsMatch(const std::vector<CheckBox>& wires1, 
                          const std::vector<CheckBox>& wires2) {
    // 统计第一个集合中各颜色的数量
    std::map<int, int> color_count1;
    for (const auto& wire : wires1) {
        color_count1[wire.cls]++;
    }
    
    // 统计第二个集合中各颜色的数量
    std::map<int, int> color_count2;
    for (const auto& wire : wires2) {
        color_count2[wire.cls]++;
    }
    
    // 比较两个颜色计数map是否相同
    return color_count1 == color_count2;
}

std::vector<MatchPair> matchSolders(
    const cv::Mat& template_img,
    const std::vector<SolderWire>& truth,
    const std::vector<SolderWire>& results,
    const TemplateMatchResult& params,//校正参数
    float iou_thresh)
{
    std::vector<MatchPair> matches;
    std::vector<bool> used(results.size(), false);

    std::cout<<"params: "<<params.maxLoc.x<<" "<<params.maxLoc.y<<" "<<params.angle<<std::endl;
    for (int i = 0; i < truth.size(); ++i)
    {
        // 将truth焊点变换到当前图像坐标系
        CheckBox transformed_truth_solder = transformCheckBox(
            template_img,
            truth[i].solder_info, 
            -params.angle,  // 反向旋转
            cv::Point2f(params.maxLoc.x, params.maxLoc.y)  // 正向平移
        );

        std::cout<<"solder before template: "<<truth[i].solder_info.left<< " "<<truth[i].solder_info.top<<" "<<truth[i].solder_info.right<<" "<<truth[i].solder_info.bottom<<std::endl;
        std::cout<<"solder after  template: "<<transformed_truth_solder.left<< " "<<transformed_truth_solder.top<<" "<<transformed_truth_solder.right<<" "<<transformed_truth_solder.bottom<<std::endl;

        float best_iou = 0;
        int best_j = -1;

        for (int j = 0; j < results.size(); ++j)
        {
            if (used[j]) continue;

            float iou = calcIoU(transformed_truth_solder, results[j].solder_info);

            if (iou > best_iou)
            {
                best_iou = iou;
                best_j = j;
            }
        }

        std::cout<<i<<" "<<best_j<<" iou: "<<best_iou<<std::endl;

        if (best_j != -1 && best_iou > iou_thresh)//正常匹配对
        {
            bool wire_count_match = false;
            bool wire_color_match = false;
            // 检查导线数量是否匹配
            if (results[best_j].wires.size() == truth[i].wires.size()) {
                wire_count_match = true;
                // 数量匹配，检查颜色
                wire_color_match = checkWireColorsMatch(results[best_j].wires, truth[i].wires);
            }
            matches.push_back({i, best_j, wire_count_match, wire_color_match});
            used[best_j] = true;
        }
        else
        {
            matches.push_back({i, -1, false, false});//检测板漏掉
        }
    }

    for (int j = 0; j < results.size(); ++j)
    {
        if (used[j]) continue;
        matches.push_back({-1, j, false, false});//检测板多检测
    }

    return matches;
}