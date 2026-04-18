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
    // std::cout<<"pts: "<<pts.size()<<std::endl;
    // ===== 局部mask =====
    std::vector<cv::Point> local_pts = extractLocalSegment(pts, endpoint);
    // std::cout<<"local_pts: "<<local_pts.size()<<std::endl;

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

// ===== 判断端点是否跨越焊点 =====
bool isAcrossSolder(const std::vector<cv::Point>& endpoints_img, const cv::Point& center, const float dist_thre)
{
    if (endpoints_img.size() < 2) return false;

    // 只取两个端点（导线通常2个端点）
    cv::Point A = endpoints_img[0];
    cv::Point B = endpoints_img[1];

    cv::Point2f AB(B.x - A.x, B.y - A.y);
    cv::Point2f AC(center.x - A.x, center.y - A.y);
    cv::Point2f BC(center.x - B.x, center.y - B.y);

    // 点积
    float dot1 = AB.x * AC.x + AB.y * AC.y;
    float dot2 = (-AB.x) * BC.x + (-AB.y) * BC.y;

    // 条件1：center 在 A 和 B 之间
    // std::cout<<"dot1: "<<dot1<<" dot2: "<<dot2<<" "<<A.x<<" "<<A.y<<" "<<B.x<<" "<<B.y<<" "<<center.x<<" "<<center.y<<std::endl;
    if (dot1 <= 0 || dot2 <= 0)
        return false;

    // ===== 计算垂足位置 t =====
    float ab2 = AB.x * AB.x + AB.y * AB.y;
    if (ab2 < 1e-6) return false;

    float t = dot1 / ab2;  // 等价于 (AC·AB)/(AB·AB)

    // ===== 垂足坐标 =====
    cv::Point2f foot(
        A.x + t * AB.x,
        A.y + t * AB.y
    );

    // ===== 计算垂足到端点距离 =====
    float distA = cv::norm(cv::Point2f(A) - foot);
    float distB = cv::norm(cv::Point2f(B) - foot);

    float minDist = std::min(distA, distB);

    // 条件2：距离满足
    if (minDist > dist_thre)
    {
        return true;
    }

    return false;
}

// 多维度判断导线与焊点是否关联
bool isWireBelongToSolder_v2(
    const CheckBox& solder,
    const CheckBox& wire,
    const cv::Point2f center,
    const float min_dist,
    const float angle_thre)
{
    if (wire.mask.empty()) return false;

    // 创建掩码矩阵
    cv::Mat float_mask(wire.mask_height, wire.mask_width, CV_32FC1, 
                    const_cast<float*>(wire.mask.data()));
    
    // 计算边界框在原图中的位置
    int box_width = wire.right - wire.left + 1;
    int box_height = wire.bottom - wire.top + 1;

    std::cout<<"mask box width: "<<box_width<<" "<<"mask box height: "<<box_height<<std::endl;
    std::cout<<"mask width: "<<wire.mask_width<<" "<<"mask height: "<<wire.mask_height<<std::endl;
    
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

    // 过滤一：计算导线端点连线，焊点到连线的垂线，判断所有端点是否都在垂线一侧，还是被垂线分开，被分开说明导线横跨焊点，不满足要求
    std::vector<cv::Point> endpoints_img;
    for (auto& ep : endpoints) {
        // 转换到图像坐标系
        cv::Point ep_img;
        ep_img.x = ep.x + wire.left;
        ep_img.y = ep.y + wire.top;
        endpoints_img.push_back(ep_img);
    }
    cv::Point solder_center(solder.center_x(), solder.center_y());
    float dist_thre = std::min(solder.width(), solder.height()) / 2.0f;
    if (isAcrossSolder(endpoints_img, solder_center, dist_thre))
    {
        std::cout<<"wire is across the solder! "<<" ep_img0: "<<endpoints_img[0].x<<" "<<endpoints_img[0].y<<" ep_img1: "<<endpoints_img[1].x<<" "<<endpoints_img[1].y<<std::endl;
        return false; // 导线横跨焊点
    }

    // ===== 过滤四：导线朝向是否向PCB中心
    float distA = cv::norm(cv::Point2f(endpoints_img[0]) - center);
    float distB = cv::norm(cv::Point2f(endpoints_img[1]) - center);
    float distS = cv::norm(cv::Point2f(solder_center) - center);
    if (distA > (distS+dist_thre) || distB > (distS+dist_thre))
    {
        std::cout<<"wire is far away from the PCB center! "<<std::endl;
        return false; // 导线远离PCB中点
    }

    for (auto& ep : endpoints) {
        // 转换到图像坐标系
        cv::Point ep_img;
        ep_img.x = ep.x + wire.left;
        ep_img.y = ep.y + wire.top;
        // ===== 过滤二：距离过滤 =====
        double dist = pointToBoxDistance(ep_img, solder);
        std::cout<<"solder: "<<solder.center_x()<<" "<<solder.center_y()<<" ep_img: "<<ep_img.x<<" "<<ep_img.y<<std::endl;
        std::cout<<"dist: "<<dist<<" dist thre: "<<std::min(min_dist, SOLDER_WIRE_DIST_THRE)<<std::endl;
        if (dist > std::min(min_dist, SOLDER_WIRE_DIST_THRE)) continue;//当两个焊点比较近，阈值要相应调整

        // ===== 过滤三：夹角方向过滤 =====
        // 线段一方向判断 端点指向导线切线方向
        float angle1 = computeDirectionFromEndpoint(bool_mask, ep);
        // 线段二方向判断 端点指向中心点
        // 计算线段夹角，判断是否满足阈值
        float dx = center.x - ep_img.x;
        float dy = center.y - ep_img.y;
        float angle2 = std::atan2(dy, dx) * 180.0f / CV_PI; // 转角度
        // 角度归一化
        auto normalizeAngle = [](float a) {
            while (a > 180) a -= 360;
            while (a < -180) a += 360;
            return a;
        };
        angle1 = normalizeAngle(angle1);
        angle2 = normalizeAngle(angle2);
        // 计算夹角（考虑方向无关）
        float diff = std::fabs(angle1 - angle2);
        diff = std::min(diff, 360.0f - diff); // 取最小夹角
        // 导线方向可能反向（180度问题）
        float diff_opposite = std::fabs(diff - 180.0f);
        float final_diff = std::min(diff, diff_opposite);
        std::cout<<"final_diff: "<<final_diff<<" angle thre: "<<angle_thre<<std::endl;
        if (final_diff > angle_thre) continue;
        return true;
    }

    return false;
}

// 通过焊点与导线切线距离判断是否关联(后面模板配对漏检时可以用上，专门分析漏掉的线)
bool isWireBelongToSolder_v3(
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
        // ===== 方向判断 =====方向判断留给对比模板有漏检时进行补救
        float angle = computeDirectionFromEndpoint(bool_mask, ep);
        // std::cout<<"angle: "<<angle<<std::endl;

        double pt_to_line_dist_thre = std::sqrt(solder.width() * solder.width() + solder.height() * solder.height()) / 2.0;

        if (isDirectionTowardsSolder(ep_img, angle, solder, pt_to_line_dist_thre)) {
            return true;
        }
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

cv::Rect get_intersection(const CheckBox& a, const CheckBox& b)
{
    int x1 = std::max(a.left, b.left);
    int y1 = std::max(a.top, b.top);
    int x2 = std::min(a.right, b.right);
    int y2 = std::min(a.bottom, b.bottom);

    if (x2 <= x1 || y2 <= y1)
        return cv::Rect();

    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

float compute_mask_iou_strict(const CheckBox& a, const CheckBox& b)
{
    cv::Rect inter = get_intersection(a, b);
    if (inter.area() == 0) return 0.0f;

    // resize
    cv::Mat mask_a(a.mask_height, a.mask_width, CV_32F, (void*)a.mask.data());
    cv::Mat mask_b(b.mask_height, b.mask_width, CV_32F, (void*)b.mask.data());

    mask_a = mask_a.clone();
    mask_b = mask_b.clone();

    cv::Mat ra, rb;
    cv::resize(mask_a, ra, cv::Size(a.width(), a.height()), 0, 0, cv::INTER_NEAREST);
    cv::resize(mask_b, rb, cv::Size(b.width(), b.height()), 0, 0, cv::INTER_NEAREST);

    // 👉 1. 计算各自mask面积
    float area_a = 0, area_b = 0;

    for (int y = 0; y < ra.rows; ++y)
    {
        const float* pa = ra.ptr<float>(y);
        for (int x = 0; x < ra.cols; ++x)
            if (pa[x] > 0.5f) area_a++;
    }

    for (int y = 0; y < rb.rows; ++y)
    {
        const float* pb = rb.ptr<float>(y);
        for (int x = 0; x < rb.cols; ++x)
            if (pb[x] > 0.5f) area_b++;
    }

    // 👉 2. 计算交集（只在ROI）
    float inter_cnt = 0;

    cv::Rect roi_a(inter.x - a.left, inter.y - a.top, inter.width, inter.height);
    cv::Rect roi_b(inter.x - b.left, inter.y - b.top, inter.width, inter.height);

    cv::Mat sub_a = ra(roi_a);
    cv::Mat sub_b = rb(roi_b);

    for (int y = 0; y < sub_a.rows; ++y)
    {
        const float* pa = sub_a.ptr<float>(y);
        const float* pb = sub_b.ptr<float>(y);

        for (int x = 0; x < sub_a.cols; ++x)
        {
            if (pa[x] > 0.5f && pb[x] > 0.5f)
                inter_cnt++;
        }
    }

    // float union_cnt = area_a + area_b - inter_cnt;
    float union_cnt = std::min(area_a, area_b);

    // if (a.left == 2157 && a.top == 2136 && b.left == 2154 && b.top == 2138)
    // {
    //     cv::imwrite("a.jpg", ra*255);
    //     cv::imwrite("b.jpg", rb*255);
    //     std::cout<<"area_a: "<<area_a<<" area_b: "<<area_b<<" inter_cnt: "<<inter_cnt<<" union_cnt: "<<union_cnt<<std::endl;
    // }

    if (union_cnt < 1e-6) return 0.0f;

    return inter_cnt / union_cnt;
}

std::vector<CheckBox> mask_nms(std::vector<CheckBox>& boxes, float iou_thresh)
{
    std::sort(boxes.begin(), boxes.end(),
              [](const CheckBox& a, const CheckBox& b) {
                    return a.score > b.score;//采用得分进行排序
                // return a.width()*a.height() > b.width()*b.height();//采用面积进行排序
              });

    std::vector<CheckBox> result;
    std::vector<bool> removed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (removed[i]) continue;

        result.push_back(boxes[i]);

        std::cout<<"box i: "<<boxes[i].left<<" "<<boxes[i].top<<" "<<boxes[i].score<<std::endl;

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (removed[j]) continue;

            // if (boxes[i].cls != boxes[j].cls) continue;不同类别也要互相进行非极大值抑制

            float mask_iou = compute_mask_iou_strict(boxes[i], boxes[j]);

                    std::cout<<"box j: "<<boxes[j].left<<" "<<boxes[j].top<<" "<<boxes[j].score<<" "<<mask_iou<<std::endl;


            if (mask_iou > iou_thresh)
                removed[j] = true;
        }
    }

    return result;
}

cv::Scalar computeDominantColor(
    const cv::Mat& img_roi,
    const cv::Mat& mask_roi,
    int step = 16) // 量化步长（推荐16或32）
{
    std::unordered_map<int, int> color_count;

    int rows = img_roi.rows;
    int cols = img_roi.cols;

    int max_count = 0;
    int best_key = 0;

    for (int y = 0; y < rows; ++y)
    {
        const uchar* m_ptr = mask_roi.ptr<uchar>(y);
        const cv::Vec3b* p_ptr = img_roi.ptr<cv::Vec3b>(y);

        for (int x = 0; x < cols; ++x)
        {
            if (m_ptr[x] == 0) continue;

            // ===== 量化颜色 =====
            int b = p_ptr[x][0] / step;
            int g = p_ptr[x][1] / step;
            int r = p_ptr[x][2] / step;

            int key = (b << 16) | (g << 8) | r;

            int cnt = ++color_count[key];

            if (cnt > max_count)
            {
                max_count = cnt;
                best_key = key;
            }
        }
    }

    // ===== 反量化（恢复颜色）=====
    int b = ((best_key >> 16) & 0xFF) * step + step / 2;
    int g = ((best_key >> 8)  & 0xFF) * step + step / 2;
    int r = ( best_key        & 0xFF) * step + step / 2;

    return cv::Scalar(b, g, r);
}

cv::Scalar computeMeanColor(
    const cv::Mat& image,   // 原图（BGR）
    const CheckBox& wire)
{

    // ===== mask转Mat =====
    cv::Mat float_mask(wire.mask_height, wire.mask_width, CV_32F,
                       const_cast<float*>(wire.mask.data()));

    // ===== resize到bbox大小 =====
    int w = wire.right - wire.left + 1;
    int h = wire.bottom - wire.top + 1;

    cv::Mat mask_resized;
    cv::resize(float_mask, mask_resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    // ===== 二值化 =====
    cv::Mat mask_bin;
    cv::threshold(mask_resized, mask_bin, 0.5, 255, cv::THRESH_BINARY);
    mask_bin.convertTo(mask_bin, CV_8U);

    // ===== ROI裁剪 =====
    cv::Rect roi(wire.left, wire.top, w, h);
    roi &= cv::Rect(0, 0, image.cols, image.rows);

    if (roi.width <= 0 || roi.height <= 0)
        return cv::Scalar(0,0,0);

    cv::Mat img_roi = image(roi);

    // mask也裁剪一致区域
    cv::Mat mask_roi = mask_bin(
        cv::Rect(roi.x - wire.left, roi.y - wire.top, roi.width, roi.height)
    );

    // // ===== 计算均值（只统计mask区域）=====
    // cv::Scalar mean_color = cv::mean(img_roi, mask_eroded);
    // ===== 统计颜色众数 =====
    cv::Scalar dominant_color = computeDominantColor(img_roi, mask_roi);

    return dominant_color; // BGR
}

void computeWiresColor(
    const cv::Mat& image,   // 原图（BGR）
    std::vector<CheckBox>& wires)
{
    for (int i = 0; i < wires.size(); i ++)
    {
        cv::Scalar color = computeMeanColor(image, wires[i]);
        wires[i].color = color;
        std::cout<<"color: "<<i<<" "<<wires[i].color<<std::endl;
    }
}

void matchWiresToSolder(
    const cv::Mat image,
    const std::vector<CheckBox>& solderings,
    const std::vector<std::vector<CheckBox>>& all_segmentations,
    std::vector<SolderWire> &results,
    const cv::Point2f center,
    const bool mode)//pcb中心点
{
    std::cout<<"solderings size: "<<solderings.size()<<std::endl;
    for (size_t i = 0; i < solderings.size(); ++i) {

        SolderWire solderwire;
        const auto& solder = solderings[i];
        //计算当前soldering与其他soldering的最小距离，作为后续端点到焊点距离判断的依据之一
        float min_dist = minCenterDistance(solderings, i);
        solderwire.solder_info = solder;
        const auto& wires = all_segmentations[i];
        std::vector<CheckBox> temp_wires;

        std::cout<<"===========================================solder index: "<<i<<" wires size: "<<wires.size()<<std::endl;
        for (size_t j = 0; j < wires.size(); ++j) {
            std::cout<<"========= process wire "<<j<<" =========="<<std::endl;
            if (isWireBelongToSolder_v2(solder, wires[j], center, min_dist, 45)) {
                std::cout<<"========= candidate success! =========="<<std::endl;
                // solderwire.wires.push_back(wires[j]);
                temp_wires.push_back(wires[j]);
            }
        }

        if (temp_wires.size() > 1)//如果当前焊点不止一个候选导线，需要判断是否去重
        {
            // 进行掩膜nms
            std::vector<CheckBox> temp_wires_nms = mask_nms(temp_wires, 0.45);
            solderwire.wires = temp_wires_nms;
        }
        else
        {
            solderwire.wires = temp_wires;
        }

        if (0 == mode && 0 == solderwire.wires.size())//标定流程空焊点跳过，检测流程保留，后续二次找回
        {
            continue;//空焊点跳过
        }

        //计算导线内部RGB，用于漏检时进行传统颜色比对
        computeWiresColor(image, solderwire.wires);
        
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

// bool existColorInCircle(
//     const cv::Mat& image,      // 原图（BGR）
//     const cv::Point& center,   // 焊点中心
//     int max_radius,                // 最大搜索半径
//     int min_radius,                // 最小搜索半径
//     const cv::Scalar& target_color, // 目标颜色（BGR）
//     int color_tol = 10,        // 颜色容忍
//     int min_count = 20)        // 最少像素数
// {
//     int rows = image.rows;
//     int cols = image.cols;

//     int count = 0;

//     // 限定ROI范围（加速）
//     int x0 = std::max(0, center.x - radius);
//     int x1 = std::min(cols - 1, center.x + radius);
//     int y0 = std::max(0, center.y - radius);
//     int y1 = std::min(rows - 1, center.y + radius);

//     for (int y = y0; y <= y1; ++y)
//     {
//         for (int x = x0; x <= x1; ++x)
//         {
//             // 圆形约束
//             int dx = x - center.x;
//             int dy = y - center.y;

//             if (dx * dx + dy * dy > max_radius * max_radius)
//                 continue;
//             if (dx * dx + dy * dy < min_radius * min_radius)
//                 continue;

//             const cv::Vec3b& pix = image.at<cv::Vec3b>(y, x);

//             // BGR
//             if (std::abs(pix[0] - target_color[0]) < color_tol &&
//                 std::abs(pix[1] - target_color[1]) < color_tol &&
//                 std::abs(pix[2] - target_color[2]) < color_tol)
//             {
//                 count++;

//                 if (count > min_count)
//                     return true;
//             }
//         }
//     }

//     return false;
// }

bool existColorInCircle(
    const cv::Mat& image,
    const cv::Point& center,
    int max_radius,
    int min_radius,
    const cv::Scalar& target_color,
    CheckBox& out_box,          // 输出
    int color_tol = 10,
    int min_count = 20)
{
    int rows = image.rows;
    int cols = image.cols;

    // 用mask记录命中像素（全图）
    cv::Mat hit_mask = cv::Mat::zeros(rows, cols, CV_8U);

    int count = 0;

    int x0 = std::max(0, center.x - max_radius);
    int x1 = std::min(cols - 1, center.x + max_radius);
    int y0 = std::max(0, center.y - max_radius);
    int y1 = std::min(rows - 1, center.y + max_radius);

    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            int dx = x - center.x;
            int dy = y - center.y;

            int dist2 = dx * dx + dy * dy;

            if (dist2 > max_radius * max_radius) continue;
            if (dist2 < min_radius * min_radius) continue;

            const cv::Vec3b& pix = image.at<cv::Vec3b>(y, x);

            if (std::abs(pix[0] - target_color[0]) < color_tol &&
                std::abs(pix[1] - target_color[1]) < color_tol &&
                std::abs(pix[2] - target_color[2]) < color_tol)
            {
                hit_mask.at<uchar>(y, x) = 255;
                count++;
            }
        }
    }

    // // ===== 计算 bounding box =====
    // std::vector<cv::Point> pts;
    // cv::findNonZero(hit_mask, pts);

    // if (pts.empty()) return false;

    // cv::Rect rect = cv::boundingRect(pts);

    // ===== 连通域分析 =====
    cv::Mat labels, stats, centroids;
    int num = cv::connectedComponentsWithStats(hit_mask, labels, stats, centroids, 8);

    if (num <= 1) return false; // 只有背景

    // ===== 找最大连通域 =====
    int max_idx = -1;
    int max_area = 0;

    for (int i = 1; i < num; ++i) // 0是背景
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area > max_area)
        {
            max_area = area;
            max_idx = i;
        }
    }

    std::cout<<"target color: "<<target_color<<" roi: "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<" hit number: "<<max_area<<std::endl;

    if (max_idx < 0 || max_area < min_count)
        return false;

    // ===== 生成最大连通域mask =====
    cv::Mat largest_mask = (labels == max_idx);
    largest_mask.convertTo(largest_mask, CV_8U, 255);

    // ===== 计算 bounding box =====
    cv::Rect rect = cv::Rect(
        stats.at<int>(max_idx, cv::CC_STAT_LEFT),
        stats.at<int>(max_idx, cv::CC_STAT_TOP),
        stats.at<int>(max_idx, cv::CC_STAT_WIDTH),
        stats.at<int>(max_idx, cv::CC_STAT_HEIGHT)
    );

    // ===== 裁剪mask =====
    cv::Mat roi_mask = hit_mask(rect).clone();
    // cv::imwrite("find_color.jpg", roi_mask);

    // ===== resize到固定mask尺寸 =====
    cv::Mat resized_mask;
    cv::resize(roi_mask, resized_mask,
               cv::Size(out_box.mask_width, out_box.mask_height),
               0, 0, cv::INTER_NEAREST);

    resized_mask.convertTo(resized_mask, CV_32F, 1.0 / 255);

    // ===== 填充CheckBox =====
    out_box.left   = rect.x;
    out_box.top    = rect.y;
    out_box.right  = rect.x + rect.width;
    out_box.bottom = rect.y + rect.height;

    out_box.mask.assign((float*)resized_mask.datastart,
                        (float*)resized_mask.dataend);

    return true;
}

void secondCheckBasedTruth(
    const cv::Mat image,
    const SolderWire& truth_info,
    SolderWire& result_info)
{
    // 统计第一个集合中各颜色的数量
    std::map<int, int> color_count1;
    for (const auto& wire : truth_info.wires) {
        color_count1[wire.cls]++;
    }
    
    // 统计第二个集合中各颜色的数量
    std::map<int, int> color_count2;
    for (const auto& wire : result_info.wires) {
        color_count2[wire.cls]++;
    }

    for (const auto& kv : color_count1)
    {
        int cls = kv.first;
        int count1 = kv.second;

        int count2 = 0;
        if (color_count2.count(cls))
            count2 = color_count2.at(cls);

        if (count1 == 1 && count2 == 0)
        {
            cv::Scalar color;
            for (const auto& wire : truth_info.wires) {
                if (wire.cls == cls)
                {
                    color = wire.color;
                    std::cout<<"go into here cls: "<<cls<< " color :"<<color<<std::endl;
                    break;
                }
            }
            //以检测板焊点为圆心，查看半径100范围内是否存在color的像素，假定各个颜色通道浮动范围为10
            cv::Point solder_center(
                result_info.solder_info.center_x(),
                result_info.solder_info.center_y());

            CheckBox temp;
            temp.cls = cls;
            temp.mask_height = 160;
            temp.mask_width = 160;
            bool found = existColorInCircle(
                image,              // 原图
                solder_center,
                100,                // 半径
                20,
                color,
                temp,
                20,                 // 颜色容忍
                20);                // 最少像素

            //如果满足条件，则
            if (found)
            {
                result_info.wires.push_back(temp);
            }
        }
    }
}

std::vector<MatchPair> matchSolders(
    const cv::Mat image,
    const cv::Mat& template_img,
    const std::vector<SolderWire>& truth,
    std::vector<SolderWire>& results,
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

            // 二次校验，如果出现了检测板导线漏检，则分析焊点周边区域颜色进行找回
            // 检测板某焊点颜色导线数量漏检为0，实际为1，其他错误情况暂时不通过校验来挽救，仍通过提高模型能力解决
            if (false == wire_count_match || false == wire_color_match)
            {
                std::cout<<"start process second find!"<<std::endl;
                secondCheckBasedTruth(image, truth[i], results[best_j]);
            }

            std::cout<<"det wire size: "<<results[best_j].wires.size()<<" temp wire size: "<<truth[i].wires.size()<<std::endl;
            std::cout<<"det color: "<<std::endl;
            for (int index_det = 0; index_det<results[best_j].wires.size(); index_det++)
            {
                std::cout<<"det color: "<<results[best_j].wires[index_det].cls<<std::endl;
            }
            std::cout<<"temp color: "<<std::endl;
            for (int index_det = 0; index_det<truth[i].wires.size(); index_det++)
            {
                std::cout<<"temp color: "<<truth[i].wires[index_det].cls<<std::endl;
            }

            // 再次刷新状态
            wire_count_match = false;
            wire_color_match = false;
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