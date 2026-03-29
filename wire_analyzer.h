/**
 * @file wire_analyzer.h
 * @brief 导线分析模块 - 判断导线是否与焊点关联
 * @date 2025-06-07
 */
#ifndef WIRE_ANALYZER_H
#define WIRE_ANALYZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include "checkbox.h"
#include "CTempMatch.h"
#include <limits>

#define SOLDER_WIRE_DIST_THRE 60.0f
#define MATCH_SOLDERS_IOU_THRE 0.2

// 焊点导线组合信息
struct SolderWire {
    CheckBox solder_info;         // 焊点坐标信息
    std::vector<CheckBox> wires;  // 对应焊线信息
};

// 模板和检测板配对信息
struct MatchPair {
    int truth_idx;  // -1表示不存在，代表检测板多检测了焊点，>=0为模板索引号
    int result_idx; // -1表示不存在，代表检测板漏检测了焊点，>=0为检测板索引号
    bool wire_count_match = false;    // 0表示数量与模板一致，1表示不一致
    bool wire_color_match = false;    // 0表示数量与模板一致，1表示不一致
};

// 统计检测板的检测结果
// solderings: 焊点检测结果
// all_segmentations: 各个焊点附近区域导线分割结果
// results: 当前板焊点导线组合信息
void matchWiresToSolder(
    const std::vector<CheckBox>& solderings,
    const std::vector<std::vector<CheckBox>>& all_segmentations,
    std::vector<SolderWire> &results);

// 结合模板结果分析当前检测板的检测结果(可选)
// img: 图像
// truth: 模板结果，需进行匹配校正
// params: 模板匹配参数
// results: 检测板结果
void analyseBasedTruth(
    const cv::Mat& img,
    const std::vector<SolderWire>& truth,//模板结果，需进行匹配校正
    const TemplateMatchResult& params,//模板匹配参数
    std::vector<SolderWire>& results);

// 根据模板结果和检测板结果，给出配对情况
// template_img: 模板图像
// truth: 模板结果，需进行匹配校正
// results: 检测板结果
// params: 模板匹配参数
// iou_thresh: 焊点配对iou阈值
std::vector<MatchPair> matchSolders(
    const cv::Mat& template_img,
    const std::vector<SolderWire>& truth,
    const std::vector<SolderWire>& results,
    const TemplateMatchResult& params,
    float iou_thresh);

#endif // WIRE_ANALYZER_H