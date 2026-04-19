#include <iostream>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "vision_detect.h"
#include "vision_segment.h"
#include "wire_analyzer.h"
#include "save_result_json.h"
#include "CTempMatch.h"

namespace fs = std::filesystem;
// 颜色调色板（按照指定顺序：blue, black, green, white, red, gray, yellow, orange, pink）
const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255, 0, 0),      // blue   - 蓝色 (BGR中蓝色是(255,0,0))
    cv::Scalar(0, 0, 0),        // black  - 黑色 (BGR中黑色是(0,0,0))
    cv::Scalar(0, 255, 0),      // green  - 绿色 (BGR中绿色是(0,255,0))
    cv::Scalar(255, 255, 255),  // white  - 白色 (BGR中白色是(255,255,255))
    cv::Scalar(0, 0, 255),      // red    - 红色 (BGR中红色是(0,0,255))
    cv::Scalar(128, 128, 128),  // gray   - 灰色 (BGR中灰色是(128,128,128))
    cv::Scalar(0, 255, 255),    // yellow - 黄色 (BGR中黄色是(0,255,255))
    cv::Scalar(0, 165, 255),    // orange - 橙色 (BGR中橙色是(0,165,255))
    cv::Scalar(147, 20, 255)    // pink   - 粉色 (BGR中粉色是(147,20,255))
};
// 画检测框
void DrawBoxes(cv::Mat& img, const std::vector<CheckBox>& boxes, const std::string& path, const std::string& output_path)
{
    for (const auto& box : boxes)
    {
        std::cout<<"x1: "<<box.left<<" y1: "<<box.top<<" x2: "<<box.right<<" y2: "<<box.bottom<<std::endl;
        cv::rectangle(
            img,
            cv::Point(box.left, box.top),
            cv::Point(box.right, box.bottom),
            cv::Scalar(0, 0, 255),
            2
        );

        std::string label =
            "cls:" + std::to_string(box.cls) +
            " " +
            cv::format("%.2f", box.score);

        int baseline = 0;

        auto size = cv::getTextSize(
            label,
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
            &baseline
        );

        cv::rectangle(
            img,
            cv::Point(box.left, box.top - size.height),
            cv::Point(box.left + size.width, box.top),
            cv::Scalar(0, 0, 255),
            -1
        );

        cv::putText(
            img,
            label,
            cv::Point(box.left, box.top),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255,255,255),
            1
        );
    }
    std::string out_name =output_path + "/det_result_" + fs::path(path).filename().string();
    cv::imwrite(out_name, img);
}

std::vector<std::string> loadLabels(const std::string& label_file) {
    std::vector<std::string> labels;
    
    // 如果文件路径为空，返回空向量
    if (label_file.empty()) {
        return labels;
    }
    
    // 打开文件
    std::ifstream file(label_file);
    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open labels file: " << label_file << std::endl;
        return labels;
    }

    // 逐行读取标签
    std::string label;
    while (std::getline(file, label)) {
        // 跳过空行
        if (!label.empty()) {
            labels.push_back(label);
        }
    }
    
    file.close();
    
    std::cout << "Loaded " << labels.size() << " labels from: " << label_file << std::endl;
    return labels;
}

void drawMask(cv::Mat& image,const CheckBox box, const cv::Scalar color)
{
    // 绘制分割掩码 - 使用OpenCV操作
    if (!box.mask.empty() && box.mask_width > 0 && box.mask_height > 0) {
        // 创建掩码矩阵
        cv::Mat float_mask(box.mask_height, box.mask_width, CV_32FC1, 
                        const_cast<float*>(box.mask.data()));
        
        // 计算边界框在原图中的位置
        int box_width = box.right - box.left + 1;
        int box_height = box.bottom - box.top + 1;
        
        if (box_width <= 0 || box_height <= 0) return;
        
        // 调整掩码大小到边界框尺寸
        cv::Mat resized_mask;
        cv::resize(float_mask, resized_mask, cv::Size(box_width, box_height), 
                0, 0, cv::INTER_LINEAR);
        
        // 二值化掩码并转换为8位
        cv::Mat bool_mask;
        cv::threshold(resized_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);
        bool_mask.convertTo(bool_mask, CV_8UC1);

        // 创建彩色掩码
        cv::Mat color_mask(box_height, box_width, CV_8UC3, color);
        
        cv::Rect valid_rect(box.left, box.top, box_width, box_height);
        
        // 获取掩码的有效部分
        cv::Mat roi_region = image(valid_rect);
        
        // 应用掩码
        for (int y = 0; y < bool_mask.rows; ++y) {
            for (int x = 0; x < bool_mask.cols; ++x) {
                if (bool_mask.at<uchar>(y, x) > 0) {
                    cv::Vec3b& pixel = roi_region.at<cv::Vec3b>(y, x);
                    cv::Vec3b mask_color = color_mask.at<cv::Vec3b>(y, x);
                    
                    pixel = cv::Vec3b(
                        pixel[0] * 0.6 + mask_color[0] * 0.4,
                        pixel[1] * 0.6 + mask_color[1] * 0.4,
                        pixel[2] * 0.6 + mask_color[2] * 0.4
                    );
                }
            }
        }
        
    }
}

void visualizeSegmentationROI(
    const cv::Mat& image,
    const std::vector<CheckBox>& det_boxes,
    const std::vector<std::vector<CheckBox>>& all_segmentations,
    const std::string& path,
    const std::string& output_path,
    int target_size,
    const std::vector<std::string>& labels = {})
{
    int image_width = image.cols;
    int image_height = image.rows;
    for (size_t det_idx = 0; det_idx < det_boxes.size(); ++det_idx) {
        const auto& det = det_boxes[det_idx];

        int center_x = det.center_x();
        int center_y = det.center_y();
        
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
    
        cv::Mat roi = image(cv::Rect(crop_x1, crop_y1, target_size, target_size)).clone();
        const auto& seg_boxes = all_segmentations[det_idx];
        std::cout<<"seg_boxes number: "<<seg_boxes.size()<<std::endl;
        
        for (size_t seg_idx = 0; seg_idx < seg_boxes.size(); ++seg_idx) {
            auto box = seg_boxes.at(seg_idx);
            cv::Scalar color = COLORS[box.cls % COLORS.size()];
            cv::rectangle(roi,
                        cv::Point(box.left - crop_x1, box.top - crop_y1),
                        cv::Point(box.right - crop_x1, box.bottom - crop_y1),
                        color, 2);

            // ---- label ----
            std::string label_text;
            if (!labels.empty() && box.cls < (int)labels.size()) {
                label_text = labels[box.cls] + " " + cv::format("%.2f", box.score);
            } else {
                label_text = cv::format("cls:%d %.2f", box.cls, box.score);
            }

            cv::putText(roi, label_text,
                        cv::Point(box.left - crop_x1, std::max(0, box.top - crop_y1 - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

            // 绘制分割掩码 - 使用OpenCV操作
            if (!box.mask.empty() && box.mask_width > 0 && box.mask_height > 0) {
                // 创建掩码矩阵
                cv::Mat float_mask(box.mask_height, box.mask_width, CV_32FC1, 
                                const_cast<float*>(box.mask.data()));
                
                // 计算边界框在原图中的位置
                int box_width = box.right - box.left + 1;
                int box_height = box.bottom - box.top + 1;
                
                if (box_width <= 0 || box_height <= 0) continue;
                
                // 调整掩码大小到边界框尺寸
                cv::Mat resized_mask;
                cv::resize(float_mask, resized_mask, cv::Size(box_width, box_height), 
                        0, 0, cv::INTER_LINEAR);
                
                // 二值化掩码并转换为8位
                cv::Mat bool_mask;
                cv::threshold(resized_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);
                bool_mask.convertTo(bool_mask, CV_8UC1);

                // cv::imwrite(output_path + "/mask_" + std::to_string(det_idx) + "_" + std::to_string(seg_idx) + ".jpg", bool_mask);
                // std::vector<cv::Point> end_pt = findEndpoints(bool_mask);
                // std::cout<<det_idx<<" "<<seg_idx<<" pt1: "<<end_pt.at(0)<<" pt2: "<<end_pt.at(1)<<std::endl;

                
                // 创建彩色掩码
                cv::Mat color_mask(box_height, box_width, CV_8UC3, color);
                
                // 创建掩码区域（只处理在roi内的部分）
                cv::Rect box_rect(box.left - crop_x1, box.top - crop_y1, box_width, box_height);
                cv::Rect roi_rect(0, 0, roi.cols, roi.rows);
                cv::Rect valid_rect = box_rect & roi_rect;  // 交集
                
                if (!valid_rect.empty()) {
                    // 计算在掩码中的对应区域
                    cv::Rect mask_rect(
                        valid_rect.x - box_rect.x,
                        valid_rect.y - box_rect.y,
                        valid_rect.width,
                        valid_rect.height
                    );
                    
                    // 获取掩码的有效部分
                    cv::Mat valid_mask = bool_mask(mask_rect);
                    cv::Mat valid_color = color_mask(mask_rect);
                    cv::Mat roi_region = roi(valid_rect);
                    
                    // 应用掩码
                    for (int y = 0; y < valid_mask.rows; ++y) {
                        for (int x = 0; x < valid_mask.cols; ++x) {
                            if (valid_mask.at<uchar>(y, x) > 0) {
                                cv::Vec3b& pixel = roi_region.at<cv::Vec3b>(y, x);
                                cv::Vec3b mask_color = valid_color.at<cv::Vec3b>(y, x);
                                
                                pixel = cv::Vec3b(
                                    pixel[0] * 0.6 + mask_color[0] * 0.4,
                                    pixel[1] * 0.6 + mask_color[1] * 0.4,
                                    pixel[2] * 0.6 + mask_color[2] * 0.4
                                );
                            }
                        }
                    }
                }
            }
        }

        std::string save_path = output_path + "/segment_" + std::to_string(det_idx) + "_" + fs::path(path).filename().string();
        cv::imwrite(save_path, roi);
    }
}

void visualizeFinalResult(
    const cv::Mat image,
    const std::vector<SolderWire> truths,
    const std::vector<SolderWire> results,
    const std::vector<MatchPair> match_results,
    const std::string output_path,
    const std::string img_name)
{
    cv::Mat draw_img = image.clone();

    // 不直接渲染模型输出结果，而是渲染后处理结果
    // for (size_t det_idx = 0; det_idx < results.size(); ++det_idx)
    // {
    //     cv::rectangle(draw_img,
    //                     cv::Point(results[det_idx].solder_info.left, results[det_idx].solder_info.top),
    //                     cv::Point(results[det_idx].solder_info.right, results[det_idx].solder_info.bottom),
    //                     cv::Scalar(0, 0, 255), 2);
    //     for (size_t seg_idx = 0; seg_idx < results[det_idx].wires.size(); ++seg_idx)
    //     {
    //         cv::Scalar color = COLORS[results[det_idx].wires[seg_idx].cls % COLORS.size()];
    //         cv::rectangle(draw_img,
    //                     cv::Point(results[det_idx].wires[seg_idx].left, results[det_idx].wires[seg_idx].top),
    //                     cv::Point(results[det_idx].wires[seg_idx].right, results[det_idx].wires[seg_idx].bottom),
    //                     color, 2);
    //         drawMask(draw_img, results[det_idx].wires[seg_idx], color);
    //     }
    // }

    for (size_t det_idx = 0; det_idx < match_results.size(); ++det_idx)
    {
        auto match = match_results[det_idx];

        cv::rectangle(draw_img,
                cv::Point(results[match.result_idx].solder_info.left, results[match.result_idx].solder_info.top),
                cv::Point(results[match.result_idx].solder_info.right, results[match.result_idx].solder_info.bottom),
                cv::Scalar(0, 0, 255), 2);
        for (size_t seg_idx = 0; seg_idx < results[match.result_idx].wires.size(); ++seg_idx)
        {
            cv::Scalar color = COLORS[results[match.result_idx].wires[seg_idx].cls % COLORS.size()];
            cv::rectangle(draw_img,
                        cv::Point(results[match.result_idx].wires[seg_idx].left, results[match.result_idx].wires[seg_idx].top),
                        cv::Point(results[match.result_idx].wires[seg_idx].right, results[match.result_idx].wires[seg_idx].bottom),
                        color, 2);
            drawMask(draw_img, results[match.result_idx].wires[seg_idx], color);
        }

        if (match.truth_idx != -1 && match.result_idx == -1)//检测板漏检
        {
            //在图片左上角打印漏掉信息
            // 在左上角写文字
            std:string label = "none exist soldering: "+std::to_string(truths[match.truth_idx].solder_info.left)+" " \
            +std::to_string(truths[match.truth_idx].solder_info.top)+" " \
            +std::to_string(truths[match.truth_idx].solder_info.right)+" " \
            +std::to_string(truths[match.truth_idx].solder_info.bottom);
            cv::putText(draw_img,
                        label,          // 文本
                        cv::Point(10, 60*det_idx),      // 位置（左上角偏移）
                        cv::FONT_HERSHEY_SIMPLEX, // 字体
                        1.0,                    // 字体大小
                        cv::Scalar(0, 0, 255),  // 颜色（BGR）
                        2);                     // 线宽
        }
        else if(match.truth_idx != -1 && match.result_idx != -1)
        {
            std::string label1;
            cv::Scalar color = cv::Scalar(0, 255, 0);
            if (match.wire_count_match)
            {
                label1 = "";
            }
            else{
                label1 = "wire count match failed!";
                color = cv::Scalar(0, 0, 255);
            }
            std::string label2;
            if (match.wire_color_match)
            {
                label2 = "";
            }
            else{
                label2 = "wire color match failed!";
                color = cv::Scalar(0, 0, 255);
            }
            cv::putText(draw_img,
                        label1+label2,          // 文本
                        cv::Point(results[match.result_idx].solder_info.left, results[match.result_idx].solder_info.top),      // 位置（左上角偏移）
                        cv::FONT_HERSHEY_SIMPLEX, // 字体
                        1.0,                    // 字体大小
                        color,  // 颜色（BGR）
                        2);   
        }
        else if (match.truth_idx == -1 && match.result_idx != -1)//检测板多检
        {
            std::string label;
            label = "error soldering";
            cv::putText(draw_img,
                        label,          // 文本
                        cv::Point(results[match.result_idx].solder_info.left, results[match.result_idx].solder_info.top),      // 位置（左上角偏移）
                        cv::FONT_HERSHEY_SIMPLEX, // 字体
                        1.0,                    // 字体大小
                        cv::Scalar(0, 0, 255),  // 颜色（BGR）
                        2);  
        }
    }
    std::string save_path = output_path + "/final_" + img_name;
    cv::imwrite(save_path, draw_img);
}

void visualizeMatchResult(
    const cv::Mat image,
    const std::vector<SolderWire> results,
    const std::string output_path,
    const std::string img_name)
{
    cv::Mat draw_img = image.clone();

    for (size_t det_idx = 0; det_idx < results.size(); ++det_idx)
    {
        cv::rectangle(draw_img,
                        cv::Point(results[det_idx].solder_info.left, results[det_idx].solder_info.top),
                        cv::Point(results[det_idx].solder_info.right, results[det_idx].solder_info.bottom),
                        cv::Scalar(0, 0, 255), 2);
        for (size_t seg_idx = 0; seg_idx < results[det_idx].wires.size(); ++seg_idx)
        {
            cv::Scalar color = COLORS[results[det_idx].wires[seg_idx].cls % COLORS.size()];
            cv::rectangle(draw_img,
                        cv::Point(results[det_idx].wires[seg_idx].left, results[det_idx].wires[seg_idx].top),
                        cv::Point(results[det_idx].wires[seg_idx].right, results[det_idx].wires[seg_idx].bottom),
                        color, 2);
            drawMask(draw_img, results[det_idx].wires[seg_idx], color);
        }
    }
    std::string save_path = output_path + "/" + img_name + ".jpg";
    cv::imwrite(save_path, draw_img);
}

// 获取文件夹图像
std::vector<std::string> GetImages(const std::string& path)
{
    std::vector<std::string> files;

    for (auto& p : fs::directory_iterator(path))
    {
        auto ext = p.path().extension().string();

        if (ext==".jpg" || ext==".png" || ext==".bmp" || ext==".jpeg")
            files.push_back(p.path().string());
    }

    return files;
}


int main(int argc, char** argv)
{
    if (argc < 11)
    {
        std::cout << "Usage:\n";
        std::cout << "./detect det_engine_path seg_engine_path reference_image input_path output_path label_path mode top left width height\n";
        return -1;
    }

    std::string det_engine_path = argv[1];
    std::string seg_engine_path = argv[2];
    std::string reference_image = argv[3];
    std::string input_path = argv[4];
    std::string output_path = argv[5];
    std::string label_path = argv[6];
    std::string mode = argv[7];// 0: pcb板标定流程 1: pcb板检测流程
    int template_left = std::stoi(argv[8]);
    int template_top = std::stoi(argv[9]);
    int template_width = std::stoi(argv[10]);
    int template_height = std::stoi(argv[11]);

    if (0 == std::stoi(mode))
    {
        // 创建输出目录
        if (!fs::exists(output_path)) {
            fs::create_directories(output_path);
        }

        // 创建模板图像，需要手动修改roi
        cv::Mat img;
        if (fs::is_regular_file(reference_image))
        {
            img = cv::imread(reference_image);
            if (img.empty())
            {
                std::cout << "read image failed\n";
                return -1;
            }
            cv::Mat roi = img(cv::Rect(template_left, template_top, template_width, template_height)).clone();
            cv::imwrite(output_path+"/template.jpg", roi);
        }
        else
        {
            std::cout << "reference_image path not valid!" << std::endl;
            return -1;
        }

        // 加载标签
        auto labels = loadLabels(label_path);
        if (!labels.empty()) {
            std::cout << "Loaded " << labels.size() << " labels" << std::endl;
        }

        VisionDetect detector;
        if (!detector.Init(det_engine_path))
        {
            std::cout << "Init detect model failed\n";
            return -1;
        }

        VisionSegment segmentor;
        if (!segmentor.Init(seg_engine_path))
        {
            std::cout << "Init segment model failed\n";
            return -1;
        }

        std::cout << "Model loaded success\n";

        uint64_t id_base = 0;

        //检测焊点
        std::vector<CheckBox> solderings;
        detector.Run(
            img,
            0.25,       // conf_thresh
            solderings,
            id_base,
            false
        );

        // debug和可视化
        std::cout << "detections: " << solderings.size() << std::endl;
        // 检测焊点结果可视化图像
        cv::Mat viz_img = img.clone();
        DrawBoxes(viz_img, solderings, reference_image, output_path);

        // 对每个检测框进行分割
        std::vector<std::vector<CheckBox>> all_segmentations;
        int crop_size = 384;//暂定384 320
        if (!solderings.empty()) {
            segmentor.RunWithBoxes(
                img,
                solderings,
                0.25,       // conf_thresh
                all_segmentations,
                id_base,
                crop_size        // crop_size
            );

            // 焊点导线分割结果可视化
            viz_img = img.clone();
            visualizeSegmentationROI(viz_img, solderings, all_segmentations, reference_image, output_path, crop_size, labels);
        }

        // 联系焊点和导线从属关系
        std::vector<SolderWire> results;
        matchWiresToSolder(img, solderings, all_segmentations, results, cv::Point2f(template_left+template_width*0.5f, template_top+template_height*0.5f), 0);
        // 匹配结果可视化
        visualizeMatchResult(img, results, output_path, "match_calibrate");

        // 保存模板结果
        save_results_binary(results, output_path+"/template_results.txt");
    }
    else{
        // 加载模板图片
        auto template_path = output_path+"/template.jpg";
        cv::Mat template_img;
        if (fs::is_regular_file(template_path))
        {
            template_img = cv::imread(template_path);
            if (template_img.empty())
            {
                std::cout << "read template image failed\n";
                return -1;
            }
        }
        else
        {
            std::cout << "template path not valid!" << std::endl;
            return -1;
        }

        // 加载模板检测结果
        auto json_path = output_path+"/template_results.txt";
        if (!fs::is_regular_file(json_path))
        {
            std::cout << "json path not valid!" << std::endl;
            return -1;
        }
        std::vector<SolderWire> reference_results;
        int ret = load_results_binary(json_path, reference_results);

        // // 模板检测结果加载后的可视化
        // auto reference_img = cv::imread(reference_image);
        // visualizeMatchResult(reference_img, reference_results, output_path, "load_template");

        // 将模板检测结果从原图原点坐标转换到模板坐标
        covert2template(reference_results, template_left, template_top);

        // 创建输出目录
        if (!fs::exists(output_path)) {
            fs::create_directories(output_path);
        }

        // 加载标签
        auto labels = loadLabels(label_path);
        if (!labels.empty()) {
            std::cout << "Loaded " << labels.size() << " labels" << std::endl;
        }

        VisionDetect detector;
        if (!detector.Init(det_engine_path))
        {
            std::cout << "Init detect model failed\n";
            return -1;
        }

        VisionSegment segmentor;
        if (!segmentor.Init(seg_engine_path))
        {
            std::cout << "Init segment model failed\n";
            return -1;
        }

        CTempMatch tempmatcher;

        std::cout << "Model loaded success\n";

        uint64_t id_base = 0;

        std::vector<std::string> images;

        if (fs::is_regular_file(input_path))
        {
            images.push_back(input_path);
        }
        else
        {
            images = GetImages(input_path);
        }

        if (images.empty())
        {
            std::cout << "No images found\n";
            return -1;
        }

        for (auto& path : images)
        {
            std::cout << "Processing: " << path << std::endl;

            cv::Mat img = cv::imread(path);

            if (img.empty())
            {
                std::cout << "read image failed\n";
                continue;
            }

            // 执行模板匹配
            std::cout<<"start TemplateMatch ..."<<std::endl;
            std::vector<TemplateMatchResult> MResult;
            int ret = tempmatcher.TemplateMatch(img, template_img, cv::Rect(0, 0, img.cols, img.rows), 0, 180, MResult);
            if (0 == ret)
                std::cout<<"TemplateMatch success ..."<<std::endl;
            else
                std::cout<<"TemplateMatch failed ..."<<std::endl;

            //检测焊点
            std::vector<CheckBox> solderings;
            detector.Run(
                img,
                0.25,       // conf_thresh
                solderings,
                id_base,
                false
            );

            // debug和可视化
            std::cout << "detections: " << solderings.size() << std::endl;
                    // 创建可视化图像
            cv::Mat viz_img = img.clone();
            DrawBoxes(viz_img, solderings, path, output_path);

            // 对每个检测框进行分割
            std::vector<std::vector<CheckBox>> all_segmentations;
            int crop_size = 384;//暂定384 320
            if (!solderings.empty()) {
                segmentor.RunWithBoxes(
                    img,
                    solderings,
                    0.25,       // conf_thresh
                    all_segmentations,
                    id_base,
                    crop_size        // crop_size
                );

                // 可视化
                viz_img = img.clone();
                visualizeSegmentationROI(viz_img, solderings, all_segmentations, path, output_path, crop_size, labels);
            }

            // 联系焊点和导线从属关系
            std::vector<SolderWire> results;
            matchWiresToSolder(img, solderings, all_segmentations, results, cv::Point2f(MResult.at(0).maxLoc.x, MResult.at(0).maxLoc.y), 1);

            // 匹配结果可视化
            visualizeMatchResult(img, results, output_path, "match_test"+fs::path(path).filename().string());

            // 和参考图片结果做比较
            std::vector<MatchPair> matchResult = matchSolders(img, template_img, reference_results, results, MResult.at(0), MATCH_SOLDERS_IOU_THRE);
            
            // 可视化和模板匹配结果
            visualizeFinalResult(img, reference_results, results, matchResult, output_path, fs::path(path).filename().string());
        }
    }

    std::cout << "Finished\n";

    return 0;
}