#include "vision_detect.h"
#include <cmath>

VisionDetect::~VisionDetect()
{
    Reset();
}

bool VisionDetect::Init(const std::string &engine_path)
{
    Reset();

    trtyolo::InferOption option;
    option.enableSwapRB();

    // model_ = new trtyolo::DetectModel(engine_path, option);
    model_ = std::make_unique<trtyolo::DetectModel>(engine_path, option);

    return model_ != nullptr;
}

void VisionDetect::Reset()
{
    if (model_) {
        model_.reset();
    }
}

// PCB板区域检测函数
std::vector<PCBRegion> 
VisionDetect::detect_pcb_regions(
    const cv::Mat &img, 
    float min_area_ratio)
{
    int img_height = img.rows;
    int img_width  = img.cols;
    int total_area = img_height * img_width;

    // ===== 1. 分离通道 =====
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    cv::Mat b = channels[0];
    cv::Mat g = channels[1];
    cv::Mat r = channels[2];

    // ===== 2. 转 float32 =====
    cv::Mat b_f, g_f, r_f;
    b.convertTo(b_f, CV_32F);
    g.convertTo(g_f, CV_32F);
    r.convertTo(r_f, CV_32F);

    // ===== 3. 绿色增强 =====
    cv::Mat r_plus_b, half_rb, enhanced;
    cv::add(r_f, b_f, r_plus_b);
    half_rb = r_plus_b * 0.5f;

    cv::subtract(g_f, half_rb, enhanced);

    // np.clip(..., 0, 255)
    cv::max(enhanced, 0, enhanced);

    cv::Mat green_enhanced;
    enhanced.convertTo(green_enhanced, CV_8U);

    // ===== 4. OTSU =====
    cv::Mat thresh;
    cv::threshold(
        green_enhanced,
        thresh,
        0,
        255,
        cv::THRESH_BINARY | cv::THRESH_OTSU
    );

    // ===== 5. 形态学 =====
    cv::Mat kernel5 = cv::Mat::ones(5, 5, CV_8U);
    cv::Mat kernel9 = cv::Mat::ones(9, 9, CV_8U);

    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel5, cv::Point(-1,-1), 2);
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN,  kernel5, cv::Point(-1,-1), 1);

    cv::erode(thresh, thresh, kernel9, cv::Point(-1,-1), 2);
    cv::dilate(thresh, thresh, kernel9, cv::Point(-1,-1), 2);

    // debug保存（等价 Python cv2.imwrite）
    // cv::imwrite("thresh.jpg", thresh);

    // ===== 6. 找轮廓 =====
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        thresh,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    std::cout << "contours: " << contours.size() << std::endl;

    if (contours.empty())
    {
        return std::vector<PCBRegion>{};
    }

    int min_area = static_cast<int>(total_area * min_area_ratio);

    std::vector<std::vector<cv::Point>> filtered_contours;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);

        cv::Rect rect = cv::boundingRect(contour);

        if (area > min_area && rect.width < 4000 && rect.height < 4000)
        {
            filtered_contours.push_back(contour);
        }
    }

    if (filtered_contours.empty())
    {
        return std::vector<PCBRegion>{};
    }

    // ===== 7. 输出区域 =====
    std::vector<PCBRegion> final_regions;

    for (const auto &contour : filtered_contours)
    {
        cv::Rect rect = cv::boundingRect(contour);

        final_regions.emplace_back(
            rect.x,
            rect.y,
            rect.x + rect.width,
            rect.y + rect.height
        );
    }

    return final_regions;
}

// 检查切片是否与任何PCB区域重叠
bool 
VisionDetect::tile_overlaps_with_pcb(
    int tile_x1, 
    int tile_y1, 
    int tile_x2, 
    int tile_y2, 
    const std::vector<PCBRegion>& pcb_regions, 
    float overlap_threshold) 
{
    if (pcb_regions.empty()) return true; // 如果没有PCB区域，默认处理所有切片
    
    int tile_area = (tile_x2 - tile_x1) * (tile_y2 - tile_y1);
    
    for (const auto& pcb : pcb_regions) {
        // 计算重叠区域
        int overlap_x1 = std::max(tile_x1, pcb.x1);
        int overlap_y1 = std::max(tile_y1, pcb.y1);
        int overlap_x2 = std::min(tile_x2, pcb.x2);
        int overlap_y2 = std::min(tile_y2, pcb.y2);
        
        if (overlap_x2 > overlap_x1 && overlap_y2 > overlap_y1) {
            int overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1);
            float overlap_ratio = static_cast<float>(overlap_area) / tile_area;
            
            if (overlap_ratio > overlap_threshold) {
                return true; // 与PCB区域有足够重叠
            }
        }
    }
    
    return false;
}

std::vector<BoundingBox>
VisionDetect::merge_boundary_boxes_by_distance(
    const std::vector<BoundingBox> &boxes,
    float distance_threshold)
{
    std::vector<BoundingBox> merged;
    std::vector<bool> used(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); i++)
    {
        if (used[i])
            continue;

        BoundingBox m = boxes[i];
        used[i] = true;

        int cx1 = (m.x1 + m.x2) / 2;
        int cy1 = (m.y1 + m.y2) / 2;

        for (size_t j = i + 1; j < boxes.size(); j++)
        {
            if (used[j])
                continue;

            if (boxes[j].cls != m.cls)
                continue;

            int cx2 = (boxes[j].x1 + boxes[j].x2) / 2;
            int cy2 = (boxes[j].y1 + boxes[j].y2) / 2;

            float d = std::sqrt(
                (cx1 - cx2) * (cx1 - cx2) +
                (cy1 - cy2) * (cy1 - cy2));

            if (d < distance_threshold)
            {
                m.x1 = std::min(m.x1, boxes[j].x1);
                m.y1 = std::min(m.y1, boxes[j].y1);
                m.x2 = std::max(m.x2, boxes[j].x2);
                m.y2 = std::max(m.y2, boxes[j].y2);

                used[j] = true;
            }
        }

        merged.push_back(m);
    }

    return merged;
}

float IoU(const BoundingBox& a, const BoundingBox& b)
{
    int xx1 = std::max(a.x1, b.x1);
    int yy1 = std::max(a.y1, b.y1);
    int xx2 = std::min(a.x2, b.x2);
    int yy2 = std::min(a.y2, b.y2);

    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);

    float inter = w * h;
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter / (areaA + areaB - inter + 1e-6f);
}

std::vector<BoundingBox> 
VisionDetect::NMS(std::vector<BoundingBox>& boxes, float iou_thresh)
{
    std::vector<BoundingBox> result;
    // 1. 按 score 排序（从大到小）
    std::sort(boxes.begin(), boxes.end(),
        [](const BoundingBox& a, const BoundingBox& b)
        {
            return a.score > b.score;
        });

    std::vector<bool> removed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (removed[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (removed[j]) continue;
            // 同类别才做抑制（关键）
            if (boxes[i].cls != boxes[j].cls)
                continue;

            if (IoU(boxes[i], boxes[j]) > iou_thresh)
            {
                removed[j] = true;
            }
        }
    }
    return result;
}

void VisionDetect::Run(const cv::Mat &img,
                       double conf_thresh,
                       std::vector<CheckBox> &out,
                       uint64_t &id_base,
                       bool append)
{
    if (!model_)
        return;

    if (!append)
        out.clear();

    int tiles_x = 5;//全图5120*5120，截取1280*1280大小区域，step为960，一共截取5*5个区域，确保有重叠区域
    int tiles_y = 5;

    int img_w = img.cols;
    int img_h = img.rows;

    if (img_w != img_h && img_w != 5120)
    {
        std::cout<<("Error: img size is not 5120!")<<std::endl;
        return;
    }

    int tile_w = 1280;
    int tile_h = 1280;
    int step_w = 960;
    int step_h = 960;

    auto pcb_regions = detect_pcb_regions(img);
    // for (int i = 0; i < pcb_regions.size(); i ++)
    //     std::cout<<pcb_regions.at(i).x1<<" "<<pcb_regions.at(i).y1<<" "<<pcb_regions.at(i).x2<<" " <<pcb_regions.at(i).y2<<std::endl;

    std::vector<BoundingBox> all_boxes;

    for (int r = 0; r < tiles_y; r++)
    {
        for (int c = 0; c < tiles_x; c++)
        {
            int x1 = c * step_w;
            int y1 = r * step_h;

            int x2 = (c == tiles_x - 1) ? img_w : x1 + tile_w;
            int y2 = (r == tiles_y - 1) ? img_h : y1 + tile_h;

            if (!tile_overlaps_with_pcb(x1, y1, x2, y2, pcb_regions))
                continue;

            cv::Mat tile = img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();//需要clone，直接引用内存不连续
            // cv::imwrite("tile.jpg", tile);
            // cv::Mat tile = cv::imread("E:\\fgm_workspace\\win_tensorrt_yolo\\TensorRT-YOLO\\imgs\\pcb\\Image_20260212174444792_grid_3_0.jpg", cv::IMREAD_COLOR);
            // std::cout << tile.isContinuous() << std::endl;
            trtyolo::Image trt_img(tile.data, tile.cols, tile.rows);

            auto result = model_->predict(trt_img);
            // std::cout<<"result num: "<<result.num<<std::endl;
            for (size_t i = 0; i < result.num; i++)
            {
                // 焊点得分不满足要求
                if (result.scores[i] < conf_thresh)
                    continue;

                // std::cout <<"soldering width: "<<result.boxes[i].right - result.boxes[i].left<<" soldering height: "<<result.boxes[i].bottom - result.boxes[i].top<<std::endl;

                BoundingBox box;

                box.x1 = result.boxes[i].left + x1;
                box.y1 = result.boxes[i].top + y1;
                box.x2 = result.boxes[i].right + x1;
                box.y2 = result.boxes[i].bottom + y1;
                box.cls = result.classes[i];
                box.score = result.scores[i];

                all_boxes.push_back(box);
            }
        }
    }

    //进行iou过滤
    auto iou_result = NMS(all_boxes, 0.1);
    //合并近距离结果
    auto merged = merge_boundary_boxes_by_distance(iou_result, 20);

    for (auto &b : merged)
    {
        // std::cout<<"x1: "<<b.x1<<" y1: "<<b.y1<<" x2: "<<b.x2<<" y2: "<<b.y2<<std::endl;
        // 焊点宽高不满足要求
        if ((b.x2 - b.x1) < solder_width_thre || (b.y2 - b.y1) < solder_height_thre)
            continue;

        CheckBox cb;

        cb.left = b.x1;
        cb.top = b.y1;
        cb.right = b.x2;
        cb.bottom = b.y2;

        cb.cls = b.cls;
        cb.score = b.score;

        cb.id = id_base++;

        out.push_back(cb);
    }
}