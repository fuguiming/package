#ifndef CHECKBOX_H
#define CHECKBOX_H

#include <vector>
#include <cstdint>

struct CheckBox {
    int left, top, right, bottom;  // 边界框坐标
    int cls;                        // 类别ID
    float score;                     // 置信度
    uint64_t id;                     // 唯一ID
    std::vector<float> mask;         // 分割掩码数据
    int mask_width;                  // 掩码宽度
    int mask_height;                 // 掩码高度
    cv::Scalar color;                // 掩膜颜色

    CheckBox() : left(0), top(0), right(0), bottom(0), cls(0), score(0), id(0), 
                 mask_width(0), mask_height(0) {}

    CheckBox(int l, int t, int r, int b, int c, float s, uint64_t tid)
        : left(l), top(t), right(r), bottom(b), cls(c), score(s), id(tid),
          mask_width(0), mask_height(0) {}

    // 获取宽度
    int width() const { return right - left; }
    // 获取高度
    int height() const { return bottom - top; }
    // 获取中心x坐标
    int center_x() const { return (left + right) / 2; }
    // 获取中心y坐标
    int center_y() const { return (top + bottom) / 2; }
    // 获取xyxy格式的边界框
    std::vector<int> xyxy() const { return {left, top, right, bottom}; }
};

#endif // CHECKBOX_H