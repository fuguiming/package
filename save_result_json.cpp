#include "save_result_json.h"

void save_results_binary(const std::vector<SolderWire>& results, const std::string& path)
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;

    // 写总数
    ofs << results.size() << "\n";

    for (const auto& sw : results)
    {
        const auto& s = sw.solder_info;

        // ===== 写焊点 =====
        ofs << s.left << " " << s.top << " " << s.right << " " << s.bottom << " "
            << s.cls << " " << s.score << " " << s.id << " "
            << s.mask_width << " " << s.mask_height << "\n";

        // mask
        ofs << s.mask.size() << "\n";
        for (auto v : s.mask) ofs << v << " ";
        ofs << "\n";

        // ===== 写 wire 数量 =====
        ofs << sw.wires.size() << "\n";

        // ===== 写每条 wire =====
        for (const auto& w : sw.wires)
        {
            ofs << w.left << " " << w.top << " " << w.right << " " << w.bottom << " "
                << w.cls << " " << w.score << " " << w.id << " "
                << w.mask_width << " " << w.mask_height << " "
                << w.color[0] << " " << w.color[1] << " " << w.color[2] << "\n";

            ofs << w.mask.size() << "\n";
            for (auto v : w.mask) ofs << v << " ";
            ofs << "\n";
        }
    }

    ofs.close();
}

int load_results_binary(const std::string& path, std::vector<SolderWire>& results)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) return -1;

    size_t num_results;
    ifs >> num_results;

    results.clear();
    results.reserve(num_results);

    for (size_t i = 0; i < num_results; ++i)
    {
        SolderWire sw;

        // ===== 读焊点 =====
        CheckBox& s = sw.solder_info;

        ifs >> s.left >> s.top >> s.right >> s.bottom
            >> s.cls >> s.score >> s.id
            >> s.mask_width >> s.mask_height;

        size_t mask_size;
        ifs >> mask_size;
        s.mask.resize(mask_size);

        for (size_t j = 0; j < mask_size; ++j)
            ifs >> s.mask[j];

        // ===== 读 wires =====
        size_t wire_num;
        ifs >> wire_num;

        sw.wires.resize(wire_num);

        for (size_t k = 0; k < wire_num; ++k)
        {
            CheckBox& w = sw.wires[k];

            ifs >> w.left >> w.top >> w.right >> w.bottom
                >> w.cls >> w.score >> w.id
                >> w.mask_width >> w.mask_height >> w.color[0] >> w.color[1] >> w.color[2];

            size_t w_mask_size;
            ifs >> w_mask_size;
            w.mask.resize(w_mask_size);

            for (size_t j = 0; j < w_mask_size; ++j)
                ifs >> w.mask[j];
        }

        results.push_back(std::move(sw));
    }

    ifs.close();
    return 0;
}

void covert2template(std::vector<SolderWire> &results, const int template_left, const int template_top)
{
    for (auto& sw : results)
    {
        CheckBox& s = sw.solder_info;
        s.left -= template_left;
        s.top -= template_top;
        s.right -= template_left;
        s.bottom -= template_top;

        for (auto& w : sw.wires)
        {
            w.left -= template_left;
            w.top -= template_top;
            w.right -= template_left;
            w.bottom -= template_top;
        }
    }
}