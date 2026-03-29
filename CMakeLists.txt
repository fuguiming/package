cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0091 NEW)

project(VisionDetectDemo LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(TensorRT-YOLO REQUIRED)

function(set_target_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /Oi /fp:fast>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od /Ob0 /Zi /RTC1>
        )

        set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY
            "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>"
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -march=native -flto=auto -DNDEBUG>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g3 -fno-omit-frame-pointer -fno-inline>
        )

        target_link_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -flto=auto>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g3>
        )
    endif()

    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    )
endfunction()

add_executable(detect
    main.cpp
    vision_detect.cpp
    vision_segment.cpp
    wire_analyzer.cpp
    CTempMatch.cpp
    save_result_json.cpp
)

link_directories(
    E:/win_tensorrt_yolo/TensorRT-YOLO/examples/package/intel64_IPP
)

target_include_directories(detect PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT-YOLO_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(detect PRIVATE
    ${OpenCV_LIBS}
    ${TensorRT-YOLO_LIBs}
    E:/win_tensorrt_yolo/TensorRT-YOLO/examples/package/intel64_IPP/ippcore.lib
    E:/win_tensorrt_yolo/TensorRT-YOLO/examples/package/intel64_IPP/ippi.lib
    E:/win_tensorrt_yolo/TensorRT-YOLO/examples/package/intel64_IPP/ipps.lib
    E:/win_tensorrt_yolo/TensorRT-YOLO/examples/package/intel64_IPP/ippcv.lib
)

set_target_compile_options(detect)

set_target_properties(detect PROPERTIES
    OUTPUT_NAME "detect"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin"
)