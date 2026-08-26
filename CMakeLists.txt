cmake_minimum_required(VERSION 3.25)

project(OpenYourBox VERSION 1.0.0 LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

set(OPENYOURBOX_JUCE_PATH "" CACHE PATH "Path to a JUCE 9.x source tree")
if(OPENYOURBOX_JUCE_PATH)
    add_subdirectory("${OPENYOURBOX_JUCE_PATH}" "${CMAKE_BINARY_DIR}/JUCE")
else()
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 9.0.1
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(JUCE)
endif()

find_package(Torch REQUIRED)
if(NOT Python3_EXECUTABLE
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# LibTorch's CMake config injects GNU-ld-only flags (--no-as-needed) that Apple's linker rejects.
if(APPLE)
    list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "(^-Wl,|^--no-as-needed|^--as-needed)")

    foreach(_torch_tgt torch_cpu_library torch_library)
        if(TARGET ${_torch_tgt})
            get_target_property(_ill ${_torch_tgt} INTERFACE_LINK_LIBRARIES)
            if(_ill)
                string(REPLACE "-Wl,--no-as-needed," "" _ill "${_ill}")
                string(REPLACE " -Wl,--as-needed" "" _ill "${_ill}")
                set_target_properties(${_torch_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_ill}")
            endif()
        endif()
    endforeach()
endif()

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9b
    GIT_SHALLOW TRUE)
FetchContent_Declare(
    imgui_node_editor
    GIT_REPOSITORY https://github.com/thedmd/imgui-node-editor.git
    GIT_TAG 021aa0ea4da13fed864bafb2a92d4c5205076866
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui imgui_node_editor)

# imgui 1.91+ already provides float * ImVec2 when IMGUI_DEFINE_MATH_OPERATORS is
# set. imgui-node-editor 021aa0e still emits its own copy.
set(_openyourbox_extra_math_h "${imgui_node_editor_SOURCE_DIR}/imgui_extra_math.h")
set(_openyourbox_extra_math_inl "${imgui_node_editor_SOURCE_DIR}/imgui_extra_math.inl")
file(READ "${_openyourbox_extra_math_h}" _openyourbox_extra_math_h_src)
if(NOT _openyourbox_extra_math_h_src MATCHES "OPENYOURBOX_SKIP_FLOAT_IMVEC2_PRODUCT")
    string(REPLACE
        "inline ImVec2 operator*(const float lhs, const ImVec2& rhs);"
        "# if IMGUI_VERSION_NUM < 19100 /* OPENYOURBOX_SKIP_FLOAT_IMVEC2_PRODUCT */\ninline ImVec2 operator*(const float lhs, const ImVec2& rhs);\n# endif"
        _openyourbox_extra_math_h_src "${_openyourbox_extra_math_h_src}")
    file(WRITE "${_openyourbox_extra_math_h}" "${_openyourbox_extra_math_h_src}")
    file(READ "${_openyourbox_extra_math_inl}" _openyourbox_extra_math_inl_src)
    string(REPLACE
        "inline ImVec2 operator*(const float lhs, const ImVec2& rhs)\n{\n    return ImVec2(lhs * rhs.x, lhs * rhs.y);\n}"
        "# if IMGUI_VERSION_NUM < 19100 /* OPENYOURBOX_SKIP_FLOAT_IMVEC2_PRODUCT */\ninline ImVec2 operator*(const float lhs, const ImVec2& rhs)\n{\n    return ImVec2(lhs * rhs.x, lhs * rhs.y);\n}\n# endif"
        _openyourbox_extra_math_inl_src "${_openyourbox_extra_math_inl_src}")
    file(WRITE "${_openyourbox_extra_math_inl}" "${_openyourbox_extra_math_inl_src}")
endif()

# Public NavigateTo(..., zoomIn=false) only recentres. Exact mode keeps zoom.
set(_openyourbox_ne_internal "${imgui_node_editor_SOURCE_DIR}/imgui_node_editor_internal.h")
file(READ "${_openyourbox_ne_internal}" _openyourbox_ne_internal_src)
if(NOT _openyourbox_ne_internal_src MATCHES "OPENYOURBOX_NAVIGATE_EXACT")
    string(REPLACE
        "auto zoomMode = zoomIn ? NavigateAction::ZoomMode::WithMargin : NavigateAction::ZoomMode::None;"
        "auto zoomMode = zoomIn ? NavigateAction::ZoomMode::WithMargin : NavigateAction::ZoomMode::Exact; /* OPENYOURBOX_NAVIGATE_EXACT */"
        _openyourbox_ne_internal_src "${_openyourbox_ne_internal_src}")
    file(WRITE "${_openyourbox_ne_internal}" "${_openyourbox_ne_internal_src}")
endif()

add_library(openyourbox_imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    OpenYourBox/Source/ui/ImGuiOpenGLBackend.cpp
    "${imgui_node_editor_SOURCE_DIR}/crude_json.cpp"
    "${imgui_node_editor_SOURCE_DIR}/imgui_canvas.cpp"
    "${imgui_node_editor_SOURCE_DIR}/imgui_node_editor.cpp"
    "${imgui_node_editor_SOURCE_DIR}/imgui_node_editor_api.cpp")
target_include_directories(openyourbox_imgui PUBLIC
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
    "${imgui_node_editor_SOURCE_DIR}")
target_compile_definitions(openyourbox_imgui PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
target_link_libraries(openyourbox_imgui PUBLIC juce::juce_opengl)

juce_add_plugin(OpenYourBox
    COMPANY_NAME "Allendia"
    BUNDLE_ID "com.allendia.openyourbox"
    PLUGIN_MANUFACTURER_CODE Alnd
    PLUGIN_CODE OyBx
    FORMATS AU VST3
    PRODUCT_NAME "OpenYourBox"
    DESCRIPTION "Live modular temporal convolution audio effect"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES Fx Distortion)

juce_generate_juce_header(OpenYourBox)

juce_add_binary_data(OpenYourBoxResources
    SOURCES Backend/freeze_worker.py Backend/train_worker.py)

target_sources(OpenYourBox PRIVATE
    OpenYourBox/Source/PluginEditor.cpp
    OpenYourBox/Source/PluginProcessor.cpp
    OpenYourBox/Source/capture/CapturePairing.cpp
    OpenYourBox/Source/capture/CaptureRecorder.cpp
    OpenYourBox/Source/dsp/LiveGraphEngine.cpp
    OpenYourBox/Source/dsp/LiveGraphPublisher.cpp
    OpenYourBox/Source/dsp/LookbackBuffer.cpp
    OpenYourBox/Source/dsp/NoiseSynthesizer.cpp
    OpenYourBox/Source/dsp/PqmfBank.cpp
    OpenYourBox/Source/dsp/RateConv.cpp
    OpenYourBox/Source/dsp/TCNModel.cpp
    OpenYourBox/Source/dsp/TorchScriptBlackBox.cpp
    OpenYourBox/Source/dsp/VariationalBottleneck.cpp
    OpenYourBox/Source/dsp/WeightLoader.cpp
    OpenYourBox/Source/dsp/WeightRandomizer.cpp
    OpenYourBox/Source/freeze/FreezeCoordinator.cpp
    OpenYourBox/Source/graph/NodeGraph.cpp
    OpenYourBox/Source/graph/NodeRenderer.cpp
    OpenYourBox/Source/graph/RaveLayouts.cpp
    OpenYourBox/Source/library/CopyrightAcknowledgment.cpp
    OpenYourBox/Source/library/TrainingLibrary.cpp
    OpenYourBox/Source/params/ParamLayout.cpp
    OpenYourBox/Source/train/TrainCoordinator.cpp
    OpenYourBox/Source/ui/CaptureSamplesPanel.cpp
    OpenYourBox/Source/ui/CopyrightModal.cpp
    OpenYourBox/Source/ui/ErrorModal.cpp
    OpenYourBox/Source/ui/ImGuiHost.cpp
    OpenYourBox/Source/ui/InfoPanel.cpp
    OpenYourBox/Source/ui/RandomizeButton.cpp
    OpenYourBox/Source/ui/TrainPanel.cpp
    OpenYourBox/Source/ui/TrainingLibraryPanel.cpp)

target_include_directories(OpenYourBox PRIVATE
    OpenYourBox/Source
    "${TORCH_INCLUDE_DIRS}")

target_compile_definitions(OpenYourBox PUBLIC
    OPENYOURBOX_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_compile_options(OpenYourBox PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wall -Wextra -Wpedantic>)

target_link_libraries(OpenYourBox PRIVATE
    OpenYourBoxResources
    openyourbox_imgui
    "${TORCH_LIBRARIES}"
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_opengl
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags)

if(APPLE)
    target_sources(OpenYourBox PRIVATE
        OpenYourBox/Source/capture/HostTransport.mm)
    set(OPENYOURBOX_USER_VST3_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")
    set_target_properties(OpenYourBox PROPERTIES
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")
    add_custom_command(TARGET OpenYourBox_VST3 POST_BUILD
        COMMAND codesign --force --deep --sign -
            "$<TARGET_BUNDLE_DIR:OpenYourBox_VST3>"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${OPENYOURBOX_USER_VST3_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "$<TARGET_BUNDLE_DIR:OpenYourBox_VST3>"
            "${OPENYOURBOX_USER_VST3_DIR}/OpenYourBox.vst3"
        VERBATIM)
    add_custom_command(TARGET OpenYourBox_AU POST_BUILD
        COMMAND codesign --force --deep --sign -
            "$<TARGET_BUNDLE_DIR:OpenYourBox_AU>"
        VERBATIM)
endif()

include(CTest)
if(BUILD_TESTING)
    juce_add_console_app(OpenYourBoxTests PRODUCT_NAME "OpenYourBox Tests")
    target_sources(OpenYourBoxTests PRIVATE
        Tests/TCNModelTests.cpp
        OpenYourBox/Source/dsp/TCNModel.cpp)
    target_include_directories(OpenYourBoxTests PRIVATE
        OpenYourBox/Source
        "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(OpenYourBoxTests PRIVATE
        "${TORCH_LIBRARIES}"
        juce::juce_core
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
    add_test(NAME OpenYourBoxTCNModel COMMAND OpenYourBoxTests)

    juce_add_console_app(OpenYourBoxProcessorTests PRODUCT_NAME "OpenYourBox Processor Tests")
    target_sources(OpenYourBoxProcessorTests PRIVATE Tests/ProcessorIntegrationTests.cpp)
    target_include_directories(OpenYourBoxProcessorTests BEFORE PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/OpenYourBox_artefacts/JuceLibraryCode"
        OpenYourBox/Source
        "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(OpenYourBoxProcessorTests PRIVATE
        OpenYourBox
        OpenYourBoxResources
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_opengl
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
    add_test(NAME OpenYourBoxProcessor COMMAND OpenYourBoxProcessorTests)

    juce_add_console_app(OpenYourBoxLiveGraphTests PRODUCT_NAME "OpenYourBox Live Graph Tests")
    target_sources(OpenYourBoxLiveGraphTests PRIVATE Tests/LiveGraphEngineTests.cpp)
    target_include_directories(OpenYourBoxLiveGraphTests BEFORE PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/OpenYourBox_artefacts/JuceLibraryCode"
        OpenYourBox/Source
        "${TORCH_INCLUDE_DIRS}")
    target_link_libraries(OpenYourBoxLiveGraphTests PRIVATE
        OpenYourBox
        OpenYourBoxResources
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_opengl
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
    add_test(NAME OpenYourBoxLiveGraph COMMAND OpenYourBoxLiveGraphTests)

    add_test(NAME OpenYourBoxFreezeWorker
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/Tests/test_freeze_worker.py")
    add_test(NAME OpenYourBoxTrainWorker
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/Tests/test_train_worker.py")
endif()
