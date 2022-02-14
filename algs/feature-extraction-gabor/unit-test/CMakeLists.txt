list(APPEND FEATURE_EXTRACTION_GABOR_UNIT_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/filter_banks.cc
        ${CMAKE_CURRENT_LIST_DIR}/patch_assembly.cc
        ${CMAKE_CURRENT_LIST_DIR}/math_tests.cc
        )

#[[ A 'filter_banks_gabor_6or_4f.bin is a binary output of a 3D filter bank from 'EO_tools' utility, developed by UPB.
    It consists of multiple 2D float arrays/matrices with differing dimensions, data is represented by 32 bit floats.
    It is converted by the following command:
    objcopy --input binary --output elf64-x86-64 --binary-architecture i386:x86-64 filter_banks_gabor_6or_4f.bin filter_banks_gabor_6or_4f.o
    Listing symbols:
    objdump -t filter_banks_gabor_6or_4f.o - it contains '_binary_filter_banks_gabor_6or_4f_bin_{start|end|size}' symbols
]]
set(PROJECT_EMBED_OBJ_FILES "")
set(PROJECT_EMBED_FILES ${CMAKE_CURRENT_LIST_DIR}/filter_banks_gabor_6or_4f.bin ${CMAKE_CURRENT_LIST_DIR}/patch_input_band1_0_0_50_50.img
        ${CMAKE_CURRENT_LIST_DIR}/patch_padded_band1_0_0_77_77.img)
foreach(FILENAME ${PROJECT_EMBED_FILES})
    get_filename_component(INPUT_BIN ${FILENAME} NAME)
    get_filename_component(OUTPUT_OBJECT_STEM ${FILENAME} NAME_WLE)
    set(OUTPUT_OBJECT ${OUTPUT_OBJECT_STEM}.o)
    get_filename_component(FILEPATH_ONLY ${FILENAME} PATH)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJECT}
            WORKING_DIRECTORY ${FILEPATH_ONLY}
            COMMAND ${CMAKE_OBJCOPY}
            ARGS -I binary -O elf64-x86-64 -B i386:x86-64 ${INPUT_BIN} ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJECT} )
    list(APPEND PROJECT_EMBED_OBJ_FILES ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJECT})
endforeach(FILENAME)

add_executable(feature-extraction-gabor-unit-test ${FEATURE_EXTRACTION_GABOR_UNIT_TEST_SOURCES} ${PROJECT_EMBED_OBJ_FILES})

target_link_libraries(feature-extraction-gabor-unit-test
        PRIVATE
        alus-gmock-main-static
        gdal
        feature-extraction-gabor-static
        )
set_target_properties(feature-extraction-gabor-unit-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )
