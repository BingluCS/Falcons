set(FALCONS_SOURCE_FILE "${CMAKE_SOURCE_DIR}/tools/falcons/falcons.cpp")
set(FALCONS_TARGET_NAME falcons)

message(STATUS "Target = ${FALCONS_TARGET_NAME}")
add_executable(${FALCONS_TARGET_NAME} ${FALCONS_SOURCE_FILE})
target_link_libraries(${FALCONS_TARGET_NAME} PRIVATE Falcons)
install(TARGETS ${FALCONS_TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES testfloat_8_8_128.dat DESTINATION ${CMAKE_INSTALL_DATADIR}/Falcons)

#add_subdirectory(demo)
